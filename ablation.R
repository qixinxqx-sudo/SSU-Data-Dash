# =============================================================================
# SSU Data Dash 2026 - EXTRA EXPERIMENTS (for the report and the video)
#
# HOW TO RUN (RStudio):
#   1. Save this file in the SAME folder as your main script (the folder that
#      already contains the "data" folder with listings.csv).
#   2. Open it in RStudio and click "Source".
#   3. Wait about 30-60 minutes. Messages print in the Console.
#   4. At the end, copy everything the Console prints from the line
#      "ABLATION RESULTS" to the end, and paste it into the chat.
#
# WHAT IT DOES
#   Re-runs your OFF-MARKET model six months x 5 times, each time changing ONE
#   thing, to see which of your choices helped and which did not:
#     full        = your model as it is (reference, with 100 trees to save time)
#     trend_on    = extrapolate the market-price trend instead of holding it flat
#     no_text     = drop the keyword features taken from the listing description
#     no_detrend  = do NOT remove the market-price index from the target
#     raw_price   = predict dollars instead of log price
#   It also lists the communities the model gets most wrong, and checks whether
#   homes started selling for less relative to list price (this would explain
#   the +4.7% bias of the list-aware model).
#   Nothing is overwritten: it only writes reports/ablation_*.csv
# =============================================================================

SEED            <- 2026
NUM_TREES       <- 100      # half of the main script, so this finishes sooner
MIN_NODE        <- 10
SAMPLE_FRACTION <- 0.4
GAP_MONTHS      <- 2
VAL_MONTHS      <- seq(as.Date("2025-12-01"), as.Date("2026-05-01"), by = "month")
DATA_DIR        <- "data"

need <- c("ranger", "dplyr", "readr", "lubridate", "stringr")
miss <- need[!sapply(need, requireNamespace, quietly = TRUE)]
if (length(miss)) install.packages(miss, repos = "https://cloud.r-project.org")
suppressPackageStartupMessages({
  library(ranger); library(dplyr); library(readr); library(lubridate); library(stringr)
})
if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
  p <- tryCatch(dirname(rstudioapi::getSourceEditorContext()$path), error = function(e) "")
  if (nzchar(p) && dir.exists(p)) setwd(p)
}
dir.create("reports", showWarnings = FALSE)
set.seed(SEED)
msg <- function(...) cat(format(Sys.time(), "%H:%M:%S"), "-", ..., "\n")
msg("Working folder:", getwd())

f <- file.path(DATA_DIR, "listings.csv")
if (!file.exists(f)) stop("Cannot find data/listings.csv. Save this script in the same folder as your main script, then Source it again.")
listings_raw <- read_csv(f, guess_max = 200000, show_col_types = FALSE, progress = FALSE)
listings_raw$date_sold   <- as.Date(listings_raw$date_sold)
listings_raw$date_listed <- as.Date(listings_raw$date_listed)
listings_raw <- listings_raw %>% filter(!is.na(sold_price), sold_price > 0, !is.na(date_sold))
msg("Training sales:", nrow(listings_raw))

# ---------------- feature building: copied unchanged from the main script -------
CAT_COLS <- c("house_type_name", "house_style", "municipality", "community")
lv <- lapply(listings_raw[CAT_COLS], function(x) sort(unique(as.character(x[!is.na(x)]))))
mode_level <- sapply(CAT_COLS, function(cn) names(sort(table(listings_raw[[cn]]), decreasing = TRUE))[1])
area_mid_of <- function(x) {
  x <- as.character(x)
  lo <- suppressWarnings(as.numeric(sub("-.*", "", x)))
  hi <- suppressWarnings(as.numeric(sub(".*-", "", x)))
  (lo + hi) / 2
}
area_by_type <- listings_raw %>%
  mutate(a = area_mid_of(house_area)) %>%
  group_by(house_type_name) %>% summarise(m = median(a, na.rm = TRUE), .groups = "drop")
area_global <- median(area_mid_of(listings_raw$house_area), na.rm = TRUE)

PAT_RENO <- "fully (redone|renovated|updated)|completely (redone|renovated)|top to bottom|turnkey|nothing (left )?to do|reimagined|spa-like|quartz|custom kitchen|designer light|heated bathroom|wide-plank|built-in appliances|high-efficiency|new roof|new windows and doors|stainless steel|frameless glass|large island|custom (closet|window)"
PAT_UPD  <- "newer (windows|furnace|flooring|laminate)|refreshed|refinished|fresh(ly)? paint|updated|upgraded|new (countertops|vanity|bathroom)|vinyl plank|renovated basement|smooth ceilings|pot lights"
PAT_NEED <- "as-is|needs|requires|cosmetic overhaul|fixer|renovator|your vision|contractor|sweat equity|worn|tired|dated|near the end|end of (their|its) life|blank canvas|project|waterproofing|significant repair|overgrown|older (appliances|furnace|windows|bathrooms)|bring your"
PAT_ORIG <- "original|clean and cared|well-maintained|well maintained|sound and liveable|move-in ready"
PAT_MOT  <- "motivated|quick closing|flexible (on )?closing|invites (your )?offers|open to offers"
PAT_AMEN <- "concierge|gym|party room|rooftop terrace|guest suite|locker|ensuite laundry|playground|visitor parking|underground parking"

prep <- function(df) {
  df$date_listed <- as.Date(df$date_listed)
  x <- as.character(df$house_area)
  df$area_mid <- area_mid_of(x)
  df$area_is_range <- as.integer(grepl("-", x, fixed = TRUE))
  fill <- area_by_type$m[match(as.character(df$house_type_name), area_by_type$house_type_name)]
  fill[is.na(fill)] <- area_global
  df$area_mid[is.na(df$area_mid)] <- fill[is.na(df$area_mid)]
  df$log_area <- log(pmax(df$area_mid, 100))
  df$has_lot <- as.integer(!is.na(df$lot_front) & !is.na(df$lot_depth))
  df$lot_front[is.na(df$lot_front)] <- 0
  df$lot_depth[is.na(df$lot_depth)] <- 0
  df$lot_area <- df$lot_front * df$lot_depth
  df$log_lot  <- log1p(df$lot_area)
  df$garage[is.na(df$garage)] <- 0
  df$parking_total[is.na(df$parking_total)] <- 0
  df$bedrooms[is.na(df$bedrooms)] <- median(listings_raw$bedrooms, na.rm = TRUE)
  df$washrooms[is.na(df$washrooms)] <- median(listings_raw$washrooms, na.rm = TRUE)
  for (cn in CAT_COLS) {
    v <- as.character(df[[cn]])
    bad <- is.na(v) | !(v %in% lv[[cn]])
    v[bad] <- mode_level[[cn]]
    df[[cn]] <- factor(v, levels = lv[[cn]])
  }
  r <- tolower(ifelse(is.na(df$remarks), "", df$remarks))
  df$renovated    <- str_count(r, PAT_RENO)
  df$updated      <- str_count(r, PAT_UPD)
  df$needs_work   <- str_count(r, PAT_NEED)
  df$original     <- str_count(r, PAT_ORIG)
  df$motivated    <- str_count(r, PAT_MOT)
  df$amenity_count <- str_count(r, PAT_AMEN)
  df$remarks_words <- str_count(r, "\\S+")
  df$list_month   <- as.integer(format(df$date_listed, "%m"))
  df
}
msg("Building features ...")
L <- prep(listings_raw)
L$log_price <- log(L$sold_price)
L$sold_ym   <- format(L$date_sold, "%Y-%m")

TEXT_FEATS <- c("renovated", "updated", "needs_work", "original", "motivated",
                "amenity_count", "remarks_words")
BASE_FEATS <- c("house_type_name", "house_style", "bedrooms", "washrooms", "parking_total",
                "garage", "area_mid", "area_is_range", "has_lot", "lot_front", "lot_depth",
                "lot_area", "municipality", "community", "latitude", "longitude",
                "list_month", TEXT_FEATS)

# ---------------- market index + forest: same as the main script ----------------
fit_hedonic <- function(train) {
  d <- train
  d$sold_month <- factor(d$sold_ym)
  m <- lm(log_price ~ house_type_name * log_area + municipality + bedrooms + washrooms +
            parking_total + garage + log_lot + has_lot + sold_month, data = d)
  cf <- coef(m)
  lev <- levels(d$sold_month)
  mc <- cf[paste0("sold_month", lev[-1])]
  mc[is.na(mc)] <- 0
  idx <- data.frame(month = lev, idx = c(0, as.numeric(mc)), stringsAsFactors = FALSE)
  all_m <- format(seq(as.Date(paste0(min(idx$month), "-01")),
                      as.Date(paste0(max(idx$month), "-01")), by = "month"), "%Y-%m")
  idx <- data.frame(month = all_m,
                    idx = approx(match(idx$month, all_m), idx$idx, seq_along(all_m))$y)
  list(model = m, idx = idx, base_month = lev[1], levels = lev)
}
hed_fund <- function(h, df) {
  df$sold_month <- factor(h$base_month, levels = h$levels)
  as.numeric(predict(h$model, newdata = df))
}
idx_forecast <- function(idx, target_date, use_trend) {
  if (!use_trend) return(mean(tail(idx$idx, 3)))
  last12 <- tail(idx, 12)
  t <- seq_len(nrow(last12))
  fit <- lm(idx ~ t, data = last12)
  last_month <- as.Date(paste0(tail(idx$month, 1), "-15"))
  hh <- as.numeric(difftime(target_date, last_month, units = "days")) / 30.4
  as.numeric(predict(fit, data.frame(t = nrow(last12) + hh)))
}
idx_lookup <- function(idx, months_chr, fc) {
  out <- idx$idx[match(months_chr, idx$month)]
  miss <- is.na(out)
  if (any(miss)) {
    early <- months_chr[miss] < min(idx$month)
    out[miss] <- ifelse(early, idx$idx[1], fc)
  }
  out
}
fit_rf <- function(df, y, feats) {
  dat <- df[, feats, drop = FALSE]
  dat$y__ <- y
  ranger(dependent.variable.name = "y__", data = dat, num.trees = NUM_TREES,
         mtry = max(3, floor(length(feats) / 3)), min.node.size = MIN_NODE,
         sample.fraction = SAMPLE_FRACTION, respect.unordered.factors = "order",
         seed = SEED, num.threads = parallel::detectCores())
}

# ---------------- the five versions ---------------------------------------------
VARIANTS <- list(
  full       = list(trend = FALSE, text = TRUE,  detrend = TRUE,  log = TRUE),
  trend_on   = list(trend = TRUE,  text = TRUE,  detrend = TRUE,  log = TRUE),
  no_text    = list(trend = FALSE, text = FALSE, detrend = TRUE,  log = TRUE),
  no_detrend = list(trend = FALSE, text = TRUE,  detrend = FALSE, log = TRUE),
  raw_price  = list(trend = FALSE, text = TRUE,  detrend = TRUE,  log = FALSE)
)

msg("Starting experiments (", length(VAL_MONTHS), "months x", length(VARIANTS), "versions ) ...")
res <- list()
for (i in seq_along(VAL_MONTHS)) {
  M      <- VAL_MONTHS[i]
  cutoff <- M %m-% months(GAP_MONTHS)
  train  <- L[L$date_sold < cutoff, ]
  val    <- L[L$date_sold >= M & L$date_sold < (M %m+% months(1)), ]
  if (nrow(val) == 0) next
  msg(sprintf("Month %s: %d training sales, %d homes to predict", format(M, "%Y-%m"), nrow(train), nrow(val)))
  h <- fit_hedonic(train)
  tr <- train
  tr$ols_fund <- hed_fund(h, tr)
  idx_sold <- idx_lookup(h$idx, tr$sold_ym, mean(tail(h$idx$idx, 3)))
  v <- val
  v$ols_fund <- hed_fund(h, v)
  for (vn in names(VARIANTS)) {
    o <- VARIANTS[[vn]]
    fc <- idx_forecast(h$idx, M + 14, o$trend)
    feats <- c(if (o$text) BASE_FEATS else setdiff(BASE_FEATS, TEXT_FEATS), "ols_fund")
    if (!o$detrend) {
      y <- tr$log_price;                       back <- function(p) exp(p)
    } else if (o$log) {
      y <- tr$log_price - idx_sold;            back <- function(p) exp(p + fc)
    } else {
      y <- tr$sold_price / exp(idx_sold);      back <- function(p) p * exp(fc)
    }
    rf <- fit_rf(tr, y, feats)
    pr <- back(predict(rf, v[, feats, drop = FALSE])$predictions)
    res[[length(res) + 1]] <- data.frame(
      variant = vn, month = format(M, "%Y-%m"), ml_num = v$ml_num,
      community = as.character(v$community), municipality = as.character(v$municipality),
      house_type_name = as.character(v$house_type_name),
      actual = v$sold_price, pred = pr, stringsAsFactors = FALSE)
    msg("   done:", vn)
  }
}

R <- bind_rows(res)
R$ape <- abs(R$pred - R$actual) / R$actual * 100
R$variant <- factor(R$variant, levels = names(VARIANTS))
write_csv(R, "reports/ablation_predictions.csv")

S <- R %>% group_by(variant) %>%
  summarise(homes = n(), MdAPE = median(ape), MAPE = mean(ape),
            median_signed_error_pct = median((pred - actual) / actual) * 100, .groups = "drop")
write_csv(S, "reports/ablation_summary.csv")

comm <- R %>% filter(variant == "full") %>% group_by(community) %>%
  summarise(homes = n(), MdAPE = median(ape), .groups = "drop") %>% filter(homes >= 30)
write_csv(comm %>% arrange(desc(MdAPE)), "reports/ablation_error_by_community.csv")

ratio <- L %>% filter(!is.na(list_price), list_price > 0) %>% group_by(sold_ym) %>%
  summarise(homes = n(), median_sold_over_list = round(median(sold_price / list_price), 4), .groups = "drop")
write_csv(ratio, "reports/ablation_sold_over_list_by_month.csv")

cat("\n\n==================== ABLATION RESULTS ====================\n")
cat("(off-market track, walk-forward Dec 2025 - May 2026, 100 trees)\n\n")
print(as.data.frame(S), digits = 4)
cat("\nWorst communities for the 'full' model (at least 30 validation homes):\n")
print(as.data.frame(head(comm %>% arrange(desc(MdAPE)), 8)), digits = 4)
cat("\nBest communities:\n")
print(as.data.frame(head(comm %>% arrange(MdAPE), 3)), digits = 4)
cat("\nCommunities with 30+ validation homes:", nrow(comm), "\n")
cat("\nMedian sold price / list price, by sale month (last 14 months):\n")
print(tail(as.data.frame(ratio), 14))
cat("\n==================== END - copy from ABLATION RESULTS ====================\n")
