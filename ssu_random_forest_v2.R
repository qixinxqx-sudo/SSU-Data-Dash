# =============================================================================
# SSU Data Dash 2026 - Random Forest on log prices (R version)
#
# HOW TO RUN (RStudio):
#   1. Put this file in an empty folder (this folder becomes your project folder).
#   2. Open it in RStudio and click "Source" (top-right of the editor window).
#   3. Wait ~15-40 minutes. Progress messages print in the Console.
#
# WHAT YOU GET (folders are created automatically next to this file):
#   submissions/submission_off_market.csv   <- hand-in file, track 1
#   submissions/submission_list_aware.csv   <- hand-in file, track 2
#   reports/results_table.tex               <- Results table for the report
#   reports/error_by_type.tex               <- Error table by house type
#   reports/numbers.tex                     <- every number quoted in the report text
#   reports/*.csv                           <- raw validation predictions, importances
#   figures/error_by_price_range.png        <- figure for the report
#   figures/importance.png                  <- figure for the report
#
# THE MODELING IDEA, IN PLAIN WORDS
#   * We predict log(price), so a 10% miss on a $500k condo counts the same as a
#     10% miss on a $3M house (the competition metric is a percentage error).
#   * Prices drift over time and a forest cannot extrapolate beyond what it saw.
#     So we first estimate a market price index from PAST sales only (an OLS
#     regression with one dummy per sale month), subtract it from log prices,
#     train the forest on the "de-trended" target, and add a forecast of the
#     index back at prediction time.
#   * Validation is walk-forward: every validation month is predicted using only
#     sales that closed at least GAP_MONTHS earlier. Nothing from the future.
#   * Off-market model: no list price. List-aware model: predicts
#     log(sold price / list price), i.e. how far above/below list the home sells.
# =============================================================================

# ----------------------------- 0. SETTINGS -----------------------------------
SEED            <- 2026
NUM_TREES       <- 200      # more trees = slightly better, slower
MIN_NODE        <- 10       # minimum rows per leaf
SAMPLE_FRACTION <- 0.4      # share of rows each tree sees
GAP_MONTHS      <- 2        # validation gap: mimics the Jun-Sep test horizon
VAL_MONTHS      <- seq(as.Date("2025-12-01"), as.Date("2026-05-01"), by = "month")
# Price-index forecast: FALSE = hold it flat, TRUE = extrapolate its trend.
# Chosen separately for each track from the validation results:
#   off-market  flat  -> MdAPE 11.43 (trend:  11.93)
#   list-aware  trend -> MdAPE  5.72 (flat:    6.01)
USE_TREND_OFF   <- FALSE
USE_TREND_LA    <- TRUE
N_BOOT          <- 1000     # bootstrap repetitions for the confidence intervals
TEST_TARGET     <- as.Date("2026-07-31")  # middle of the Jun-Sep 2026 test window
DATA_DIR        <- "data"

# --------------------------- 1. PACKAGES / FOLDERS ---------------------------
need <- c("ranger", "dplyr", "readr", "lubridate", "stringr", "ggplot2", "tidyr")
miss <- need[!sapply(need, requireNamespace, quietly = TRUE)]
if (length(miss)) install.packages(miss, repos = "https://cloud.r-project.org")
suppressPackageStartupMessages({
  library(ranger); library(dplyr); library(readr)
  library(lubridate); library(stringr); library(ggplot2); library(tidyr)
})

# Work in the folder that contains this script (RStudio only; harmless elsewhere)
if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
  p <- tryCatch(dirname(rstudioapi::getSourceEditorContext()$path), error = function(e) "")
  if (nzchar(p) && dir.exists(p)) setwd(p)
}
for (d in c(DATA_DIR, "submissions", "reports", "figures")) dir.create(d, showWarnings = FALSE)
set.seed(SEED)
msg <- function(...) cat(format(Sys.time(), "%H:%M:%S"), "-", ..., "\n")
msg("Working folder:", getwd())

# ------------------------------- 2. LOAD DATA --------------------------------
BASE_URL <- "https://huggingface.co/datasets/Baliyang/SSU_data_dash/resolve/main/"
get_file <- function(f) {
  dest <- file.path(DATA_DIR, f)
  if (!file.exists(dest)) {
    msg("Downloading", f, "(the big one is ~85 MB, please wait)")
    options(timeout = 3600)
    download.file(paste0(BASE_URL, f, "?download=true"), dest, mode = "wb", quiet = FALSE)
  }
  dest
}
read_any <- function(f) read_csv(get_file(f), guess_max = 200000, show_col_types = FALSE, progress = FALSE)

msg("Reading data ...")
listings_raw <- read_any("listings.csv")
test_off_raw <- read_any("test_off_market.csv")
test_la_raw  <- read_any("test_list_aware.csv")
listings_raw$date_sold   <- as.Date(listings_raw$date_sold)
listings_raw$date_listed <- as.Date(listings_raw$date_listed)
stopifnot(all(c("sold_price", "date_sold", "date_listed") %in% names(listings_raw)))
listings_raw <- listings_raw %>% filter(!is.na(sold_price), sold_price > 0, !is.na(date_sold))
msg("Training sales:", nrow(listings_raw), "| off-market test homes:", nrow(test_off_raw),
    "| list-aware test homes:", nrow(test_la_raw))
msg("Sales span", format(min(listings_raw$date_sold)), "to", format(max(listings_raw$date_sold)))

# ----------------------------- 3. FEATURE BUILDING ---------------------------
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

# Keyword groups used to turn the written description (remarks) into numbers.
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
  n_unseen <- 0
  for (cn in CAT_COLS) {
    v <- as.character(df[[cn]])
    bad <- is.na(v) | !(v %in% lv[[cn]])
    n_unseen <- n_unseen + sum(bad)
    v[bad] <- mode_level[[cn]]
    df[[cn]] <- factor(v, levels = lv[[cn]])
  }
  if (n_unseen > 0) msg("Note:", n_unseen, "category values unseen in training were mapped to the most common level")
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
L  <- prep(listings_raw)
L$log_price <- log(L$sold_price)
L$sold_ym   <- format(L$date_sold, "%Y-%m")
T_OFF <- prep(test_off_raw)
T_LA  <- prep(test_la_raw)
stopifnot("list_price" %in% names(T_LA))
L_has_list <- !is.na(L$list_price) & L$list_price > 0

BASE_FEATS <- c("house_type_name", "house_style", "bedrooms", "washrooms", "parking_total",
                "garage", "area_mid", "area_is_range", "has_lot", "lot_front", "lot_depth",
                "lot_area", "municipality", "community", "latitude", "longitude",
                "list_month", "renovated", "updated", "needs_work", "original", "motivated",
                "amenity_count", "remarks_words")

# --------------------------- 4. MARKET INDEX (OLS) ---------------------------
# OLS on log price with one dummy per sale month. The month coefficients form a
# price index for "the same kind of home" over time. Also gives a baseline
# "fundamental value" (price of the home at the base month) used as a feature.
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
  # fill any missing calendar month by linear interpolation
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
  h <- as.numeric(difftime(target_date, last_month, units = "days")) / 30.4
  as.numeric(predict(fit, data.frame(t = nrow(last12) + h)))
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

# ------------------------------ 5. MODEL HELPERS -----------------------------
fit_rf <- function(df, y, feats, importance = "none") {
  dat <- df[, feats, drop = FALSE]
  dat$y__ <- y
  ranger(dependent.variable.name = "y__", data = dat, num.trees = NUM_TREES,
         mtry = max(3, floor(length(feats) / 3)), min.node.size = MIN_NODE,
         sample.fraction = SAMPLE_FRACTION, respect.unordered.factors = "order",
         importance = importance, seed = SEED, num.threads = parallel::detectCores())
}

# Builds everything needed to predict both tracks for a set of rows, given a
# training window. `train_L` = past sales; `target_date` = when the predicted
# homes will sell (used only to forecast the price index).
fit_and_predict <- function(train_L, off_rows, la_rows, target_date, importance = "none") {
  h  <- fit_hedonic(train_L)
  fc_off <- idx_forecast(h$idx, target_date, USE_TREND_OFF)
  fc_la  <- idx_forecast(h$idx, target_date, USE_TREND_LA)

  tr <- train_L
  tr$ols_fund <- hed_fund(h, tr)
  tr$idx_sold <- idx_lookup(h$idx, tr$sold_ym, fc_off)

  # --- off-market forest: de-trended log price, no list price anywhere ---
  feats_off <- c(BASE_FEATS, "ols_fund")
  rf_off <- fit_rf(tr, tr$log_price - tr$idx_sold, feats_off, importance)

  # --- list-aware forest: log(sold / list) ---
  trl <- tr[!is.na(tr$list_price) & tr$list_price > 0, ]
  trl$log_list <- log(trl$list_price)
  trl$list_vs_ols <- trl$log_list - (trl$ols_fund +
                       idx_lookup(h$idx, format(trl$date_listed, "%Y-%m"), fc_la))
  feats_la <- c(BASE_FEATS, "ols_fund", "log_list", "list_vs_ols")
  rf_la <- fit_rf(trl, trl$log_price - trl$log_list, feats_la, importance)

  out <- list(h = h, fc = fc_off, fc_la = fc_la, rf_off = rf_off, rf_la = rf_la, train = tr)
  if (!is.null(off_rows)) {
    o <- off_rows
    o$ols_fund <- hed_fund(h, o)
    out$off_pred <- exp(predict(rf_off, o[, feats_off, drop = FALSE])$predictions + fc_off)
    out$ols_pred <- exp(o$ols_fund + fc_off)
  }
  if (!is.null(la_rows)) {
    a <- la_rows
    a$ols_fund <- hed_fund(h, a)
    a$log_list <- log(a$list_price)
    a$list_vs_ols <- a$log_list - (a$ols_fund +
                       idx_lookup(h$idx, format(a$date_listed, "%Y-%m"), fc_la))
    out$la_pred <- a$list_price * exp(predict(rf_la, a[, feats_la, drop = FALSE])$predictions)
  }
  out
}

group_median_baseline <- function(train_L, rows, cutoff) {
  recent <- train_L %>% filter(date_sold >= cutoff %m-% months(12))
  if (nrow(recent) < 500) recent <- train_L
  g1 <- recent %>% group_by(house_type_name, community) %>%
    summarise(p1 = median(sold_price), n1 = n(), .groups = "drop") %>% filter(n1 >= 3)
  g2 <- recent %>% group_by(house_type_name, municipality) %>%
    summarise(p2 = median(sold_price), .groups = "drop")
  g3 <- recent %>% group_by(house_type_name) %>% summarise(p3 = median(sold_price), .groups = "drop")
  r <- rows %>% select(house_type_name, community, municipality) %>%
    left_join(g1[, c("house_type_name", "community", "p1")], by = c("house_type_name", "community")) %>%
    left_join(g2, by = c("house_type_name", "municipality")) %>%
    left_join(g3, by = "house_type_name")
  coalesce(r$p1, r$p2, r$p3, median(recent$sold_price))
}

# ---------------------- 6. WALK-FORWARD VALIDATION ---------------------------
msg("Starting walk-forward validation (", length(VAL_MONTHS), "months ) ...")
val_list <- vector("list", length(VAL_MONTHS))
for (i in seq_along(VAL_MONTHS)) {
  M      <- VAL_MONTHS[i]
  cutoff <- M %m-% months(GAP_MONTHS)
  train  <- L[L$date_sold < cutoff, ]
  val    <- L[L$date_sold >= M & L$date_sold < (M %m+% months(1)), ]
  if (nrow(val) == 0) next
  val_la <- val[!is.na(val$list_price) & val$list_price > 0, ]
  msg(sprintf("Validation month %s: train on %d sales before %s, predict %d homes",
              format(M, "%Y-%m"), nrow(train), format(cutoff), nrow(val)))
  res <- fit_and_predict(train, val, val_la, target_date = M + 14)
  val_list[[i]] <- list(
    off = data.frame(track = "Off-market", month = format(M, "%Y-%m"), ml_num = val$ml_num,
                     house_type_name = as.character(val$house_type_name),
                     municipality = as.character(val$municipality),
                     bedrooms = val$bedrooms, actual = val$sold_price,
                     rf = res$off_pred, ols = res$ols_pred,
                     base = group_median_baseline(train, val, cutoff)),
    la  = data.frame(track = "List-aware", month = format(M, "%Y-%m"), ml_num = val_la$ml_num,
                     house_type_name = as.character(val_la$house_type_name),
                     municipality = as.character(val_la$municipality),
                     bedrooms = val_la$bedrooms, actual = val_la$sold_price,
                     rf = res$la_pred, ols = NA_real_, base = val_la$list_price))
}
VOFF <- bind_rows(lapply(val_list, `[[`, "off"))
VLA  <- bind_rows(lapply(val_list, `[[`, "la"))
write_csv(bind_rows(VOFF, VLA), "reports/validation_predictions.csv")

ape <- function(p, a) abs(p - a) / a
boot_ci <- function(x, fn) {
  n <- length(x)
  quantile(replicate(N_BOOT, fn(x[sample.int(n, n, replace = TRUE)])), c(0.025, 0.975))
}
summ <- function(track, model, p, a) {
  e <- ape(p, a) * 100
  e <- e[is.finite(e)]
  ci_md <- unname(boot_ci(e, median)); ci_mn <- unname(boot_ci(e, mean))
  data.frame(track = track, model = model, n = length(e),
             mdape = median(e), mdape_lo = ci_md[1], mdape_hi = ci_md[2],
             mape = mean(e), mape_lo = ci_mn[1], mape_hi = ci_mn[2])
}
msg("Computing validation metrics + bootstrap intervals ...")
RES <- bind_rows(
  summ("Off-market", "Group-median baseline", VOFF$base, VOFF$actual),
  summ("Off-market", "OLS price index + attributes", VOFF$ols, VOFF$actual),
  summ("Off-market", "Random forest (log price)", VOFF$rf, VOFF$actual),
  summ("List-aware", "List price baseline", VLA$base, VLA$actual),
  summ("List-aware", "Random forest (log sale/list)", VLA$rf, VLA$actual))
write_csv(RES, "reports/validation_results.csv")
print(as.data.frame(RES), digits = 4)

# ---- results_table.tex (matches the template's \input{...results_table.tex}) --
fmt <- function(v, lo, hi) sprintf("%.2f [%.2f, %.2f]", v, lo, hi)
rows <- apply(RES, 1, function(r) {
  sprintf("%s & %s & %s & %s \\\\", r[["track"]], r[["model"]],
          fmt(as.numeric(r[["mdape"]]), as.numeric(r[["mdape_lo"]]), as.numeric(r[["mdape_hi"]])),
          fmt(as.numeric(r[["mape"]]), as.numeric(r[["mape_lo"]]), as.numeric(r[["mape_hi"]])))
})
tex_table <- c("\\begin{tabular}{llcc}", "\\toprule",
               "Track & Model & MdAPE (\\%) & MAPE (\\%) \\\\", "\\midrule",
               rows[1:3], "\\midrule", rows[4:5], "\\bottomrule", "\\end{tabular}")
writeLines(tex_table, "reports/results_table.tex")

# ----------------------- 7. WHERE THE MODEL FAILS ----------------------------
ALL <- bind_rows(VOFF, VLA) %>% mutate(ape = ape(rf, actual) * 100)
ALL <- ALL %>% group_by(track) %>% mutate(decile = ntile(actual, 10)) %>% ungroup()
ALL$track <- factor(ALL$track, levels = c("Off-market", "List-aware"))

dec_tab <- ALL %>% group_by(track, decile) %>%
  summarise(price_lo = min(actual), price_hi = max(actual), mdape = median(ape), .groups = "drop")
write_csv(dec_tab, "reports/error_by_decile.csv")

p1 <- ggplot(ALL, aes(factor(decile), ape)) +
  geom_boxplot(outlier.shape = NA, fill = "grey85") +
  facet_wrap(~track) +
  coord_cartesian(ylim = c(0, quantile(ALL$ape, 0.95, na.rm = TRUE))) +
  labs(x = "Sold-price decile (1 = cheapest, 10 = most expensive)",
       y = "Absolute percentage error (%)") +
  theme_minimal(base_size = 11)
ggsave("figures/error_by_price_range.png", p1, width = 8, height = 3.6, dpi = 200)

type_tab <- ALL %>% group_by(track, house_type_name) %>%
  summarise(n = n(), mdape = median(ape), .groups = "drop")
muni_tab <- ALL %>% group_by(track, municipality) %>%
  summarise(n = n(), mdape = median(ape), .groups = "drop")
month_tab <- ALL %>% group_by(track, month) %>%
  summarise(n = n(), mdape = median(ape), .groups = "drop")
bed_tab <- ALL %>% mutate(beds = pmin(bedrooms, 5)) %>% group_by(track, beds) %>%
  summarise(n = n(), mdape = median(ape), .groups = "drop")
write_csv(type_tab, "reports/error_by_type.csv"); write_csv(muni_tab, "reports/error_by_municipality.csv")
write_csv(month_tab, "reports/error_by_month.csv"); write_csv(bed_tab, "reports/error_by_bedrooms.csv")

tw <- type_tab %>% select(-n) %>% pivot_wider(names_from = track, values_from = mdape) %>%
  arrange(desc(`Off-market`))
ntab <- type_tab %>% filter(track == "Off-market") %>% select(house_type_name, n)
tw <- left_join(tw, ntab, by = "house_type_name")
tex_escape <- function(s) gsub("_", "\\\\_", gsub("&", "\\\\&", s))
type_rows <- sprintf("%s & %d & %.2f & %s \\\\", tex_escape(tw$house_type_name), tw$n, tw$`Off-market`,
                     ifelse(is.na(tw$`List-aware`), "--", sprintf("%.2f", tw$`List-aware`)))
writeLines(c("\\begin{tabular}{lrrr}", "\\toprule",
             "House type & Homes (off-mkt.) & Off-market MdAPE (\\%) & List-aware MdAPE (\\%) \\\\",
             "\\midrule", type_rows, "\\bottomrule", "\\end{tabular}"), "reports/error_by_type.tex")

# ------------------------- 8. FINAL FIT + SUBMISSIONS ------------------------
msg("Fitting final models on ALL sales and predicting the test homes ...")
final <- fit_and_predict(L, T_OFF, T_LA, target_date = TEST_TARGET, importance = "permutation")

fallback_off <- group_median_baseline(L, T_OFF, max(L$date_sold) + 1)
pred_off <- final$off_pred
bad <- !is.finite(pred_off) | pred_off <= 0
pred_off[bad] <- fallback_off[bad]
pred_la <- final$la_pred
bad2 <- !is.finite(pred_la) | pred_la <= 0
pred_la[bad2] <- T_LA$list_price[bad2]

sub_off <- data.frame(ml_num = T_OFF$ml_num, predicted_price = round(pred_off))
sub_la  <- data.frame(ml_num = T_LA$ml_num,  predicted_price = round(pred_la))
stopifnot(nrow(sub_off) == nrow(test_off_raw), nrow(sub_la) == nrow(test_la_raw),
          !anyNA(sub_off$predicted_price), !anyNA(sub_la$predicted_price),
          !anyDuplicated(sub_off$ml_num), !anyDuplicated(sub_la$ml_num))
write_csv(sub_off, "submissions/submission_off_market.csv")
write_csv(sub_la,  "submissions/submission_list_aware.csv")
msg("Wrote submissions/submission_off_market.csv and submission_list_aware.csv")
msg(sprintf("Forecast price-index level used for the Jun-Sep test months: %.4f", final$fc))

# Feature importance (permutation importance from the final forests)
imp_off <- sort(final$rf_off$variable.importance, decreasing = TRUE)
imp_la  <- sort(final$rf_la$variable.importance,  decreasing = TRUE)
imp_df <- bind_rows(
  data.frame(track = "Off-market", feature = names(imp_off), importance = as.numeric(imp_off)),
  data.frame(track = "List-aware", feature = names(imp_la),  importance = as.numeric(imp_la)))
write_csv(imp_df, "reports/feature_importance.csv")
top_imp <- imp_df %>% group_by(track) %>% slice_max(importance, n = 10) %>% ungroup()
top_imp$track <- factor(top_imp$track, levels = c("Off-market", "List-aware"))
p2 <- ggplot(top_imp, aes(reorder(feature, importance), importance)) +
  geom_col(fill = "grey40") + coord_flip() + facet_wrap(~track, scales = "free") +
  labs(x = NULL, y = "Permutation importance (increase in squared error)") +
  theme_minimal(base_size = 10)
ggsave("figures/importance.png", p2, width = 8, height = 3.4, dpi = 200)

# Price-index picture for sanity checking (not needed in the report)
p3 <- ggplot(final$h$idx, aes(as.Date(paste0(month, "-01")), idx)) + geom_line() +
  labs(x = NULL, y = "Log price index (OLS month effects)") + theme_minimal()
ggsave("figures/price_index.png", p3, width = 7, height = 3, dpi = 150)

# ----------------- 9. numbers.tex : every number the report quotes -----------
g <- function(track, model, col) RES[[col]][RES$track == track & RES$model == model]
off_rf   <- "Random forest (log price)"; off_base <- "Group-median baseline"; off_ols <- "OLS price index + attributes"
la_rf    <- "Random forest (log sale/list)"; la_base <- "List price baseline"
top3 <- function(tr) { v <- imp_df %>% filter(track == tr) %>% arrange(desc(importance)) %>% pull(feature); tex_escape(v[1:3]) }
worst_type <- type_tab %>% filter(track == "Off-market") %>% arrange(desc(mdape)) %>% slice(1)
best_type  <- type_tab %>% filter(track == "Off-market") %>% arrange(mdape) %>% slice(1)
worst_dec  <- dec_tab %>% filter(track == "Off-market") %>% arrange(desc(mdape)) %>% slice(1)
best_dec   <- dec_tab %>% filter(track == "Off-market") %>% arrange(mdape) %>% slice(1)
worst_mo   <- month_tab %>% filter(track == "Off-market") %>% arrange(desc(mdape)) %>% slice(1)
best_mo    <- month_tab %>% filter(track == "Off-market") %>% arrange(mdape) %>% slice(1)
worst_mu   <- muni_tab %>% filter(track == "Off-market") %>% arrange(desc(mdape)) %>% slice(1)
la_worst_dec <- dec_tab %>% filter(track == "List-aware") %>% arrange(desc(mdape)) %>% slice(1)
bias_off <- median((VOFF$rf - VOFF$actual) / VOFF$actual) * 100
bias_la  <- median((VLA$rf - VLA$actual) / VLA$actual) * 100
rel <- function(b, m) (b - m) / b * 100
mac <- function(n, v) sprintf("\\newcommand{\\%s}{%s}", n, v)
nums <- c(
  mac("nTrain", format(nrow(L), big.mark = "{,}")),
  mac("nValOff", format(nrow(VOFF), big.mark = "{,}")),
  mac("nValLa", format(nrow(VLA), big.mark = "{,}")),
  mac("nTestOff", format(nrow(sub_off), big.mark = "{,}")),
  mac("nTestLa", format(nrow(sub_la), big.mark = "{,}")),
  mac("gapMonths", GAP_MONTHS),
  mac("offMdAPE", sprintf("%.1f", g("Off-market", off_rf, "mdape"))),
  mac("offMAPE", sprintf("%.1f", g("Off-market", off_rf, "mape"))),
  mac("offBaseMdAPE", sprintf("%.1f", g("Off-market", off_base, "mdape"))),
  mac("offOlsMdAPE", sprintf("%.1f", g("Off-market", off_ols, "mdape"))),
  mac("offGainBase", sprintf("%.0f", rel(g("Off-market", off_base, "mdape"), g("Off-market", off_rf, "mdape")))),
  mac("offGainOls", sprintf("%.0f", rel(g("Off-market", off_ols, "mdape"), g("Off-market", off_rf, "mdape")))),
  mac("laMdAPE", sprintf("%.1f", g("List-aware", la_rf, "mdape"))),
  mac("laMAPE", sprintf("%.1f", g("List-aware", la_rf, "mape"))),
  mac("laBaseMdAPE", sprintf("%.1f", g("List-aware", la_base, "mdape"))),
  mac("laGainBase", sprintf("%.0f", rel(g("List-aware", la_base, "mdape"), g("List-aware", la_rf, "mdape")))),
  mac("offBiasPct", sprintf("%+.1f", bias_off)),
  mac("laBiasPct", sprintf("%+.1f", bias_la)),
  mac("worstType", tex_escape(worst_type$house_type_name)), mac("worstTypeMd", sprintf("%.1f", worst_type$mdape)),
  mac("bestType", tex_escape(best_type$house_type_name)),   mac("bestTypeMd", sprintf("%.1f", best_type$mdape)),
  mac("worstDecile", worst_dec$decile), mac("worstDecileMd", sprintf("%.1f", worst_dec$mdape)),
  mac("bestDecile", best_dec$decile),   mac("bestDecileMd", sprintf("%.1f", best_dec$mdape)),
  mac("worstDecilePriceLo", format(round(worst_dec$price_lo, -3), big.mark = "{,}")),
  mac("laWorstDecile", la_worst_dec$decile), mac("laWorstDecileMd", sprintf("%.1f", la_worst_dec$mdape)),
  mac("worstMonth", worst_mo$month), mac("worstMonthMd", sprintf("%.1f", worst_mo$mdape)),
  mac("bestMonth", best_mo$month),   mac("bestMonthMd", sprintf("%.1f", best_mo$mdape)),
  mac("worstMuni", tex_escape(worst_mu$municipality)), mac("worstMuniMd", sprintf("%.1f", worst_mu$mdape)),
  mac("offTopA", top3("Off-market")[1]), mac("offTopB", top3("Off-market")[2]), mac("offTopC", top3("Off-market")[3]),
  mac("laTopA", top3("List-aware")[1]),  mac("laTopB", top3("List-aware")[2]),  mac("laTopC", top3("List-aware")[3]),
  mac("indexLevel", sprintf("%.3f", final$fc)),
  mac("trendUsed", sprintf("%s for the off-market model and %s for the list-aware model",
        ifelse(USE_TREND_OFF, "extrapolated linearly", "held flat at the average of its last three months"),
        ifelse(USE_TREND_LA,  "extrapolated linearly", "held flat at the average of its last three months")))
)
writeLines(nums, "reports/numbers.tex")
msg("Wrote reports/numbers.tex")

cat("\n==================== DONE ====================\n")
cat("Validation (walk-forward, Dec 2025 - May 2026):\n")
print(as.data.frame(RES[, c("track", "model", "n", "mdape", "mape")]), digits = 4)
cat("\nTrend settings used: off-market =", USE_TREND_OFF, "| list-aware =", USE_TREND_LA, "\n")
