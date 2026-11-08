# SSU Data Dash 2026: Qixin Xiao

Solo submission. Random forest on log prices for Toronto-area sold-price prediction (off-market and list-aware tracks).

**Video:** https://youtu.be/ugm2MfIyd_I

## Where things are

| Path | What it is |
|---|---|
| `ssu_random_forest_v2.R` | The full pipeline: features, price index, walk-forward validation, final models, submissions, report tables and figures |
| `submissions/submission_off_market.csv` | Hand-in predictions, off-market track |
| `submissions/submission_list_aware.csv` | Hand-in predictions, list-aware track |
| `report/report.tex` | Report source (PDF: `report/report.pdf`) |
| `reports/` | Tables and numbers the report reads, plus validation outputs and experiment results |
| `figures/` | Figures used in the report |
| `ablation.R` | Extra experiments: one-change-at-a-time tests, community errors |

## Important: how the submission files were made

The two submission CSVs were written by the R script `ssu_random_forest_v2.R`, **not** by the starter's `run.py`. The starter Python files (`run.py`, `features.py`, `models.py`, `data.py`, `check_submission.py`) are unchanged. Running `python run.py predict` would overwrite the CSVs with the starter's predictions, so please use the R script to reproduce our results.

## Reproducing

1. Install R (with RStudio if you like). Put `ssu_random_forest_v2.R` in an empty folder.
2. Run `Rscript ssu_random_forest_v2.R` (or open it in RStudio and click Source). It installs missing packages, downloads the data from Hugging Face, and takes about 15-40 minutes.
3. It writes `submissions/`, `reports/` and `figures/` next to the script.
4. Optional: run `ablation.R` in the same folder for the extra experiments (30-60 minutes).
5. Compile `report/report.tex` with pdflatex (Overleaf works: upload the whole repository).
