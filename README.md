# Urban Home-Farming Start-up — End-to-End Validation (Synthetic Data)

Individual assignment: synthetic data → cleaning & transformation → descriptive & diagnostic analytics,
validating a vertical balcony growing kit (₹4,999) + ₹299/month refill subscription for Indian city homes,
end to end through to the sales pipeline (option a).

**All data is synthetic and illustrative.** It is generated from business rules to test the idea and the
analysis method; it is not real market evidence.

## What is where

| Assignment part | Marks | File(s) |
|---|---|---|
| 1. Synthetic data for end-to-end validation | 10 | `01_generate_data.py` → `data/raw/` |
| 2. Data cleaning & transformation | 10 | `02_clean_transform.ipynb` → `data/clean/` (+ `cleaning_log.csv`) |
| 3. Descriptive & diagnostic analytics, correlation graphs, every graph explained | 30 | `03_eda.ipynb` (28 figures → `figures/`), `app.py` (Streamlit dashboard) |
| Predictive + prescriptive (as asked in class): classification, clustering, association rules, regression, accuracy / confusion matrices, break-even | — | `04_predictive_prescriptive.ipynb`, `models.py`, dashboard pages 7–8 |
| Report (written by the student, with dashboard screenshots) | — | `REPORT.pdf` / `REPORT.docx` (add before submitting); screenshots in `screenshots/` |

### Datasets
| File | Rows | Description |
|---|---|---|
| `data/raw/survey_raw.csv` | 3,045 | Consumer survey, form-tool export (question-text headers, typed amounts, multi-selects) |
| `data/raw/pipeline_raw.csv` | 4,326 | 6-month pilot CRM export: lead → contact → demo → purchase → refill → retention |
| `data/raw/planted_noise_log.csv`, `survey_answer_key_planted_issues.csv` | — | Answer keys of the deliberately planted data problems |
| `data/clean/survey_clean.csv` | 2,940 × 72 | Cleaned + engineered survey |
| `data/clean/pipeline_clean.csv` | 4,200 × 43 | Cleaned + engineered pipeline |
| `data/clean/cleaning_log.csv` | 31 actions | Every cleaning action and rows affected |

### Code
- `models.py` — the four algorithms (classification incl. pilot lead scoring, k-means clustering, Apriori association rules, OLS regression), algorithm comparison, and the 48-month break-even model.
- `analytics.py` — one function per chart: computes the numbers **and** the explanation (What it shows / Why / So what), so the notebook, dashboard and report always agree.
- `charts.py` — draws each result in matplotlib (notebook, report) and Plotly (dashboard) with one palette.
- `src/02_clean_transform.py`, `src/03_eda.py` — notebook sources (`# %%` cells); `src/build_nb.py` builds and executes them.

## How to run
```bash
pip install -r requirements.txt
python 01_generate_data.py                                                     # data/raw/
jupyter nbconvert --to notebook --execute --inplace 02_clean_transform.ipynb   # data/clean/
jupyter nbconvert --to notebook --execute --inplace 03_eda.ipynb               # figures/ 01-28
jupyter nbconvert --to notebook --execute --inplace 04_predictive_prescriptive.ipynb   # figures/ 29-36
streamlit run app.py                                                           # http://localhost:8501
```
The notebooks are already executed, so they can be read without running anything.

## Dashboard pages
Overview · 1 Data preparation · 2 Survey: descriptive · 3 Survey: diagnostic · 4 Pipeline: descriptive ·
5 Pipeline: diagnostic · 6 End-to-end validation (stated vs actual, LTV:CAC by channel, finalised pipeline) ·
7 Predictive: 4 algorithms (tabs) · 8 Prescriptive: break-even & plan.
Sidebar filters (city tier, income, space; city, channel, month) recalculate every chart **and** its explanation.

## Headline result
Viable as a focused, demo-led business, not a mass-market online product: launch at ₹4,999 in gated housing
societies, acquire through society demos and referrals (LTV:CAC 2.9x and 5.2x; Instagram 0.4x), call leads within
2 hours (25.1% vs 3.6% conversion), and get the first harvest inside 35 days (monthly churn 6.1% vs 13.4%).
With the pilot's economics it never breaks even in 48 months; with this strategy monthly profit starts in month 28
and cash pays back in month 48 (~₹79 lakh peak funding). Best algorithms: logistic regression (AUC 0.87) and OLS
regression (R² 0.74); weakest: association rules (max lift 1.47).
