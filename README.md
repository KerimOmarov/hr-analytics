# HR Analytics: Employee Attrition Risk Prediction

> Identifying which data science professionals are most likely to seek a job change and the organizational factors that drive it.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![DuckDB](https://img.shields.io/badge/SQL-DuckDB-yellow)](https://duckdb.org)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Project Overview

An end-to-end people-analytics case study built on a dataset of 19,158 data science professionals. Using SQL analysis, exploratory data analysis, and a Random Forest classifier with SHAP explainability, the project identifies the key drivers of employee attrition risk and produces an actionable workforce segmentation framework.

**Target audience:** HR analytics teams, people ops leaders, and talent acquisition managers at data-driven organizations.

**Key result:** The model achieves AUC-ROC of 0.81, correctly ranking a job-seeker above a non-seeker 81% of the time. City development index and years of experience are the dominant attrition signals - training hours alone are not.

---

## Business Insights

1. **City development index** is the strongest single predictor - candidates from low-CDI cities are ~1.9× more likely to seek a change.
2. **2–4 year experience band** is the peak flight-risk window - the most critical retention intervention point.
3. **Company size** inversely predicts stability - employees from firms with <50 staff show 35%+ attrition intent.
4. **Training hours do not predict attrition** (r ≈ 0.01) - upskilling is not a retention signal on its own.
5. **Missing company data** is itself a risk signal - likely freelancers with structurally higher mobility.

---

## Project Structure

```
hr-analytics/
├── data/
│   ├── raw/                    # Original Kaggle CSVs (not committed)
│   └── processed/              # Cleaned, model-ready datasets
│
├── notebooks/
│   ├── 01_data_inspection.ipynb    # Shape, types, missing values, data quality report
│   ├── 02_sql_analysis.ipynb       # 12 SQL queries via DuckDB (window functions, CTEs)
│   ├── 03_eda.ipynb                # Full EDA: distributions, bivariate, multivariate, insights
│   ├── 04_preprocessing.ipynb      # Encoding, imputation, feature engineering
│   └── 05_modeling.ipynb           # LR → RF → Tuned RF, SHAP, risk tiers
│
├── sql/
│   └── hr_queries.sql              # Standalone SQL - all 12 queries with comments
│
├── src/
│   ├── data_utils.py               # Reusable encoding, imputation, validation functions
│   └── viz_utils.py                # Reusable chart functions with consistent styling
│
├── outputs/
│   ├── figures/                    # All saved chart PNGs (named descriptively)
│   ├── summary_report.md           # Plain-English business findings
│   ├── hr_attrition_rf_model.pkl   # Saved tuned Random Forest model
│   └── feature_columns.pkl         # Feature column list for inference
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Methodology

```
Data Inspection → SQL Analysis → EDA → Preprocessing → Modeling → Explainability
```

| Phase | Notebook | Tool | Output |
|---|---|---|---|
| Data inspection | `01_data_inspection` | pandas | Data quality report, missing value audit |
| SQL analysis | `02_sql_analysis` | DuckDB | 12 business queries - RANK, CTEs, QUALIFY |
| EDA | `03_eda` | matplotlib, seaborn | 11 charts, 5 key business insights |
| Preprocessing | `04_preprocessing` | pandas, scikit-learn | Ordinal encoding, imputation strategy, 3 engineered features |
| Modeling | `05_modeling` | scikit-learn, SHAP | RF tuned via RandomizedSearchCV, AUC-ROC 0.81, risk tier segmentation |

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/hr-analytics
cd hr-analytics

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset via Kaggle CLI
#    (Requires kaggle.json API token in ~/.kaggle/)
kaggle datasets download arashnic/hr-analytics-job-change-of-data-scientists
unzip hr-analytics-job-change-of-data-scientists.zip -d data/raw/

# 5. Run notebooks in order
jupyter notebook
# 01 → 02 → 03 → 04 → 05
```

> **Note:** Raw data is not committed to this repo. Download via Kaggle as shown above.

---

## Model Performance

| Model | AUC-ROC | F1 (job-seekers) |
|---|---|---|
| Dummy baseline | 0.50 | 0.00 |
| Logistic Regression | 0.79 | 0.59 |
| Random Forest | 0.81 | 0.63 |
| **RF Tuned (final)** | **0.81** | **0.64** |

Evaluated on a stratified 20% hold-out set.  
Primary metric: **AUC-ROC** (chosen for robustness to class imbalance).  
*Raw accuracy is not reported - misleading on a 75/25 imbalanced dataset.*

---

## Dataset

**Source:** [Kaggle - HR Analytics Job Change of Data Scientists](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists)  
**Author:** Arashnic  
**Train size:** 19,158 rows × 14 features  
**Target:** Binary - `1` = actively seeking job change, `0` = not seeking  
**Class balance:** ~75.1% / 24.9%

---

## SQL Highlights

The `sql/hr_queries.sql` file contains 12 business-framed queries demonstrating:

- `RANK()` + `QUALIFY` for top-N filtering without subqueries
- `SUM() OVER (ROWS BETWEEN ...)` for running totals
- `AVG(AVG(...)) OVER ()` for deviation-from-mean analysis
- Two-stage CTEs for segment classification then aggregation
- `TRY_CAST` for safe string-to-numeric conversion on mixed columns
- `COALESCE` for NULL handling in grouping

---

## Key Preprocessing Decisions

| Column | Decision | Reason |
|---|---|---|
| `experience` | Manual ordinal map (`<1`→0, `>20`→21) | LabelEncoder would destroy ordinal meaning |
| `company_type`, `company_size`, `gender` | Unknown category + `was_missing` flag | Missingness correlated with target - capture the signal |
| `training_hours` | `log1p` transform | Right-skewed (skew = 1.82); log normalizes |
| `city` | Dropped | 123 unique values; `city_development_index` captures the signal |
| All imputation stats | Fit on TRAIN only | Prevents data leakage to test set |

---

## Technologies Used

`Python 3.10` · `pandas` · `numpy` · `matplotlib` · `seaborn`  
`scikit-learn` · `SHAP` · `DuckDB` · `joblib`

---
