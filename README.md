# Loan Default Prediction Risk Modeling

End-to-end supervised machine learning project for loan default prediction. The project demonstrates exploratory data analysis, feature review, model comparison, cross-validation, threshold tuning, and model interpretation for an imbalanced binary classification problem.

## Project Objective

Predict `loan_status` using borrower and loan-level features, excluding the identifier column `id`. The workflow starts with a transparent Logistic Regression baseline and progresses to tree-based models.

## Key Results

- Selected model: Gradient Boosting
- Test ROC-AUC: 0.954
- Test PR-AUC: 0.872
- Tuned test F1: 0.808
- Tuned threshold: 0.417
- Test precision: 90.8%
- Test recall: 72.8%

## Analytical Workflow

1. Data quality and structure review
2. Target distribution analysis for class imbalance
3. Numerical feature analysis using correlation and distribution review
4. Categorical feature analysis using segment-level default rates
5. Baseline Logistic Regression model
6. Tree-based model comparison using Random Forest and Gradient Boosting
7. Five-fold cross-validation on the training set using PR-AUC as the main selection metric
8. Out-of-fold threshold tuning to maximize F1
9. Final evaluation on an untouched holdout test set
10. Permutation importance for model interpretation

## Main EDA Insights

- `loan_percent_income` is the strongest numerical risk signal, showing that affordability is central to default risk.
- `loan_int_rate` is strongly associated with default, suggesting pricing is aligned with credit risk.
- `person_income` and `person_emp_length` are directionally protective.
- `loan_grade`, `person_home_ownership`, `loan_intent`, and `cb_person_default_on_file` show meaningful segment-level separation.
- The target is imbalanced, so accuracy alone is not sufficient for model evaluation.

## Repository Structure

```text
.
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_Modeling_Pipeline.ipynb
├── scripts/
│   └── 02_modeling_pipeline.py
├── figures/
│   └── key EDA and model result charts
├── outputs/
│   └── model metrics and feature importance summaries
├── reports/
│   └── Management_Report.pdf
├── data/
│   └── sample.csv is intentionally not included
└── requirements.txt
```

## Data Availability

The original dataset is not included in this public repository due to data sharing restrictions. To reproduce the workflow, place a compatible loan-level dataset at:

```text
data/sample.csv
```

The expected target column is `loan_status`, and the identifier column is `id`.

## How to Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/02_modeling_pipeline.py
```

## Portfolio Notes

This project highlights practical credit-risk modeling skills:

- EDA tied directly to modeling choices
- Proper handling of class imbalance
- Train/test isolation to avoid leakage
- Cross-validation-based model selection
- Threshold tuning based on operational tradeoffs
- Business-oriented model interpretation

Prepared by Juana Zhang.
