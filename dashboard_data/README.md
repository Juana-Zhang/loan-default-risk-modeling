# Tableau Public Dashboard Data

This folder contains aggregated, dashboard-ready data for a public Tableau portfolio dashboard.

The original row-level dataset is intentionally not included. These CSV files are derived summary tables intended to support visualization without exposing borrower-level records.

## Recommended Dashboard Sections

### 1. Executive Overview

Use:
- `overview_metrics.csv`
- `target_distribution.csv`

Suggested visuals:
- KPI cards for records, default rate, selected model, ROC-AUC, PR-AUC, F1
- Donut or bar chart for default vs. non-default share

### 2. Borrower and Loan Risk Signals

Use:
- `numeric_correlations.csv`
- `binned_numeric_default_rates.csv`

Suggested visuals:
- Bar chart of correlations to default
- Line or bar chart of default rate by numeric buckets, especially:
  - `loan_percent_income`
  - `loan_int_rate`
  - `person_income`
  - `person_emp_length`

### 3. Segment-Level Risk

Use:
- `segment_default_rates.csv`

Suggested visuals:
- Default rate by loan grade
- Default rate by home ownership
- Default rate by loan intent
- Default rate by prior default file

### 4. Model Performance

Use:
- `model_cv_comparison.csv`
- `final_model_metrics.csv`

Suggested visuals:
- Model comparison bar chart using CV ROC-AUC and CV PR-AUC
- Final test KPI cards for precision, recall, F1, ROC-AUC, and PR-AUC

### 5. Model Interpretation

Use:
- `feature_importance.csv`

Suggested visual:
- Horizontal bar chart of top predictive drivers

## Public Sharing Note

These files are suitable for a public portfolio dashboard because they are aggregated and do not contain individual-level borrower records. Do not upload the original `sample.csv` to Tableau Public.
