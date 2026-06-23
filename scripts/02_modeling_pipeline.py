"""
Supervised Machine Learning Model - Loan Default Prediction

Author: Juana Zhang
Dataset: sample.csv
Target variable: loan_status

Technical guidelines implemented in this script:
    - Use loan_status as the target variable.
    - Exclude id from the independent variables.
    - Use all remaining non-ID columns as candidate predictors.
    - Begin with a baseline Logistic Regression model.
    - Progress to tree-based models and compare performance.

Modeling approach:
    1. Load and validate the dataset.
    2. Split features into numerical and categorical groups.
    3. Apply scaling for Logistic Regression and one-hot encoding for all
       categorical variables.
    4. Train Logistic Regression, Random Forest, and Gradient Boosting models.
    5. Evaluate models with metrics appropriate for an imbalanced target.
    6. Select the best model by PR-AUC and generate feature importance outputs.

Run from project root:
    python scripts/02_modeling_pipeline.py

Outputs:
    - outputs/model_metrics.csv
    - outputs/cross_validation_metrics.csv
    - outputs/feature_importance.csv
    - outputs/best_model.joblib
    - outputs/modeling_summary.json
    - figures/*.png
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)


DATA_PATH = ROOT / "data" / "sample.csv"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = ROOT / "figures"
RANDOM_STATE = 42


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    (ROOT / ".matplotlib-cache").mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset is not included in this public portfolio repository. "
            "Place a compatible loan-level CSV at data/sample.csv to reproduce the analysis."
        )
    df = pd.read_csv(DATA_PATH)
    required = {"id", "loan_status"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def get_feature_groups(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Separate model inputs into numerical and categorical feature groups."""
    categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object", "string"]).columns.tolist()
    return numeric_features, categorical_features


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric_features, categorical_features = get_feature_groups(X)

    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def evaluate_model(name: str, estimator: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    probabilities = estimator.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    return {
        "model": name,
        "threshold": 0.5,
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_best_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, dict]:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1_scores = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_idx = int(np.nanargmax(f1_scores[:-1]))
    threshold = float(thresholds[best_idx])
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    metrics = {
        "threshold": threshold,
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return threshold, metrics


def plot_eda(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(7, 4))
    target_counts = df["loan_status"].value_counts(normalize=True).sort_index() * 100
    sns.barplot(x=target_counts.index.astype(str), y=target_counts.values, ax=ax, color="#4C78A8")
    ax.set_title("Loan Status Distribution")
    ax.set_xlabel("Loan Status")
    ax.set_ylabel("Share of Records (%)")
    for i, value in enumerate(target_counts.values):
        ax.text(i, value + 0.6, f"{value:.1f}%", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "target_distribution.png", dpi=180)
    plt.close(fig)

    categorical = ["loan_grade", "person_home_ownership", "loan_intent", "cb_person_default_on_file"]
    for col in categorical:
        rates = df.groupby(col)["loan_status"].agg(["count", "mean"]).sort_values("mean", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(x=rates.index.astype(str), y=rates["mean"] * 100, ax=ax, color="#4C78A8")
        ax.set_title(f"Default Rate by {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Default Rate (%)")
        ax.tick_params(axis="x", rotation=25)
        for i, value in enumerate(rates["mean"] * 100):
            ax.text(i, value + 0.7, f"{value:.1f}%", ha="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"default_rate_by_{col}.png", dpi=180)
        plt.close(fig)

    numeric_cols = [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
    ]
    corr = df[numeric_cols + ["loan_status"]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Numerical Feature Correlation")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "correlation_heatmap.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.kdeplot(data=df, x="loan_percent_income", hue="loan_status", common_norm=False, fill=True, alpha=0.35, ax=ax)
    ax.set_title("Loan-to-Income Burden by Loan Status")
    ax.set_xlabel("Loan Amount / Annual Income")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "loan_percent_income_distribution.png", dpi=180)
    plt.close(fig)


def plot_model_outputs(metrics: pd.DataFrame, feature_importance: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    display = metrics.melt(id_vars="model", value_vars=["roc_auc", "pr_auc", "f1"], var_name="metric", value_name="score")
    sns.barplot(data=display, x="model", y="score", hue="metric", ax=ax, palette=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_title("Model Performance Comparison")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)

    top = feature_importance.head(12).copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=top, y="feature", x="importance", ax=ax, color="#4C78A8")
    ax.set_title("Top Predictive Drivers")
    ax.set_xlabel("Permutation Importance - ROC AUC Decrease")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "feature_importance.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    df = load_data()
    plot_eda(df)

    # Assignment requirement: use all non-ID variables as predictors.
    X = df.drop(columns=["id", "loan_status"])
    y = df["loan_status"]
    numeric_features, categorical_features = get_feature_groups(X)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train_full
    )

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X, scale_numeric=True)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X, scale_numeric=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=350,
                        min_samples_leaf=10,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X, scale_numeric=False)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=300,
                        learning_rate=0.06,
                        l2_regularization=0.03,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_rows = []
    test_rows = []
    trained_models = {}

    for name, pipeline in models.items():
        cv_result = cross_validate(
            pipeline,
            X_train_full,
            y_train_full,
            cv=cv,
            scoring=["roc_auc", "average_precision", "f1", "recall", "precision"],
            n_jobs=1,
        )
        cv_rows.append(
            {
                "model": name,
                "cv_roc_auc_mean": cv_result["test_roc_auc"].mean(),
                "cv_roc_auc_std": cv_result["test_roc_auc"].std(),
                "cv_pr_auc_mean": cv_result["test_average_precision"].mean(),
                "cv_pr_auc_std": cv_result["test_average_precision"].std(),
                "cv_f1_mean": cv_result["test_f1"].mean(),
                "cv_recall_mean": cv_result["test_recall"].mean(),
                "cv_precision_mean": cv_result["test_precision"].mean(),
            }
        )
        pipeline.fit(X_train_full, y_train_full)
        trained_models[name] = pipeline
        test_rows.append(evaluate_model(name, pipeline, X_test, y_test))

    metrics = pd.DataFrame(test_rows).sort_values("pr_auc", ascending=False)
    cv_metrics = pd.DataFrame(cv_rows).sort_values("cv_pr_auc_mean", ascending=False)
    best_model_name = metrics.iloc[0]["model"]
    best_model = trained_models[best_model_name]

    threshold_model = models[best_model_name]
    threshold_model.fit(X_train, y_train)
    valid_probabilities = threshold_model.predict_proba(X_valid)[:, 1]
    tuned_threshold, validation_threshold_metrics = find_best_f1_threshold(y_valid, valid_probabilities)
    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    test_predictions_tuned = (test_probabilities >= tuned_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_predictions_tuned).ravel()
    tuned_test_metrics = {
        "threshold": tuned_threshold,
        "precision": precision_score(y_test, test_predictions_tuned, zero_division=0),
        "recall": recall_score(y_test, test_predictions_tuned, zero_division=0),
        "f1": f1_score(y_test, test_predictions_tuned, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    perm = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=8,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    feature_importance = (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    plot_model_outputs(metrics, feature_importance)

    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    cv_metrics.to_csv(OUTPUT_DIR / "cross_validation_metrics.csv", index=False)
    feature_importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    joblib.dump(best_model, OUTPUT_DIR / "best_model.joblib")

    categorical_rates = {}
    for col in X.select_dtypes(include=["object", "string"]).columns:
        categorical_rates[col] = (
            df.groupby(col)["loan_status"]
            .agg(count="count", default_rate="mean")
            .sort_values("default_rate", ascending=False)
            .reset_index()
            .to_dict(orient="records")
        )

    summary = {
        "data": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "target_positive_count": int(y.sum()),
            "target_negative_count": int((1 - y).sum()),
            "target_positive_rate": float(y.mean()),
            "missing_values_total": int(df.isna().sum().sum()),
            "duplicate_ids": int(df["id"].duplicated().sum()),
            "duplicate_rows_excluding_id": int(df.drop(columns=["id"]).duplicated().sum()),
        },
        "feature_setup": {
            "target": "loan_status",
            "excluded_columns": ["id"],
            "predictor_count": int(X.shape[1]),
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "categorical_encoding": "OneHotEncoder(handle_unknown='ignore')",
        },
        "numerical_summary": df.drop(columns=["id", "loan_status"]).describe().T.round(4).to_dict(orient="index"),
        "categorical_default_rates": categorical_rates,
        "target_correlations": df.drop(columns=["id"]).corr(numeric_only=True)["loan_status"]
        .drop("loan_status")
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .round(6)
        .to_dict(),
        "model_metrics": metrics.to_dict(orient="records"),
        "cross_validation_metrics": cv_metrics.to_dict(orient="records"),
        "best_model": best_model_name,
        "validation_threshold_metrics": validation_threshold_metrics,
        "tuned_test_metrics": tuned_test_metrics,
        "feature_importance": feature_importance.to_dict(orient="records"),
    }
    with open(OUTPUT_DIR / "modeling_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Target variable: loan_status")
    print("Excluded column: id")
    print("Predictors used:", list(X.columns))
    print("Numerical features:", numeric_features)
    print("Categorical features:", categorical_features)
    print("Best model:", best_model_name)
    print(metrics.round(4).to_string(index=False))
    print("Tuned threshold test metrics:", {k: round(v, 4) if isinstance(v, float) else v for k, v in tuned_test_metrics.items()})


if __name__ == "__main__":
    main()
