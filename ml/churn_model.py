from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from ml.common import MODELS_DIR, RANDOM_STATE, get_logger, load, print_metric_block, save_report

warnings.filterwarnings("ignore")
log = get_logger("ml.churn")

TARGET = "is_dormant"

LEAKY_COLS = ["days_since_last_sale", "last_sale", "first_sale", "active_days",
              "orders_per_active_month"]
ID_COLS = ["seller_id", "seller_city"]

CATEGORICAL = ["seller_state"]


def build_features(sp: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = sp.drop(columns=LEAKY_COLS + ID_COLS)
    y = df.pop(TARGET)

    df["seller_state"] = df["seller_state"].fillna("unknown")
    df = pd.get_dummies(df, columns=CATEGORICAL, drop_first=True)

    return df, y


def train() -> dict:
    sp = load("seller_performance")
    X, y = build_features(sp)

    log.info("sellers=%d  features=%d  dormant_rate=%.3f", len(X), X.shape[1], y.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        early_stopping_rounds=25,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "n_sellers": int(len(X)),
        "n_features": int(X.shape[1]),
        "base_rate_dormant": float(y.mean()),
        "test_size": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "best_iteration": int(model.best_iteration),
    }
    print_metric_block("SELLER CHURN — XGBoost Classifier", metrics)
    print("\n" + classification_report(y_test, pred, target_names=["active", "dormant"]))

    importance = (pd.Series(model.feature_importances_, index=X.columns)
                    .sort_values(ascending=False))
    print("\nTop 10 features by importance:")
    print(importance.head(10).to_string())

    model.save_model(MODELS_DIR / "seller_churn_xgb.json")
    X.assign(**{TARGET: y}).to_csv(MODELS_DIR / "seller_churn_features.csv", index=False)

    report = {
        "model": "XGBoost Classifier",
        "target": "is_dormant (seller, no sale in 90 days)",
        "metrics": metrics,
        "top_features": importance.head(10).round(4).to_dict(),
        "excluded_leaky_columns": LEAKY_COLS,
    }
    save_report("seller_churn", report)

    full_proba = model.predict_proba(X)[:, 1]
    scores = sp[["seller_id", "seller_state", "total_revenue", "avg_review_score",
                "late_rate", "is_dormant"]].copy()
    scores["churn_probability"] = full_proba
    scores["risk_band"] = pd.cut(full_proba, [0, .3, .6, 1.0],
                                 labels=["low", "medium", "high"], include_lowest=True)
    scores.to_csv(MODELS_DIR / "seller_churn_scores.csv", index=False)
    log.info("scored %d sellers -> ml/models/seller_churn_scores.csv", len(scores))

    return report


if __name__ == "__main__":
    train()
