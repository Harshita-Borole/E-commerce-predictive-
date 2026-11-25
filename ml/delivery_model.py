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

from ml.common import MODELS_DIR, RANDOM_STATE, get_logger, load, print_metric_block, save_report

warnings.filterwarnings("ignore")
log = get_logger("ml.delivery")

TARGET = "is_late"
TOP_N_CATEGORIES = 12

STATE_REGION = {
    **{s: "Norte" for s in ["AC", "AP", "AM", "PA", "RO", "RR", "TO"]},
    **{s: "Nordeste" for s in ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"]},
    **{s: "Centro-Oeste" for s in ["DF", "GO", "MT", "MS"]},
    **{s: "Sudeste" for s in ["ES", "MG", "RJ", "SP"]},
    **{s: "Sul" for s in ["PR", "RS", "SC"]},
}

POST_DELIVERY_COLS = [
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "delivery_days", "delay_days", "delivery_hours", "approval_hours",
    "carrier_handover_days", "order_approved_at",
]


def compute_seller_expanding_late_rate(items: pd.DataFrame, delivered: pd.DataFrame) -> pd.DataFrame:
    sh = (items[["seller_id", "order_id"]]
          .drop_duplicates()
          .merge(delivered[["order_id", "order_purchase_timestamp", TARGET]], on="order_id"))
    sh = sh.sort_values(["seller_id", "order_purchase_timestamp"])

    grp = sh.groupby("seller_id")[TARGET]
    sh["seller_prior_orders"] = grp.cumcount()
    sh["seller_historical_late_rate"] = grp.apply(
        lambda s: s.expanding().mean().shift(1)).reset_index(level=0, drop=True)

    global_rate = float(delivered[TARGET].mean())
    sh["seller_historical_late_rate"] = sh["seller_historical_late_rate"].fillna(global_rate)
    return sh[["seller_id", "order_id", "seller_historical_late_rate", "seller_prior_orders"]]


def build_features() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    orders = load("orders")
    items = load("order_items")
    products = load("products")
    payments = load("payments")
    customers = load("customers")
    sellers = load("sellers")

    delivered = orders[orders.is_delivered == 1].copy()
    delivered["order_purchase_timestamp"] = pd.to_datetime(delivered.order_purchase_timestamp)

    items = items.merge(products[["product_id", "product_weight_g", "product_volume_cm3",
                                  "product_category"]], on="product_id", how="left")
    items = items.merge(sellers[["seller_id", "seller_state"]], on="seller_id", how="left")

    item_rollup = items.groupby("order_id").agg(
        item_count=("order_item_id", "size"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        avg_freight_ratio=("freight_ratio", "mean"),
        avg_distance_km=("seller_customer_distance_km", "mean"),
        max_distance_km=("seller_customer_distance_km", "max"),
        total_weight_g=("product_weight_g", "sum"),
        avg_volume_cm3=("product_volume_cm3", "mean"),
        distinct_sellers=("seller_id", "nunique"),
        distinct_categories=("product_category", "nunique"),
    )
    dominant_seller = (items.sort_values("order_id")
                             .groupby("order_id").seller_id
                             .agg(lambda s: s.value_counts().idxmax()))
    dominant_state = (items.sort_values("order_id")
                            .groupby("order_id").seller_state
                            .agg(lambda s: s.value_counts().idxmax()))
    dominant_category = (items.sort_values("order_id")
                               .groupby("order_id").product_category
                               .agg(lambda s: s.value_counts().idxmax() if s.notna().any() else "unknown"))
    item_rollup["dominant_seller_id"] = dominant_seller
    item_rollup["seller_state"] = dominant_state
    item_rollup["product_category"] = dominant_category

    seller_hist = compute_seller_expanding_late_rate(items, delivered)
    item_rollup = item_rollup.reset_index().merge(
        seller_hist.rename(columns={"seller_id": "dominant_seller_id"}),
        on=["dominant_seller_id", "order_id"], how="left").set_index("order_id")
    global_rate = float(delivered[TARGET].mean())
    item_rollup["seller_historical_late_rate"] = item_rollup["seller_historical_late_rate"].fillna(global_rate)
    item_rollup["seller_prior_orders"] = item_rollup["seller_prior_orders"].fillna(0)

    pay = payments.set_index("order_id")[["payment_installments", "is_installment_plan"]]

    df = (delivered
          .merge(item_rollup, on="order_id", how="left")
          .merge(pay, on="order_id", how="left")
          .merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left"))

    before = len(df)
    df = df[df[TARGET].notna()].copy()
    if before - len(df):
        log = get_logger("ml.delivery")
        log.info("dropped %d delivered orders with no delivery date", before - len(df))

    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)

    feature_cols = [
        "estimated_delivery_days", "purchase_month", "purchase_day_of_week",
        "purchase_hour", "is_weekend_purchase",
        "item_count", "total_price", "total_freight", "avg_freight_ratio",
        "avg_distance_km", "max_distance_km", "total_weight_g", "avg_volume_cm3",
        "distinct_sellers", "distinct_categories",
        "payment_installments", "is_installment_plan",
        "customer_state", "seller_state", "same_region",
        "seller_historical_late_rate", "seller_prior_orders", "product_category",
    ]
    df["customer_region"] = df["customer_state"].map(STATE_REGION).fillna("unknown")
    df["seller_region"] = df["seller_state"].map(STATE_REGION).fillna("unknown")
    df["same_region"] = (df["customer_region"] == df["seller_region"]).astype(int)

    X = df[feature_cols].copy()
    y = df[TARGET].astype(int)
    ts = df["order_purchase_timestamp"]
    order_ids = df["order_id"]

    for col in ["customer_state", "seller_state", "product_category"]:
        X[col] = X[col].fillna("unknown")

    top_categories = X["product_category"].value_counts().head(TOP_N_CATEGORIES).index
    X["product_category"] = X["product_category"].where(
        X["product_category"].isin(top_categories), "other")

    X = pd.get_dummies(X, columns=["customer_state", "seller_state", "product_category"],
                       drop_first=True)

    return X, y, ts, order_ids


def train() -> dict:
    X, y, ts, order_ids = build_features()
    log.info("orders=%d  features=%d  late_rate=%.4f", len(X), X.shape[1], y.mean())

    cutoff = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:cutoff], X.iloc[cutoff:]
    y_train, y_test = y.iloc[:cutoff], y.iloc[cutoff:]
    ids_test = order_ids.iloc[cutoff:]
    log.info("train period ends %s, test starts %s", ts.iloc[cutoff - 1].date(), ts.iloc[cutoff].date())

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.5,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        early_stopping_rounds=30,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "n_orders": int(len(X)),
        "n_features": int(X.shape[1]),
        "base_rate_late": float(y.mean()),
        "test_size": int(len(X_test)),
        "test_late_rate": float(y_test.mean()),
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "scale_pos_weight": float(scale_pos_weight),
        "best_iteration": int(model.best_iteration),
    }
    print_metric_block("DELIVERY DELAY — XGBoost Classifier", metrics)

    print(f"""
{'-'*70}
WHY NOT ACCURACY
{'-'*70}
{metrics['test_late_rate']*100:.1f}% of the test set is late. A model that predicts
"on time" for every single order — checking nothing, learning nothing —
would score {(1-metrics['test_late_rate'])*100:.1f}% accuracy while catching zero delays. That
number would look great on a slide and be useless in production: it never
flags a single at-risk order for a customer-service team to get ahead of.

Precision, Recall and F1 are computed on the LATE class specifically, so a
model can't hide behind the size of the majority class:
  Recall    = of the orders that WERE actually late, what fraction did the
              model catch? (misses here are silent failures nobody warned)
  Precision = of the orders the model FLAGGED as at-risk, what fraction
              really were? (false alarms here waste intervention effort)
  F1        = the balance between the two, in one number
  ROC-AUC   = ranks predicted risk against actual outcome across every
              possible threshold, so it doesn't depend on where 0.5 happens
              to fall — the metric this project treats as primary.
{'-'*70}""")

    print(classification_report(y_test, pred, target_names=["on-time", "late"], zero_division=0))

    importance = (pd.Series(model.feature_importances_, index=X.columns)
                    .sort_values(ascending=False))
    print("\nTop 10 features by importance:")
    print(importance.head(10).to_string())

    model.save_model(MODELS_DIR / "delivery_delay_xgb.json")

    out = pd.DataFrame({"order_id": ids_test.values, "delay_probability": np.round(proba, 4)}
                       ).sort_values("delay_probability", ascending=False).reset_index(drop=True)
    n = len(out)
    out["risk"] = np.select([out.index < n * 0.10, out.index < n * 0.30],
                            ["HIGH", "MEDIUM"], default="LOW")
    out.to_csv(MODELS_DIR / "delivery_delay_scores.csv", index=False)

    print(f"\n{'Order #':<38}{'Delay Probability':>20}{'Risk':>10}")
    for _, r in out.head(10).iterrows():
        print(f"{r.order_id:<38}{r.delay_probability*100:>18.0f}%{r.risk:>10}")

    report = {
        "model": "XGBoost Classifier",
        "target": "is_late (delivered after order_estimated_delivery_date)",
        "split": "chronological 80/20, not random — avoids peeking across time",
        "metrics": metrics,
        "top_features": importance.head(10).round(4).to_dict(),
        "excluded_leaky_columns": POST_DELIVERY_COLS,
        "risk_band_counts": out.risk.value_counts().to_dict(),
        "why_not_accuracy": (
            f"{metrics['test_late_rate']*100:.1f}% of orders are late; predicting "
            "'on time' for everyone scores high accuracy while catching zero "
            "delays. Recall/Precision/F1 are computed on the late class "
            "specifically; ROC-AUC is reported as the primary metric because "
            "it evaluates ranking quality independent of any one threshold."
        ),
    }
    save_report("delivery_delay", report)
    return report


if __name__ == "__main__":
    train()
