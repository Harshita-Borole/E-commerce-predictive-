from __future__ import annotations

from ml import (
    churn_model,
    customer_churn_model,
    delivery_model,
    review_sentiment,
    sales_forecast,
    segmentation,
)
from ml.common import get_logger

log = get_logger("ml.run_all")

BAR = "=" * 78


def main() -> int:
    results = {}

    for name, module in [
        ("Seller Churn (XGBoost Classifier)", churn_model),
        ("Customer Churn (XGBoost Classifier)", customer_churn_model),
        ("Delivery Delay (XGBoost Classifier)", delivery_model),
        ("Sales Forecast (XGBoost Regressor)", sales_forecast),
        ("Customer Segmentation (K-Means)", segmentation),
        ("Review Sentiment (TF-IDF + LogReg)", review_sentiment),
    ]:
        print(f"\n{BAR}\n{name}\n{BAR}")
        try:
            results[name] = module.train()
        except Exception as exc:
            log.error("FAILED: %s — %s", name, exc)
            results[name] = {"error": str(exc)}

    print(f"\n{BAR}\nSUMMARY\n{BAR}")
    for name, report in results.items():
        status = "FAILED" if "error" in report else "OK"
        headline = ""
        m = report.get("metrics", {})
        if name.startswith("Seller Churn"):
            headline = f"ROC-AUC {m.get('roc_auc', 0):.3f}  PR-AUC {m.get('pr_auc', 0):.3f}"
        elif name.startswith("Customer Churn"):
            headline = f"ROC-AUC {m.get('roc_auc', 0):.3f}  top-decile lift {m.get('top_decile_lift', 0):.2f}x"
        elif name.startswith("Delivery Delay"):
            headline = f"ROC-AUC {m.get('roc_auc', 0):.3f}  PR-AUC {m.get('pr_auc', 0):.3f}"
        elif name.startswith("Sales Forecast"):
            beat = m.get("beats_naive_baseline")
            headline = f"beats naive baseline: {beat}"
        elif name.startswith("Customer Segmentation"):
            headline = f"k={m.get('k_selected')}  silhouette={m.get('silhouette_selected', 0):.3f}"
        elif name.startswith("Review Sentiment"):
            headline = f"accuracy {m.get('accuracy', 0):.3f}  macro-F1 {m.get('macro_f1', 0):.3f}"
        print(f"  [{status:6}] {name:<40} {headline}")

    print(f"\nreports  -> docs/ml_reports/*.json")
    print(f"artifacts -> ml/models/*")
    return 0 if all("error" not in r for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
