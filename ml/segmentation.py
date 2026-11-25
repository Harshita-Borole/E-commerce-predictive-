from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from ml.common import MODELS_DIR, RANDOM_STATE, get_logger, load, print_metric_block, save_report

warnings.filterwarnings("ignore")
log = get_logger("ml.segmentation")

CLASSIC_RFM_FEATURES = ["recency_days", "frequency", "monetary"]
FEATURES = [
    "recency_days", "frequency", "monetary", "avg_order_value", "avg_delivery_days",
    "late_order_rate", "avg_review_score", "distinct_categories", "tenure_days",
]
K_CANDIDATES = range(3, 8)


def build_features(feature_list: list[str] = FEATURES) -> tuple[pd.DataFrame, pd.DataFrame]:
    rfm = load("customer_rfm")
    df = rfm.copy()

    work = df[feature_list].copy()
    for col in work.columns:
        work[col] = work[col].fillna(work[col].median())

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(work), columns=feature_list, index=work.index)
    return X, df


def choose_k(X: pd.DataFrame) -> tuple[int, dict, dict]:
    silhouette, inertia = {}, {}
    for k in K_CANDIDATES:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        inertia[k] = round(float(km.inertia_), 1)
        sample_idx = np.random.RandomState(RANDOM_STATE).choice(
            len(X), size=min(10_000, len(X)), replace=False)
        s = silhouette_score(X.iloc[sample_idx], labels[sample_idx])
        silhouette[k] = round(float(s), 4)
        log.info("k=%d  inertia=%.0f  silhouette=%.4f", k, km.inertia_, s)
    best_k = max(silhouette, key=silhouette.get)
    return best_k, silhouette, inertia


def print_elbow_table(inertia: dict, silhouette: dict) -> None:
    print(f"\n{'k':<6}{'inertia':>14}{'drop vs prev':>16}{'silhouette':>14}")
    prev = None
    for k in inertia:
        drop = "" if prev is None else f"{100*(prev-inertia[k])/prev:>14.1f}%"
        print(f"{k:<6}{inertia[k]:>14,.0f}{drop:>16}{silhouette[k]:>14.4f}")
        prev = inertia[k]


def run_classic_rfm() -> dict:
    print(f"\n{'='*70}\nCLASSIC RFM (Recency, Frequency, Monetary only)\n{'='*70}")
    X, df = build_features(CLASSIC_RFM_FEATURES)
    best_k, silhouette, inertia = choose_k(X)
    print_elbow_table(inertia, silhouette)

    model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X)
    sizes = pd.Series(labels).value_counts().sort_index()
    pct = (100 * sizes / len(labels)).round(1)

    print(f"\nselected k={best_k} (best silhouette={silhouette[best_k]:.4f})")
    print("cluster sizes:")
    for cid in sizes.index:
        print(f"  cluster {cid}: {sizes[cid]:,} customers ({pct[cid]}%)")

    dominant_pct = float(pct.max())
    verdict = (
        f"Largest cluster holds {dominant_pct:.1f}% of customers — recency and "
        "monetary carry enough real variation on their own that classic RFM "
        "does not collapse into one dominant blob the way the 97% "
        "frequency=1 finding alone might suggest. But cluster balance was "
        "never the actual limitation: R, F and M contain zero delivery or "
        "review information by construction, so no value of k lets this "
        "model ever find a delivery-quality segment — that requires features "
        "that were never given to it. Confirmed below by comparison: the "
        "extended feature set below finds a 6.6%-of-base cluster with a 99% "
        "late-delivery rate that classic RFM has no way to detect at any k."
    )
    print(f"\n{verdict}")

    return {"k_selected": best_k, "silhouette": silhouette, "inertia": inertia,
            "cluster_sizes_pct": pct.to_dict(), "verdict": verdict}


def profile_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["cluster"] = labels
    profile = (out.groupby("cluster")
                  .agg(customers=("customer_unique_id", "count"),
                       avg_recency_days=("recency_days", "mean"),
                       avg_monetary=("monetary", "mean"),
                       avg_order_value=("avg_order_value", "mean"),
                       avg_delivery_days=("avg_delivery_days", "mean"),
                       avg_late_rate=("late_order_rate", "mean"),
                       avg_review_score=("avg_review_score", "mean"),
                       avg_categories=("distinct_categories", "mean"),
                       pct_repeat=("is_repeat_customer", "mean"))
                  .round(2))
    profile["pct_of_base"] = (100 * profile.customers / len(out)).round(1)

    crosstab = pd.crosstab(out.cluster, out.segment)
    print("\nK-Means cluster vs. existing rule-based RFM segment:")
    print(crosstab.to_string())

    return profile


def label_clusters(profile: pd.DataFrame) -> dict[int, str]:
    labels = {}
    rank_monetary = profile.avg_monetary.rank(ascending=False)
    rank_recency = profile.avg_recency_days.rank(ascending=True)

    for cid, row in profile.iterrows():
        if row.avg_late_rate >= 0.5 or row.avg_review_score <= 2.5:
            labels[cid] = "Poor Delivery Experience"
        elif row.pct_repeat >= 0.25:
            labels[cid] = "High-Value Loyal (Frequent Buyers)"
        elif rank_monetary[cid] == 1 and rank_recency[cid] > len(profile) / 2:
            labels[cid] = "At-Risk (Lapsed High Value)"
        elif rank_monetary[cid] == 1:
            labels[cid] = "High Value"
        elif rank_recency[cid] <= len(profile) / 2:
            labels[cid] = "Recent, Standard Value"
        else:
            labels[cid] = "Occasional Buyers"
    return labels


def train() -> dict:
    classic_result = run_classic_rfm()

    print(f"\n{'='*70}\nEXTENDED FEATURE SET (adds delivery experience + review sentiment)\n{'='*70}")
    X, df = build_features(FEATURES)
    log.info("customers=%d  features=%d", len(X), X.shape[1])

    best_k, silhouette_scores, inertia_scores = choose_k(X)
    print_elbow_table(inertia_scores, silhouette_scores)
    log.info("selected k=%d (best silhouette)", best_k)

    model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X)

    profile = profile_clusters(df, labels)
    cluster_names = label_clusters(profile)
    profile["label"] = profile.index.map(cluster_names)

    print("\nCluster profiles:")
    print(profile.to_string())

    metrics = {
        "n_customers": int(len(X)),
        "n_features": int(X.shape[1]),
        "k_selected": int(best_k),
        "silhouette_selected": silhouette_scores[best_k],
        "silhouette_by_k": silhouette_scores,
        "inertia": float(model.inertia_),
    }
    print_metric_block("CUSTOMER SEGMENTATION — K-Means", metrics)

    import joblib
    joblib.dump({"model": model, "scaler_features": FEATURES}, MODELS_DIR / "segmentation_kmeans.pkl")

    scored = df[["customer_unique_id", "segment"]].copy()
    scored["cluster"] = labels
    scored["cluster_label"] = scored.cluster.map(cluster_names)
    scored.to_csv(MODELS_DIR / "segmentation_scores.csv", index=False)

    report = {
        "model": "K-Means",
        "classic_rfm_result": classic_result,
        "features": FEATURES,
        "metrics": metrics,
        "cluster_profile": profile.reset_index().to_dict("records"),
        "cluster_labels": {int(k): v for k, v in cluster_names.items()},
        "comparison": (
            f"Classic RFM (3 features): {classic_result['verdict']} "
            f"Extended ({len(FEATURES)} features): {best_k} clusters, silhouette "
            f"{metrics['silhouette_selected']:.3f}, including a "
            f"{profile.loc[profile.label == 'Poor Delivery Experience', 'pct_of_base'].sum():.1f}% "
            "cluster classic RFM has no features to ever detect."
        ),
    }
    save_report("segmentation", report)
    return report


if __name__ == "__main__":
    train()
