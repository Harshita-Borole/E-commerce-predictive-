from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from ml.common import MODELS_DIR, RANDOM_STATE, get_logger, load, print_metric_block, save_report

warnings.filterwarnings("ignore")
log = get_logger("ml.sentiment")

PT_STOPWORDS = [
    "a", "o", "e", "é", "de", "do", "da", "em", "um", "uma", "os", "as",
    "que", "com", "não", "para", "por", "no", "na", "nos", "nas", "se",
    "foi", "muito", "mais", "meu", "minha", "eu", "ele", "ela", "mas",
    "ao", "aos", "das", "dos", "já", "tem", "sua", "seu", "só", "pois",
]


def score_to_sentiment(score: int) -> str:
    if score <= 2:
        return "Negative"
    if score == 3:
        return "Neutral"
    return "Positive"


def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"http\S+|www\S+", " ", s)
    s = re.sub(r"[^a-zà-ÿ\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_dataset() -> pd.DataFrame:
    reviews = load("reviews")
    df = reviews[reviews.review_comment_message.notna()].copy()
    df["text_clean"] = df.review_comment_message.astype(str).map(clean_text)
    df = df[df.text_clean.str.len() >= 3]
    df["sentiment"] = df.review_score.map(score_to_sentiment)
    return df


def train_classifier(df: pd.DataFrame) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        df.text_clean, df.sentiment, test_size=0.2, random_state=RANDOM_STATE,
        stratify=df.sentiment)

    vectorizer = TfidfVectorizer(
        max_features=8000, ngram_range=(1, 2), min_df=3, max_df=0.6,
        stop_words=PT_STOPWORDS, sublinear_tf=True)
    Xt_train = vectorizer.fit_transform(X_train)
    Xt_test = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced",
                               C=1.0, random_state=RANDOM_STATE)
    model.fit(Xt_train, y_train)

    pred = model.predict(Xt_test)
    proba = model.predict_proba(Xt_test)

    metrics = {
        "n_reviews_with_text": int(len(df)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "vocab_size": int(len(vectorizer.vocabulary_)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "class_distribution": df.sentiment.value_counts(normalize=True).round(4).to_dict(),
    }
    print_metric_block("REVIEW SENTIMENT — TF-IDF + Logistic Regression", metrics)
    print("\n" + classification_report(y_test, pred))

    cm = confusion_matrix(y_test, pred, labels=["Negative", "Neutral", "Positive"])
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(pd.DataFrame(cm, index=["Negative", "Neutral", "Positive"],
                       columns=["Negative", "Neutral", "Positive"]).to_string())

    feature_names = np.array(vectorizer.get_feature_names_out())
    print("\nTop terms per class (by model coefficient):")
    for i, cls in enumerate(model.classes_):
        top_idx = np.argsort(model.coef_[i])[-12:][::-1]
        print(f"  {cls:10} {', '.join(feature_names[top_idx])}")

    return {"model": model, "vectorizer": vectorizer, "metrics": metrics,
            "confusion_matrix": cm.tolist()}


def attach_predictions(df: pd.DataFrame, model, vectorizer) -> pd.DataFrame:
    Xt = vectorizer.transform(df.text_clean)
    df = df.copy()
    df["predicted_sentiment"] = model.predict(Xt)
    proba = model.predict_proba(Xt)
    df["predicted_confidence"] = proba.max(axis=1)
    df["agrees_with_rating"] = df.predicted_sentiment == df.sentiment
    return df


def join_category_seller(df: pd.DataFrame) -> pd.DataFrame:
    items = load("order_items")
    products = load("products")
    sellers = load("sellers")

    items = items.merge(products[["product_id", "product_category"]], on="product_id", how="left")
    dominant_category = (items.sort_values("order_id").groupby("order_id").product_category
                              .agg(lambda s: s.value_counts().idxmax() if s.notna().any() else "unknown"))
    dominant_seller = (items.sort_values("order_id").groupby("order_id").seller_id
                            .agg(lambda s: s.value_counts().idxmax()))

    out = df.copy()
    out["product_category"] = out.order_id.map(dominant_category).fillna("unknown")
    out["seller_id"] = out.order_id.map(dominant_seller)
    return out


def find_complaint_topics(df: pd.DataFrame, n_topics: int = 6, n_words: int = 8) -> dict:
    negative = df[df.predicted_sentiment == "Negative"]
    if len(negative) < 50:
        return {"note": "too few negative-predicted reviews for stable topics"}

    vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=5, max_df=0.5,
                          stop_words=PT_STOPWORDS)
    Xt = vec.fit_transform(negative.text_clean)

    nmf = NMF(n_components=n_topics, random_state=RANDOM_STATE, max_iter=400)
    W = nmf.fit_transform(Xt)
    feature_names = np.array(vec.get_feature_names_out())

    topics = {}
    dominant_topic = W.argmax(axis=1)
    for t in range(n_topics):
        top_idx = np.argsort(nmf.components_[t])[-n_words:][::-1]
        words = feature_names[top_idx].tolist()
        count = int((dominant_topic == t).sum())
        topics[f"topic_{t}"] = {"top_words": words, "review_count": count}

    return {"n_negative_reviews": int(len(negative)), "topics": topics}


def train() -> dict:
    df = build_dataset()
    log.info("reviews with usable text: %d", len(df))

    clf = train_classifier(df)
    scored = attach_predictions(df, clf["model"], clf["vectorizer"])
    scored = join_category_seller(scored)

    agreement_rate = float(scored.agrees_with_rating.mean())
    print(f"\nText-predicted sentiment agrees with star-rating-derived label: "
          f"{agreement_rate*100:.1f}% of reviews")

    disagreements = scored[~scored.agrees_with_rating]
    interesting = disagreements[(disagreements.sentiment == "Positive")
                                & (disagreements.predicted_sentiment == "Negative")]
    print(f"Positive-rated (4-5 star) but text reads Negative: {len(interesting)} cases "
          "(possible sarcasm, or a problem later resolved before rating)")

    overall_pct = (scored.predicted_sentiment.value_counts(normalize=True) * 100).round(1)
    print_metric_block("OVERALL SENTIMENT (reviews with text, n={:,})".format(len(scored)),
                       overall_pct.to_dict())

    by_category = (scored.groupby("product_category")
                         .agg(reviews=("order_id", "size"),
                              avg_rating=("review_score", "mean"),
                              negative_pct=("predicted_sentiment",
                                           lambda s: 100 * (s == "Negative").mean()))
                         .round(2))
    by_category = by_category[by_category.reviews >= 30].sort_values("negative_pct", ascending=False)
    by_category_real = by_category.drop(index="unknown", errors="ignore")
    print("\nWorst 10 categories by text-predicted negative %  (min 30 reviews):")
    print(by_category_real.head(10).to_string())

    by_seller = (scored.groupby("seller_id")
                       .agg(reviews=("order_id", "size"),
                            avg_rating=("review_score", "mean"),
                            negative_pct=("predicted_sentiment",
                                         lambda s: 100 * (s == "Negative").mean()))
                       .round(2))
    by_seller = by_seller[by_seller.reviews >= 20].sort_values("negative_pct", ascending=False)
    print("\nWorst 10 sellers by text-predicted negative %  (min 20 reviews):")
    print(by_seller.head(10).to_string())

    topics = find_complaint_topics(scored)
    print(f"\n{'-'*70}\nCOMPLAINT TOPICS (NMF over text-predicted-negative reviews)\n{'-'*70}")
    if "topics" in topics:
        for name, t in topics["topics"].items():
            print(f"  {name} ({t['review_count']} reviews): {', '.join(t['top_words'])}")

    import joblib
    joblib.dump({"model": clf["model"], "vectorizer": clf["vectorizer"]},
               MODELS_DIR / "review_sentiment_model.pkl")
    scored[["order_id", "review_score", "sentiment", "predicted_sentiment",
           "predicted_confidence", "agrees_with_rating", "product_category",
           "seller_id"]].to_csv(MODELS_DIR / "review_sentiment_scores.csv", index=False)
    by_category.to_csv(MODELS_DIR / "review_sentiment_by_category.csv")
    by_seller.to_csv(MODELS_DIR / "review_sentiment_by_seller.csv")

    report = {
        "model": "TF-IDF + Logistic Regression (Portuguese, 3-class)",
        "coverage_note": (
            f"Covers reviews WITH text only ({len(df):,} of 98,673, 41.3%). "
            "review_score exists for all reviews and is a complete signal on "
            "its own (Module 5, SQL Query 18) — this model's contribution is "
            "sentiment learned from what customers actually WROTE, not a "
            "restatement of the rating they already gave."
        ),
        "metrics": clf["metrics"],
        "confusion_matrix": clf["confusion_matrix"],
        "rating_agreement_rate": agreement_rate,
        "overall_sentiment_pct": overall_pct.to_dict(),
        "worst_categories": by_category_real.head(10).reset_index().to_dict("records"),
        "worst_sellers": by_seller.head(10).reset_index().to_dict("records"),
        "complaint_topics": topics,
    }
    save_report("review_sentiment", report)
    return report


if __name__ == "__main__":
    train()
