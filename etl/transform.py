from __future__ import annotations

import logging
import unicodedata

import numpy as np
import pandas as pd

from etl import config as cfg

log = logging.getLogger("etl.transform")

EARTH_RADIUS_KM = 6371.0


def _strip_accents(s: pd.Series) -> pd.Series:
    return (s.fillna("")
             .astype(str)
             .str.normalize("NFKD")
             .str.encode("ascii", errors="ignore")
             .str.decode("utf-8"))


def standardise_city(s: pd.Series) -> pd.Series:
    out = _strip_accents(s).str.lower().str.strip()
    out = out.str.replace(r"[^a-z0-9 ]+", " ", regex=True)
    out = out.str.replace(r"\s+", " ", regex=True).str.strip()
    return out.replace("", np.nan)


def _modal_by_group(df: pd.DataFrame, key: str, value: str) -> pd.Series:
    counts = df.groupby([key, value], observed=True).size().reset_index(name="_n")
    counts = counts.sort_values("_n", ascending=False).drop_duplicates(key)
    return counts.set_index(key)[value]


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class Quarantine:
    def __init__(self) -> None:
        self.frames: list[pd.DataFrame] = []
        self.counts: dict[str, int] = {}

    def reject(self, table: str, rows: pd.DataFrame, reason: str) -> None:
        if rows.empty:
            return
        rec = rows.copy()
        rec.insert(0, "_reject_reason", reason)
        rec.insert(0, "_source_table", table)
        self.frames.append(rec)
        key = f"{table}: {reason}"
        self.counts[key] = self.counts.get(key, 0) + len(rows)
        log.warning("quarantined %s row(s) — %s", f"{len(rows):,}", key)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def to_frame(self) -> pd.DataFrame:
        if not self.frames:
            return pd.DataFrame(columns=["_source_table", "_reject_reason"])
        return pd.concat(self.frames, ignore_index=True)


def clean_geolocation(geo: pd.DataFrame) -> pd.DataFrame:
    before = len(geo)
    g = geo.copy()
    g["geolocation_city"] = standardise_city(g["geolocation_city"])
    g["geolocation_state"] = g["geolocation_state"].str.upper().str.strip()

    centroid = (g.groupby("geolocation_zip_code_prefix")
                 .agg(geo_lat=("geolocation_lat", "median"),
                      geo_lng=("geolocation_lng", "median"),
                      geo_ping_count=("geolocation_lat", "size")))

    centroid["geo_city"] = _modal_by_group(g, "geolocation_zip_code_prefix", "geolocation_city")
    centroid["geo_state"] = _modal_by_group(g, "geolocation_zip_code_prefix", "geolocation_state")

    out = centroid.reset_index().rename(
        columns={"geolocation_zip_code_prefix": "zip_code_prefix"})
    log.info("geolocation collapsed %s -> %s rows (one per zip prefix)",
             f"{before:,}", f"{len(out):,}")
    return out


def clean_customers(customers: pd.DataFrame, geo: pd.DataFrame) -> pd.DataFrame:
    c = customers.copy()
    c["customer_city"] = standardise_city(c["customer_city"])
    c["customer_state"] = c["customer_state"].str.upper().str.strip()

    c = c.merge(geo[["zip_code_prefix", "geo_lat", "geo_lng"]],
                left_on="customer_zip_code_prefix", right_on="zip_code_prefix",
                how="left").drop(columns="zip_code_prefix")
    c = c.rename(columns={"geo_lat": "customer_lat", "geo_lng": "customer_lng"})
    log.info("customers cleaned — %s rows, %s unique people",
             f"{len(c):,}", f"{c[cfg.PERSON_KEY].nunique():,}")
    return c


def clean_sellers(sellers: pd.DataFrame, geo: pd.DataFrame) -> pd.DataFrame:
    s = sellers.copy()
    s["seller_city"] = standardise_city(s["seller_city"])
    s["seller_state"] = s["seller_state"].str.upper().str.strip()
    s = s.merge(geo[["zip_code_prefix", "geo_lat", "geo_lng"]],
                left_on="seller_zip_code_prefix", right_on="zip_code_prefix",
                how="left").drop(columns="zip_code_prefix")
    s = s.rename(columns={"geo_lat": "seller_lat", "geo_lng": "seller_lng"})
    log.info("sellers cleaned — %s rows", f"{len(s):,}")
    return s


def clean_products(products: pd.DataFrame, translation: pd.DataFrame) -> pd.DataFrame:
    p = products.copy()

    p = p.rename(columns={"product_name_lenght": "product_name_length",
                          "product_description_lenght": "product_description_length"})

    tr = dict(zip(translation["product_category_name"],
                  translation["product_category_name_english"]))
    tr.update(cfg.CATEGORY_TRANSLATION_PATCH)
    p["product_category"] = (p["product_category_name"].map(tr)
                             .fillna(p["product_category_name"])
                             .fillna(cfg.UNKNOWN_CATEGORY))

    p["has_category"] = p["product_category_name"].notna()

    dims = ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]
    for col in dims:
        med = p.groupby("product_category")[col].transform("median")
        p[col] = p[col].fillna(med).fillna(p[col].median())

    for col in ["product_name_length", "product_description_length", "product_photos_qty"]:
        p[col] = p[col].fillna(0)

    p["product_volume_cm3"] = (p.product_length_cm * p.product_height_cm * p.product_width_cm)
    p["product_density"] = p.product_weight_g / p.product_volume_cm3.replace(0, np.nan)

    log.info("products cleaned — %s rows, %s categories",
             f"{len(p):,}", p.product_category.nunique())
    return p


def clean_orders(orders: pd.DataFrame, q: Quarantine) -> pd.DataFrame:
    o = orders.copy()

    for col in cfg.ORDER_DATE_COLS:
        o[col] = pd.to_datetime(o[col], errors="coerce")

    bad_status = o[~o.order_status.isin(cfg.VALID_ORDER_STATUS)]
    q.reject("orders", bad_status, "order_status not in the known set")
    o = o[o.order_status.isin(cfg.VALID_ORDER_STATUS)]

    impossible = o[o.order_delivered_customer_date < o.order_purchase_timestamp]
    q.reject("orders", impossible, "delivered before it was purchased")
    o = o.drop(index=impossible.index)

    dupes = o[o.duplicated("order_id", keep="first")]
    q.reject("orders", dupes, "duplicate order_id")
    o = o.drop_duplicates("order_id", keep="first")

    o["delivery_days"] = (o.order_delivered_customer_date
                          - o.order_purchase_timestamp).dt.days
    o["delay_days"] = (o.order_delivered_customer_date
                       - o.order_estimated_delivery_date).dt.days
    o["is_late"] = (o.delay_days > 0).astype("float").where(o.delay_days.notna())
    o["delivery_hours"] = (o.order_delivered_customer_date
                           - o.order_purchase_timestamp).dt.total_seconds() / 3600
    o["approval_hours"] = (o.order_approved_at
                           - o.order_purchase_timestamp).dt.total_seconds() / 3600
    o["carrier_handover_days"] = (o.order_delivered_carrier_date
                                  - o.order_approved_at).dt.days

    o["estimated_delivery_days"] = (o.order_estimated_delivery_date
                                    - o.order_purchase_timestamp).dt.days
    o["purchase_year"] = o.order_purchase_timestamp.dt.year
    o["purchase_month"] = o.order_purchase_timestamp.dt.month
    o["purchase_day_of_week"] = o.order_purchase_timestamp.dt.dayofweek
    o["purchase_hour"] = o.order_purchase_timestamp.dt.hour
    o["is_weekend_purchase"] = o.purchase_day_of_week.isin([5, 6]).astype(int)
    o["purchase_date"] = o.order_purchase_timestamp.dt.date

    o["is_delivered"] = (o.order_status == cfg.REVENUE_STATUS).astype(int)

    log.info("orders cleaned — %s rows (%s delivered)",
             f"{len(o):,}", f"{int(o.is_delivered.sum()):,}")
    return o


def clean_order_items(items: pd.DataFrame, orders: pd.DataFrame,
                      sellers: pd.DataFrame, customers: pd.DataFrame,
                      q: Quarantine) -> pd.DataFrame:
    it = items.copy()
    it["shipping_limit_date"] = pd.to_datetime(it["shipping_limit_date"], errors="coerce")
    it["price"] = pd.to_numeric(it["price"], errors="coerce")
    it["freight_value"] = pd.to_numeric(it["freight_value"], errors="coerce")

    bad_money = it[(it.price < 0) | (it.freight_value < 0) | it.price.isna()]
    q.reject("order_items", bad_money, "negative or unparseable price/freight")
    it = it.drop(index=bad_money.index)

    dupes = it[it.duplicated(["order_id", "order_item_id"], keep="first")]
    q.reject("order_items", dupes, "duplicate (order_id, order_item_id)")
    it = it.drop_duplicates(["order_id", "order_item_id"], keep="first")

    orphans = it[~it.order_id.isin(orders.order_id)]
    q.reject("order_items", orphans, "order_id not present in orders")
    it = it.drop(index=orphans.index)

    it["item_revenue"] = it.price + it.freight_value
    it["freight_ratio"] = it.freight_value / it.price.replace(0, np.nan)

    geo_cols = it.merge(orders[["order_id", "customer_id"]], on="order_id", how="left") \
                 .merge(customers[["customer_id", "customer_lat", "customer_lng"]],
                        on="customer_id", how="left") \
                 .merge(sellers[["seller_id", "seller_lat", "seller_lng"]],
                        on="seller_id", how="left")
    if len(geo_cols) != len(it):
        raise ValueError(
            f"coordinate join changed row count ({len(it):,} -> {len(geo_cols):,}); "
            "a join key is not unique")

    it["seller_customer_distance_km"] = np.round(
        haversine_km(geo_cols.seller_lat.values, geo_cols.seller_lng.values,
                     geo_cols.customer_lat.values, geo_cols.customer_lng.values), 2)

    log.info("order_items cleaned — %s rows across %s orders",
             f"{len(it):,}", f"{it.order_id.nunique():,}")
    return it


def aggregate_payments(payments: pd.DataFrame, q: Quarantine) -> pd.DataFrame:
    p = payments.copy()
    p["payment_value"] = pd.to_numeric(p["payment_value"], errors="coerce")
    p["payment_installments"] = pd.to_numeric(p["payment_installments"],
                                              errors="coerce").fillna(0).astype(int)

    bad = p[(p.payment_value < 0) | p.payment_value.isna()]
    q.reject("payments", bad, "negative or unparseable payment_value")
    p = p.drop(index=bad.index)

    dominant = (p.sort_values("payment_value", ascending=False)
                 .drop_duplicates("order_id")
                 .set_index("order_id")["payment_type"])

    agg = (p.groupby("order_id")
             .agg(payment_total=("payment_value", "sum"),
                  payment_installments=("payment_installments", "max"),
                  payment_methods_used=("payment_type", "nunique"),
                  payment_slices=("payment_sequential", "count")))
    agg["payment_type"] = agg.index.map(dominant)
    agg["is_installment_plan"] = (agg.payment_installments > 1).astype(int)

    out = agg.reset_index()
    log.info("payments aggregated %s slices -> %s orders", f"{len(p):,}", f"{len(out):,}")
    return out


def clean_reviews(reviews: pd.DataFrame, orders: pd.DataFrame, q: Quarantine) -> pd.DataFrame:
    r = reviews.copy()
    r["review_creation_date"] = pd.to_datetime(r["review_creation_date"], errors="coerce")
    r["review_answer_timestamp"] = pd.to_datetime(r["review_answer_timestamp"], errors="coerce")
    r["review_score"] = pd.to_numeric(r["review_score"], errors="coerce").astype("Int64")

    lo, hi = cfg.REVIEW_SCORE_RANGE
    bad = r[~r.review_score.between(lo, hi) | r.review_score.isna()]
    q.reject("reviews", bad, f"review_score outside {lo}-{hi}")
    r = r.drop(index=bad.index)
    r["review_score"] = r["review_score"].astype(int)

    orphans = r[~r.order_id.isin(orders.order_id)]
    q.reject("reviews", orphans, "order_id not present in orders")
    r = r.drop(index=orphans.index)

    before = len(r)
    r = (r.sort_values("review_answer_timestamp")
          .drop_duplicates("order_id", keep="last"))
    log.info("reviews deduplicated %s -> %s rows (one per order)",
             f"{before:,}", f"{len(r):,}")

    r["has_review_text"] = r.review_comment_message.notna().astype(int)
    r["review_text_length"] = r.review_comment_message.fillna("").str.len()
    r["is_negative_review"] = (r.review_score <= 2).astype(int)
    r["is_positive_review"] = (r.review_score >= 4).astype(int)
    r["response_days"] = (r.review_answer_timestamp
                          - r.review_creation_date).dt.total_seconds() / 86400
    return r


def build_customer_rfm(orders: pd.DataFrame, items: pd.DataFrame,
                       customers: pd.DataFrame, reviews: pd.DataFrame,
                       products: pd.DataFrame) -> pd.DataFrame:
    snapshot = pd.Timestamp(cfg.SNAPSHOT_DATE)

    delivered = orders[orders.order_status == cfg.REVENUE_STATUS]
    base = (items.merge(delivered[["order_id", "customer_id", "order_purchase_timestamp",
                                   "delivery_days", "is_late"]],
                        on="order_id", how="inner")
                 .merge(customers[["customer_id", cfg.PERSON_KEY, "customer_state",
                                   "customer_city"]], on="customer_id", how="left")
                 .merge(products[["product_id", "product_category"]],
                        on="product_id", how="left"))

    rfm = (base.groupby(cfg.PERSON_KEY)
               .agg(recency_days=("order_purchase_timestamp",
                                  lambda s: (snapshot - s.max()).days),
                    frequency=("order_id", "nunique"),
                    monetary=("item_revenue", "sum"),
                    total_items=("order_id", "size"),
                    avg_item_price=("price", "mean"),
                    total_freight=("freight_value", "sum"),
                    first_purchase=("order_purchase_timestamp", "min"),
                    last_purchase=("order_purchase_timestamp", "max"),
                    distinct_categories=("product_category", "nunique"),
                    customer_state=("customer_state", "first")))

    per_order = base.drop_duplicates("order_id")
    experience = (per_order.groupby(cfg.PERSON_KEY)
                           .agg(avg_delivery_days=("delivery_days", "mean"),
                                late_orders=("is_late", "sum")))
    rfm = rfm.join(experience)

    rfm["avg_order_value"] = rfm.monetary / rfm.frequency
    rfm["tenure_days"] = (rfm.last_purchase - rfm.first_purchase).dt.days
    rfm["late_order_rate"] = (rfm.late_orders / rfm.frequency).clip(upper=1.0)
    rfm["is_repeat_customer"] = (rfm.frequency > 1).astype(int)

    fav = (base.groupby([cfg.PERSON_KEY, "product_category"]).size()
               .reset_index(name="_n").sort_values("_n", ascending=False)
               .drop_duplicates(cfg.PERSON_KEY).set_index(cfg.PERSON_KEY)["product_category"])
    rfm["favourite_category"] = rfm.index.map(fav)

    rev = (reviews.merge(delivered[["order_id", "customer_id"]], on="order_id")
                  .merge(customers[["customer_id", cfg.PERSON_KEY]], on="customer_id")
                  .groupby(cfg.PERSON_KEY).review_score.mean())
    rfm["avg_review_score"] = rfm.index.map(rev)

    rfm["r_score"] = pd.qcut(rfm.recency_days, 5, labels=[5, 4, 3, 2, 1],
                             duplicates="drop").astype(int)
    rfm["m_score"] = pd.qcut(rfm.monetary.rank(method="first"), 5,
                             labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["f_score"] = np.select(
        [rfm.frequency >= 4, rfm.frequency == 3, rfm.frequency == 2],
        [5, 4, 3], default=1)
    rfm["rfm_score"] = rfm.r_score + rfm.f_score + rfm.m_score

    def _segment(row) -> str:
        r, f, m = row.r_score, row.f_score, row.m_score
        if f >= 3 and m >= 4:            return "Champions"
        if f >= 3:                       return "Loyal"
        if r >= 4 and m >= 4:            return "Recent High-Value"
        if r >= 4:                       return "Recent"
        if r <= 2 and m >= 4:            return "At-Risk High-Value"
        if r <= 2:                       return "Lapsed"
        return "Occasional"

    rfm["segment"] = rfm.apply(_segment, axis=1)
    out = rfm.reset_index()
    log.info("customer RFM built — %s customers, %.2f%% repeat",
             f"{len(out):,}", 100 * out.is_repeat_customer.mean())
    return out


def build_seller_performance(items: pd.DataFrame, orders: pd.DataFrame,
                             sellers: pd.DataFrame, reviews: pd.DataFrame,
                             products: pd.DataFrame) -> pd.DataFrame:
    snapshot = pd.Timestamp(cfg.SNAPSHOT_DATE)

    delivered = orders[orders.order_status == cfg.REVENUE_STATUS]
    base = (items.merge(delivered[["order_id", "order_purchase_timestamp",
                                   "delivery_days", "is_late"]],
                        on="order_id", how="inner")
                 .merge(products[["product_id", "product_category"]],
                        on="product_id", how="left"))

    perf = (base.groupby("seller_id")
                .agg(total_orders=("order_id", "nunique"),
                     total_items=("order_id", "size"),
                     total_revenue=("item_revenue", "sum"),
                     avg_item_price=("price", "mean"),
                     avg_freight=("freight_value", "mean"),
                     distinct_products=("product_id", "nunique"),
                     distinct_categories=("product_category", "nunique"),
                     first_sale=("order_purchase_timestamp", "min"),
                     last_sale=("order_purchase_timestamp", "max")))

    per_seller_order = base.drop_duplicates(["seller_id", "order_id"])
    experience = (per_seller_order.groupby("seller_id")
                                  .agg(avg_delivery_days=("delivery_days", "mean"),
                                       late_orders=("is_late", "sum"),
                                       avg_distance_km=("seller_customer_distance_km", "mean")))
    perf = perf.join(experience)

    perf["avg_order_value"] = perf.total_revenue / perf.total_orders
    perf["late_rate"] = (perf.late_orders / perf.total_orders).clip(upper=1.0)
    perf["active_days"] = (perf.last_sale - perf.first_sale).dt.days
    perf["days_since_last_sale"] = (snapshot - perf.last_sale).dt.days
    perf["orders_per_active_month"] = perf.total_orders / ((perf.active_days / 30).clip(lower=1))

    perf["is_dormant"] = (perf.days_since_last_sale > cfg.CHURN_WINDOW_DAYS).astype(int)

    rev = (reviews.merge(items[["order_id", "seller_id"]].drop_duplicates(), on="order_id")
                  .groupby("seller_id")
                  .agg(avg_review_score=("review_score", "mean"),
                       negative_reviews=("is_negative_review", "sum"),
                       total_reviews=("review_score", "size")))
    perf = perf.join(rev)
    perf["negative_review_rate"] = perf.negative_reviews / perf.total_reviews

    perf = perf.reset_index().merge(
        sellers[["seller_id", "seller_state", "seller_city"]], on="seller_id", how="left")

    log.info("seller performance built — %s sellers, %.1f%% dormant (>%dd)",
             f"{len(perf):,}", 100 * perf.is_dormant.mean(), cfg.CHURN_WINDOW_DAYS)
    return perf


def transform(
    raw: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict, Quarantine]:
    q = Quarantine()

    geolocation = clean_geolocation(raw["geolocation"])
    customers   = clean_customers(raw["customers"], geolocation)
    sellers     = clean_sellers(raw["sellers"], geolocation)
    products    = clean_products(raw["products"], raw["category_translation"])
    orders      = clean_orders(raw["orders"], q)
    order_items = clean_order_items(raw["order_items"], orders, sellers, customers, q)
    payments    = aggregate_payments(raw["payments"], q)
    reviews     = clean_reviews(raw["reviews"], orders, q)

    customer_rfm       = build_customer_rfm(orders, order_items, customers, reviews, products)
    seller_performance = build_seller_performance(order_items, orders, sellers,
                                                  reviews, products)

    tables = {
        "customers": customers,
        "orders": orders,
        "order_items": order_items,
        "payments": payments,
        "reviews": reviews,
        "products": products,
        "sellers": sellers,
        "geolocation": geolocation,
        "customer_rfm": customer_rfm,
        "seller_performance": seller_performance,
    }

    delivered_items = order_items.merge(
        orders.loc[orders.order_status == cfg.REVENUE_STATUS, ["order_id"]], on="order_id")

    stats = {
        "row_counts": {k: len(v) for k, v in tables.items()},
        "quarantined": q.counts,
        "quarantined_total": q.total,
        "features_created": [
            "delivery_days", "delay_days", "is_late", "approval_hours",
            "carrier_handover_days", "estimated_delivery_days", "purchase_hour",
            "purchase_day_of_week", "is_weekend_purchase", "item_revenue",
            "freight_ratio", "seller_customer_distance_km", "product_volume_cm3",
            "product_density", "payment_total", "is_installment_plan",
            "review_text_length", "is_negative_review", "response_days",
            "recency_days", "frequency", "monetary", "rfm_score", "segment",
            "late_rate", "days_since_last_sale", "is_dormant",
        ],
        "revenue_delivered": float(delivered_items.item_revenue.sum()),
        "delivered_items": int(len(delivered_items)),
    }
    return tables, stats, q
