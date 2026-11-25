from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from etl import config as cfg

log = logging.getLogger("etl.load")

SCHEMA_SQL = cfg.ROOT / "database" / "schema.sql"
VIEWS_SQL = cfg.ROOT / "database" / "views.sql"

STATE_REGION = {
    **{s: "Norte" for s in ["AC", "AP", "AM", "PA", "RO", "RR", "TO"]},
    **{s: "Nordeste" for s in ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"]},
    **{s: "Centro-Oeste" for s in ["DF", "GO", "MT", "MS"]},
    **{s: "Sudeste" for s in ["ES", "MG", "RJ", "SP"]},
    **{s: "Sul" for s in ["PR", "RS", "SC"]},
}

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"]


def write_processed(tables: dict[str, pd.DataFrame],
                    out_dir: Path | None = None) -> dict[str, int]:
    out_dir = Path(out_dir) if out_dir else cfg.PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, int] = {}
    for name, df in tables.items():
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8")
        written[name] = len(df)
        log.info("wrote %-22s %8s rows -> %s", name, f"{len(df):,}", path.name)
    return written


def write_quarantine(quarantine, out_dir: Path | None = None) -> int:
    out_dir = Path(out_dir) if out_dir else cfg.QUARANTINE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = quarantine.to_frame()
    path = out_dir / "rejected_rows.csv"
    frame.to_csv(path, index=False, encoding="utf-8")
    if len(frame):
        log.warning("wrote %s rejected row(s) -> %s", f"{len(frame):,}", path)
    else:
        log.info("no rows rejected — empty quarantine file written")
    return len(frame)


def _split_statements(sql: str) -> list[str]:
    sql = re.sub(r"^\s*--.*$", "", sql, flags=re.MULTILINE)
    return [s.strip() for s in sql.split(";") if s.strip()]


def run_sql_file(path: Path, engine=None, use_database: bool = False) -> int:
    if engine is None:
        engine = create_engine(cfg.mysql_url(include_db=use_database), future=True)

    statements = _split_statements(Path(path).read_text(encoding="utf-8"))
    with engine.begin() as conn:
        for stmt in statements:
            conn.exec_driver_sql(stmt)
    log.info("executed %s (%d statements)", Path(path).name, len(statements))
    return len(statements)


def _to_sql(df: pd.DataFrame, table: str, engine, chunksize: int = 5_000) -> int:
    clean = df.replace({np.nan: None, pd.NaT: None})
    clean.to_sql(table, engine, if_exists="append", index=False,
                 chunksize=chunksize, method="multi")
    log.info("loaded %-24s %8s rows", table, f"{len(df):,}")
    return len(df)


def _date_key(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s, errors="coerce")
    return (d.dt.year * 10_000 + d.dt.month * 100 + d.dt.day).astype("Int64")


def build_dim_date(start: str = "2016-01-01", end: str = "2018-12-31") -> pd.DataFrame:
    d = pd.DataFrame({"full_date": pd.date_range(start, end, freq="D")})
    d["date_key"] = (d.full_date.dt.year * 10_000
                     + d.full_date.dt.month * 100 + d.full_date.dt.day)
    d["year"] = d.full_date.dt.year
    d["quarter"] = d.full_date.dt.quarter
    d["month"] = d.full_date.dt.month
    d["month_name"] = d.month.map(lambda m: MONTHS[m - 1])
    d["calendar_month"] = d.full_date.dt.strftime("%Y-%m")
    d["day_of_month"] = d.full_date.dt.day
    d["day_of_week"] = d.full_date.dt.dayofweek
    d["day_name"] = d.day_of_week.map(lambda x: DAYS[x])
    d["week_of_year"] = d.full_date.dt.isocalendar().week.astype(int)
    d["is_weekend"] = d.day_of_week.isin([5, 6]).astype(int)
    d["full_date"] = d.full_date.dt.date

    return d[["date_key", "full_date", "year", "quarter", "month", "month_name",
              "calendar_month", "day_of_month", "day_of_week", "day_name",
              "week_of_year", "is_weekend"]]


def build_dim_geography(geolocation: pd.DataFrame) -> pd.DataFrame:
    g = geolocation.rename(columns={
        "geo_city": "city", "geo_state": "state",
        "geo_lat": "latitude", "geo_lng": "longitude"}).copy()
    g["region"] = g.state.map(STATE_REGION)
    g = g.sort_values("zip_code_prefix").reset_index(drop=True)
    g["geography_key"] = np.arange(1, len(g) + 1)
    return g[["geography_key", "zip_code_prefix", "city", "state",
              "region", "latitude", "longitude"]]


def build_dim_customer(customers: pd.DataFrame, geo: pd.DataFrame) -> pd.DataFrame:
    c = customers.sort_values("customer_id").drop_duplicates(cfg.PERSON_KEY, keep="first")

    zip_to_key = geo.set_index("zip_code_prefix").geography_key
    out = pd.DataFrame({
        "customer_unique_id": c[cfg.PERSON_KEY].values,
        "geography_key": c.customer_zip_code_prefix.map(zip_to_key).values,
        "city": c.customer_city.values,
        "state": c.customer_state.values,
    })
    out["region"] = out.state.map(STATE_REGION)
    out = out.sort_values("customer_unique_id").reset_index(drop=True)
    out.insert(0, "customer_key", np.arange(1, len(out) + 1))
    return out


def build_dim_product(products: pd.DataFrame) -> pd.DataFrame:
    p = products.sort_values("product_id").reset_index(drop=True)
    out = pd.DataFrame({
        "product_key": np.arange(1, len(p) + 1),
        "product_id": p.product_id,
        "product_category": p.product_category.fillna(cfg.UNKNOWN_CATEGORY),
        "product_category_pt": p.product_category_name,
        "weight_g": p.product_weight_g,
        "length_cm": p.product_length_cm,
        "height_cm": p.product_height_cm,
        "width_cm": p.product_width_cm,
        "volume_cm3": p.product_volume_cm3,
        "photos_qty": p.product_photos_qty,
        "name_length": p.product_name_length,
        "description_length": p.product_description_length,
    })
    return out


def build_dim_seller(sellers: pd.DataFrame, geo: pd.DataFrame) -> pd.DataFrame:
    s = sellers.sort_values("seller_id").reset_index(drop=True)
    zip_to_key = geo.set_index("zip_code_prefix").geography_key
    out = pd.DataFrame({
        "seller_key": np.arange(1, len(s) + 1),
        "seller_id": s.seller_id,
        "geography_key": s.seller_zip_code_prefix.map(zip_to_key),
        "city": s.seller_city,
        "state": s.seller_state,
    })
    out["region"] = out.state.map(STATE_REGION)
    return out


def load_to_warehouse(tables: dict[str, pd.DataFrame],
                      recreate: bool = True) -> dict[str, int]:
    if recreate:
        log.info("running %s", SCHEMA_SQL.name)
        run_sql_file(SCHEMA_SQL, use_database=False)

    engine = create_engine(cfg.mysql_url(), future=True)
    loaded: dict[str, int] = {}

    customers   = tables["customers"]
    orders      = tables["orders"]
    order_items = tables["order_items"]
    payments    = tables["payments"]
    reviews     = tables["reviews"]
    products    = tables["products"]
    sellers     = tables["sellers"]
    geolocation = tables["geolocation"]

    dim_date = build_dim_date()
    dim_geo = build_dim_geography(geolocation)
    dim_cust = build_dim_customer(customers, dim_geo)
    dim_prod = build_dim_product(products)
    dim_sell = build_dim_seller(sellers, dim_geo)

    loaded["dim_date"] = _to_sql(dim_date, "dim_date", engine)
    loaded["dim_geography"] = _to_sql(dim_geo, "dim_geography", engine)
    loaded["dim_customer"] = _to_sql(dim_cust, "dim_customer", engine)
    loaded["dim_product"] = _to_sql(dim_prod, "dim_product", engine)
    loaded["dim_seller"] = _to_sql(dim_sell, "dim_seller", engine)

    person_key = dim_cust.set_index("customer_unique_id").customer_key
    cust_id_to_key = (customers.set_index("customer_id")[cfg.PERSON_KEY]
                               .map(person_key))
    product_key = dim_prod.set_index("product_id").product_key
    seller_key = dim_sell.set_index("seller_id").seller_key

    item_rollup = order_items.groupby("order_id").agg(
        item_count=("order_item_id", "size"), order_revenue=("item_revenue", "sum"))

    fo = pd.DataFrame({
        "order_id": orders.order_id.values,
        "customer_key": orders.customer_id.map(cust_id_to_key).values,
        "customer_id": orders.customer_id.values,
        "purchase_date_key": _date_key(orders.order_purchase_timestamp).values,
        "approved_date_key": _date_key(orders.order_approved_at).values,
        "delivered_date_key": _date_key(orders.order_delivered_customer_date).values,
        "estimated_date_key": _date_key(orders.order_estimated_delivery_date).values,
        "order_status": orders.order_status.values,
        "is_delivered": orders.is_delivered.values,
        "estimated_delivery_days": orders.estimated_delivery_days.values,
        "purchase_hour": orders.purchase_hour.values,
        "is_weekend_purchase": orders.is_weekend_purchase.values,
        "delivery_days": orders.delivery_days.values,
        "delay_days": orders.delay_days.values,
        "is_late": orders.is_late.values,
        "approval_hours": orders.approval_hours.values,
        "carrier_handover_days": orders.carrier_handover_days.values,
    })
    fo["item_count"] = fo.order_id.map(item_rollup.item_count).fillna(0).astype(int)
    fo["order_revenue"] = fo.order_id.map(item_rollup.order_revenue).fillna(0.0)
    fo = fo.sort_values("order_id").reset_index(drop=True)
    fo.insert(0, "order_key", np.arange(1, len(fo) + 1))
    loaded["fact_orders"] = _to_sql(fo, "fact_orders", engine)

    order_key = fo.set_index("order_id").order_key
    order_cust_key = fo.set_index("order_id").customer_key

    foi = pd.DataFrame({
        "order_id": order_items.order_id.values,
        "order_item_id": order_items.order_item_id.values,
        "order_key": order_items.order_id.map(order_key).values,
        "customer_key": order_items.order_id.map(order_cust_key).values,
        "product_key": order_items.product_id.map(product_key).values,
        "seller_key": order_items.seller_id.map(seller_key).values,
        "purchase_date_key": order_items.order_id.map(
            fo.set_index("order_id").purchase_date_key).values,
        "order_status": order_items.order_id.map(
            fo.set_index("order_id").order_status).values,
        "is_delivered": order_items.order_id.map(
            fo.set_index("order_id").is_delivered).values,
        "review_score": order_items.order_id.map(
            reviews.set_index("order_id").review_score).values,
        "is_negative_review": order_items.order_id.map(
            reviews.set_index("order_id").is_negative_review).values,
        "price": order_items.price.values,
        "freight_value": order_items.freight_value.values,
        "item_revenue": order_items.item_revenue.values,
        "freight_ratio": order_items.freight_ratio.values,
        "seller_customer_distance_km": order_items.seller_customer_distance_km.values,
        "shipping_limit_date": order_items.shipping_limit_date.values,
    })
    foi.insert(0, "order_item_key", np.arange(1, len(foi) + 1))
    loaded["fact_order_items"] = _to_sql(foi, "fact_order_items", engine)

    fp = payments.copy()
    fp["order_key"] = fp.order_id.map(order_key)
    fp = fp.dropna(subset=["order_key"])
    fp.insert(0, "payment_key", np.arange(1, len(fp) + 1))
    fp = fp[["payment_key", "order_key", "order_id", "payment_type",
             "payment_installments", "payment_total", "payment_slices",
             "payment_methods_used", "is_installment_plan"]]
    loaded["fact_payments"] = _to_sql(fp, "fact_payments", engine)

    fr = pd.DataFrame({
        "review_id": reviews.review_id.values,
        "order_key": reviews.order_id.map(order_key).values,
        "order_id": reviews.order_id.values,
        "customer_key": reviews.order_id.map(order_cust_key).values,
        "review_creation_date_key": _date_key(reviews.review_creation_date).values,
        "review_score": reviews.review_score.values,
        "is_negative_review": reviews.is_negative_review.values,
        "is_positive_review": reviews.is_positive_review.values,
        "has_review_text": reviews.has_review_text.values,
        "review_text_length": reviews.review_text_length.values,
        "response_days": reviews.response_days.values,
        "review_comment_title": reviews.review_comment_title.values,
        "review_comment_message": reviews.review_comment_message.values,
    })
    fr.insert(0, "review_key", np.arange(1, len(fr) + 1))
    loaded["fact_reviews"] = _to_sql(fr, "fact_reviews", engine)

    rfm = tables["customer_rfm"].copy()
    rfm["customer_key"] = rfm[cfg.PERSON_KEY].map(person_key)
    rfm = rfm.dropna(subset=["customer_key"])
    rfm_cols = ["customer_key", "customer_unique_id", "recency_days", "frequency",
                "monetary", "avg_order_value", "total_items", "avg_item_price",
                "total_freight", "tenure_days", "first_purchase", "last_purchase",
                "avg_delivery_days", "late_orders", "late_order_rate",
                "avg_review_score", "distinct_categories", "favourite_category",
                "is_repeat_customer", "r_score", "f_score", "m_score",
                "rfm_score", "segment"]
    loaded["agg_customer_rfm"] = _to_sql(rfm[rfm_cols], "agg_customer_rfm", engine)

    perf = tables["seller_performance"].copy()
    perf["seller_key"] = perf.seller_id.map(seller_key)
    perf = perf.dropna(subset=["seller_key"])
    perf_cols = ["seller_key", "seller_id", "total_orders", "total_items",
                 "total_revenue", "avg_order_value", "avg_item_price", "avg_freight",
                 "avg_delivery_days", "late_orders", "late_rate", "avg_distance_km",
                 "distinct_products", "distinct_categories", "first_sale", "last_sale",
                 "active_days", "days_since_last_sale", "orders_per_active_month",
                 "avg_review_score", "negative_reviews", "total_reviews",
                 "negative_review_rate", "is_dormant"]
    loaded["agg_seller_performance"] = _to_sql(perf[perf_cols],
                                               "agg_seller_performance", engine)

    if VIEWS_SQL.exists():
        run_sql_file(VIEWS_SQL, engine=engine)

    engine.dispose()
    return loaded


def verify_warehouse() -> pd.DataFrame:
    engine = create_engine(cfg.mysql_url(), future=True)
    b = cfg.BASELINE
    rows = []

    with engine.connect() as conn:
        def scalar(sql: str):
            return conn.execute(text(sql)).scalar()

        checks = [
            ("dim_customer rows (people)", "SELECT COUNT(*) FROM dim_customer",
             b["unique_customers"]),
            ("dim_seller rows", "SELECT COUNT(*) FROM dim_seller", b["sellers"]),
            ("dim_geography rows", "SELECT COUNT(*) FROM dim_geography",
             b["geo_zip_prefixes"]),
            ("fact_orders rows", "SELECT COUNT(*) FROM fact_orders", b["total_orders"]),
            ("fact_order_items rows", "SELECT COUNT(*) FROM fact_order_items", 112_650),
            ("delivered orders",
             "SELECT COUNT(*) FROM fact_orders WHERE is_delivered = 1",
             b["delivered_orders"]),
            ("late orders",
             "SELECT COUNT(*) FROM fact_orders WHERE is_late = 1 AND is_delivered = 1",
             b["late_orders"]),
            ("REVENUE (delivered)",
             """SELECT ROUND(SUM(i.item_revenue), 2) FROM fact_order_items i
                JOIN fact_orders o ON o.order_key = i.order_key
                WHERE o.is_delivered = 1""", b["total_revenue"]),
            ("orphan facts (order_items -> orders)",
             """SELECT COUNT(*) FROM fact_order_items i
                LEFT JOIN fact_orders o ON o.order_key = i.order_key
                WHERE o.order_key IS NULL""", 0),
            ("orphan facts (order_items -> customer)",
             """SELECT COUNT(*) FROM fact_order_items i
                LEFT JOIN dim_customer c ON c.customer_key = i.customer_key
                WHERE c.customer_key IS NULL""", 0),
        ]

        for name, sql, expected in checks:
            actual = scalar(sql)
            actual = float(actual) if actual is not None else None
            ok = (actual is not None
                  and abs(actual - float(expected)) <= cfg.REVENUE_TOLERANCE)
            rows.append({"check": name, "expected": expected,
                         "actual": actual, "result": "PASS" if ok else "FAIL"})

    engine.dispose()
    return pd.DataFrame(rows)
