from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from etl import config as cfg

log = logging.getLogger("etl.validate")

CRITICAL, ERROR, WARNING = "CRITICAL", "ERROR", "WARNING"


class ValidationFailure(RuntimeError):
    pass


@dataclass
class Check:
    name: str
    passed: bool
    severity: str
    detail: str = ""

    @property
    def symbol(self) -> str:
        return "PASS" if self.passed else self.severity


@dataclass
class ValidationReport:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, passed: bool, severity: str = ERROR, detail: str = "") -> None:
        self.checks.append(Check(name, passed, severity, detail))
        level = logging.INFO if passed else (
            logging.CRITICAL if severity == CRITICAL else logging.WARNING)
        log.log(level, "[%-8s] %-46s %s",
                "PASS" if passed else severity, name, detail)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    @property
    def critical_failures(self) -> list[Check]:
        return [c for c in self.failures if c.severity == CRITICAL]

    @property
    def passed(self) -> bool:
        return not self.critical_failures

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"check": c.name, "result": c.symbol, "detail": c.detail}
                             for c in self.checks])

    def raise_if_failed(self) -> None:
        if self.critical_failures:
            lines = "\n".join(f"  - {c.name}: {c.detail}" for c in self.critical_failures)
            raise ValidationFailure(
                f"{len(self.critical_failures)} critical check(s) failed:\n{lines}")


def validate(tables: dict[str, pd.DataFrame]) -> ValidationReport:
    rep = ValidationReport()

    customers   = tables["customers"]
    orders      = tables["orders"]
    order_items = tables["order_items"]
    payments    = tables["payments"]
    reviews     = tables["reviews"]
    products    = tables["products"]
    sellers     = tables["sellers"]
    geolocation = tables["geolocation"]
    rfm         = tables["customer_rfm"]

    for name, df, keys in [
        ("customers",   customers,   ["customer_id"]),
        ("orders",      orders,      ["order_id"]),
        ("order_items", order_items, ["order_id", "order_item_id"]),
        ("payments",    payments,    ["order_id"]),
        ("reviews",     reviews,     ["order_id"]),
        ("products",    products,    ["product_id"]),
        ("sellers",     sellers,     ["seller_id"]),
        ("geolocation", geolocation, ["zip_code_prefix"]),
    ]:
        n = int(df.duplicated(keys).sum())
        rep.add(f"PK unique: {name} ({'+'.join(keys)})", n == 0, CRITICAL,
                "unique" if n == 0 else f"{n:,} duplicate keys")

    for label, left, lcol, right, rcol in [
        ("orders -> customers",     orders,      "customer_id", customers,   "customer_id"),
        ("order_items -> orders",   order_items, "order_id",    orders,      "order_id"),
        ("order_items -> products", order_items, "product_id",  products,    "product_id"),
        ("order_items -> sellers",  order_items, "seller_id",   sellers,     "seller_id"),
        ("payments -> orders",      payments,    "order_id",    orders,      "order_id"),
        ("reviews -> orders",       reviews,     "order_id",    orders,      "order_id"),
    ]:
        n = int((~left[lcol].isin(right[rcol])).sum())
        rep.add(f"FK intact: {label}", n == 0, CRITICAL,
                "0 orphans" if n == 0 else f"{n:,} orphan rows")

    rep.add("no negative prices", int((order_items.price < 0).sum()) == 0, CRITICAL)
    rep.add("no negative freight", int((order_items.freight_value < 0).sum()) == 0, CRITICAL)
    rep.add("no negative payments", int((payments.payment_total < 0).sum()) == 0, CRITICAL)

    lo, hi = cfg.REVIEW_SCORE_RANGE
    bad_scores = int((~reviews.review_score.between(lo, hi)).sum())
    rep.add(f"review scores within {lo}-{hi}", bad_scores == 0, CRITICAL,
            "" if bad_scores == 0 else f"{bad_scores:,} out of range")

    neg_delivery = int((orders.delivery_days < 0).sum())
    rep.add("no negative delivery times", neg_delivery == 0, CRITICAL,
            "" if neg_delivery == 0 else f"{neg_delivery:,} rows")

    start, end = pd.Timestamp(cfg.DATA_START_DATE), pd.Timestamp(cfg.SNAPSHOT_DATE)
    ts = orders.order_purchase_timestamp
    out_of_range = int(((ts < start) | (ts > end + pd.Timedelta(days=1))).sum())
    rep.add(f"purchase dates within {cfg.DATA_START_DATE}..{cfg.SNAPSHOT_DATE}",
            out_of_range == 0, ERROR,
            "" if out_of_range == 0 else f"{out_of_range:,} outside range")

    n_null_purchase = int(ts.isna().sum())
    rep.add("no null purchase timestamps", n_null_purchase == 0, CRITICAL,
            "" if n_null_purchase == 0 else f"{n_null_purchase:,} nulls")

    delivered_no_date = int(((orders.order_status == cfg.REVENUE_STATUS)
                             & orders.order_delivered_customer_date.isna()).sum())
    rep.add("delivered orders carry a delivery date", delivered_no_date <= 10, WARNING,
            f"{delivered_no_date} delivered rows lack a delivery date")

    geo_ok = len(geolocation) == geolocation.zip_code_prefix.nunique()
    rep.add("geolocation is one row per zip prefix", geo_ok, CRITICAL,
            f"{len(geolocation):,} rows / {geolocation.zip_code_prefix.nunique():,} prefixes")

    pay_ok = len(payments) == payments.order_id.nunique()
    rep.add("payments aggregated to one row per order", pay_ok, CRITICAL,
            f"{len(payments):,} rows")

    rev_ok = len(reviews) == reviews.order_id.nunique()
    rep.add("reviews deduplicated to one row per order", rev_ok, CRITICAL,
            f"{len(reviews):,} rows")

    null_cat = int(products.product_category.isna().sum())
    rep.add("every product resolves to a category", null_cat == 0, ERROR,
            "" if null_cat == 0 else f"{null_cat:,} null categories")

    b = cfg.BASELINE

    n_people = int(customers[cfg.PERSON_KEY].nunique())
    rep.add("unique customers == baseline", n_people == b["unique_customers"], ERROR,
            f"{n_people:,} (expected {b['unique_customers']:,})")

    n_delivered = int((orders.order_status == cfg.REVENUE_STATUS).sum())
    rep.add("delivered orders == baseline", n_delivered == b["delivered_orders"], ERROR,
            f"{n_delivered:,} (expected {b['delivered_orders']:,})")

    delivered_ids = orders.loc[orders.order_status == cfg.REVENUE_STATUS, "order_id"]
    di = order_items[order_items.order_id.isin(delivered_ids)]

    revenue = float(di.item_revenue.sum())
    delta = abs(revenue - b["total_revenue"])
    rep.add("REVENUE RECONCILES to the cent", delta <= cfg.REVENUE_TOLERANCE, CRITICAL,
            f"R$ {revenue:,.2f} vs baseline R$ {b['total_revenue']:,.2f} (delta R$ {delta:,.2f})")

    price_delta = abs(float(di.price.sum()) - b["price_sum"])
    rep.add("price component reconciles", price_delta <= cfg.REVENUE_TOLERANCE, CRITICAL,
            f"delta R$ {price_delta:,.2f}")

    freight_delta = abs(float(di.freight_value.sum()) - b["freight_sum"])
    rep.add("freight component reconciles", freight_delta <= cfg.REVENUE_TOLERANCE, CRITICAL,
            f"delta R$ {freight_delta:,.2f}")

    rep.add("delivered line items == baseline", len(di) == b["delivered_items"], ERROR,
            f"{len(di):,} (expected {b['delivered_items']:,})")

    rfm_ok = int(rfm[cfg.PERSON_KEY].duplicated().sum()) == 0
    rep.add("customer_rfm is one row per person", rfm_ok, CRITICAL)

    rfm_money = abs(float(rfm.monetary.sum()) - revenue)
    rep.add("RFM monetary ties back to revenue", rfm_money <= 1.0, ERROR,
            f"delta R$ {rfm_money:,.2f}")

    bad_late = int(((orders.is_late.notna()) & (~orders.is_late.isin([0, 1]))).sum())
    rep.add("is_late is strictly binary", bad_late == 0, ERROR)

    delivered = orders[orders.order_status == cfg.REVENUE_STATUS]
    n_late = int(delivered.is_late.fillna(0).sum())
    rep.add("late order count == baseline", n_late == b["late_orders"], ERROR,
            f"{n_late:,} late ({100 * n_late / len(delivered):.2f}%), "
            f"expected {b['late_orders']:,}")

    avg_dd = float(delivered.delivery_days.mean())
    rep.add("avg delivery days == baseline", abs(avg_dd - b["avg_delivery_days"]) < 0.05,
            ERROR, f"{avg_dd:.2f} days (expected {b['avg_delivery_days']:.2f})")

    perf = tables["seller_performance"]
    n_bad = int((rfm.late_order_rate > 1).sum())
    rep.add("customer late_order_rate within 0-1", n_bad == 0, ERROR,
            "" if n_bad == 0 else f"{n_bad:,} customers above 1.0")

    n_bad = int((perf.late_rate > 1).sum())
    rep.add("seller late_rate within 0-1", n_bad == 0, ERROR,
            "" if n_bad == 0 else f"{n_bad:,} sellers above 1.0")

    n_bad = int((perf.late_orders > perf.total_orders).sum())
    rep.add("seller late_orders <= total_orders", n_bad == 0, ERROR,
            "" if n_bad == 0 else f"{n_bad:,} sellers exceed their own order count")

    n_bad = int((rfm.late_orders > rfm.frequency).sum())
    rep.add("customer late_orders <= orders placed", n_bad == 0, ERROR,
            "" if n_bad == 0 else f"{n_bad:,} customers exceed their own order count")

    return rep


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from etl.extract import extract
    from etl.transform import transform

    raw, _ = extract()
    tables, _, _ = transform(raw)
    report = validate(tables)
    print(f"\n{len(report.checks)} checks, {len(report.failures)} failed")
    report.raise_if_failed()
