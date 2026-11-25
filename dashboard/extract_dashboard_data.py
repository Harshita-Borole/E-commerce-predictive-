from __future__ import annotations

import json

import pandas as pd
from sqlalchemy import create_engine

from etl import config as cfg

OUT = cfg.ROOT / "dashboard" / "dashboard_data.json"

RELIABLE_FROM, RELIABLE_TO = "2017-01", "2018-08"

QUERIES: dict[str, str] = {

    "kpis": """
        SELECT
            (SELECT ROUND(SUM(item_revenue), 2) FROM vw_delivered_items)          AS total_revenue,
            (SELECT COUNT(*) FROM fact_orders)                                    AS total_orders,
            (SELECT COUNT(*) FROM fact_orders WHERE is_delivered = 1)             AS delivered_orders,
            (SELECT COUNT(*) FROM dim_customer)                                   AS total_customers,
            (SELECT COUNT(*) FROM dim_seller)                                     AS total_sellers,
            (SELECT COUNT(*) FROM dim_product)                                    AS total_products,
            (SELECT ROUND(SUM(item_revenue) / COUNT(DISTINCT order_key), 2)
               FROM vw_delivered_items)                                           AS avg_order_value,
            (SELECT ROUND(AVG(review_score), 2) FROM fact_reviews)                AS avg_review_score,
            (SELECT ROUND(100.0 * (1 - AVG(is_late)), 2)
               FROM fact_orders WHERE is_delivered = 1)                           AS on_time_pct,
            (SELECT ROUND(AVG(delivery_days), 1)
               FROM fact_orders WHERE is_delivered = 1)                           AS avg_delivery_days,
            (SELECT ROUND(100.0 * AVG(CASE WHEN frequency > 1 THEN 1 ELSE 0 END), 2)
               FROM agg_customer_rfm)                                             AS repeat_rate_pct,
            (SELECT ROUND(100.0 * AVG(is_negative_review), 2) FROM fact_reviews)  AS negative_review_pct
    """,

    "monthly_trend": """
        SELECT d.calendar_month, COUNT(DISTINCT v.order_key) AS orders,
               COUNT(DISTINCT v.customer_key) AS customers,
               ROUND(SUM(v.item_revenue), 2) AS revenue
        FROM vw_delivered_items v
        JOIN dim_date d ON d.date_key = v.purchase_date_key
        GROUP BY d.calendar_month ORDER BY d.calendar_month
    """,

    "late_trend": """
        SELECT d.calendar_month, COUNT(*) AS delivered_orders,
               ROUND(100.0 * AVG(o.is_late), 2) AS late_pct
        FROM fact_orders o JOIN dim_date d ON d.date_key = o.purchase_date_key
        WHERE o.is_delivered = 1
        GROUP BY d.calendar_month ORDER BY d.calendar_month
    """,

    "category_revenue": """
        SELECT p.product_category, COUNT(*) AS items_sold,
               ROUND(SUM(v.item_revenue), 2) AS revenue,
               ROUND(AVG(v.price), 2) AS avg_price
        FROM vw_delivered_items v JOIN dim_product p ON p.product_key = v.product_key
        GROUP BY p.product_category ORDER BY revenue DESC LIMIT 10
    """,

    "category_rating": """
        SELECT p.product_category, COUNT(*) AS reviewed_orders,
               ROUND(AVG(r.review_score), 2) AS avg_review_score,
               ROUND(100.0 * AVG(o.is_late), 2) AS late_pct
        FROM fact_reviews r
        JOIN fact_orders o      ON o.order_key = r.order_key
        JOIN fact_order_items i ON i.order_key = o.order_key
        JOIN dim_product p      ON p.product_key = i.product_key
        WHERE o.is_delivered = 1
        GROUP BY p.product_category
        HAVING reviewed_orders >= 300
        ORDER BY avg_review_score DESC
    """,

    "top_products": """
        SELECT LEFT(p.product_id, 10) AS product, p.product_category,
               COUNT(*) AS units_sold, ROUND(SUM(v.item_revenue), 2) AS revenue
        FROM vw_delivered_items v JOIN dim_product p ON p.product_key = v.product_key
        GROUP BY p.product_key, p.product_id, p.product_category
        ORDER BY revenue DESC LIMIT 10
    """,

    "worst_products": """
        SELECT LEFT(p.product_id, 10) AS product, p.product_category,
               COUNT(*) AS units_sold,
               ROUND(AVG(r.review_score), 2) AS avg_review_score,
               ROUND(100.0 * AVG(v.is_late), 1) AS late_pct
        FROM vw_delivered_items v
        JOIN dim_product  p ON p.product_key = v.product_key
        JOIN fact_reviews r ON r.order_key   = v.order_key
        GROUP BY p.product_key, p.product_id, p.product_category
        HAVING units_sold >= 20 AND avg_review_score < 3.2
        ORDER BY avg_review_score ASC LIMIT 10
    """,

    "segments": """
        SELECT segment, COUNT(*) AS customers, ROUND(SUM(monetary), 2) AS revenue,
               ROUND(AVG(monetary), 2) AS avg_value,
               ROUND(AVG(recency_days), 0) AS avg_recency_days,
               ROUND(AVG(avg_review_score), 2) AS avg_review
        FROM agg_customer_rfm GROUP BY segment ORDER BY revenue DESC
    """,

    "frequency_dist": """
        SELECT frequency AS orders_placed, COUNT(*) AS customers,
               ROUND(SUM(monetary), 2) AS revenue
        FROM agg_customer_rfm GROUP BY frequency ORDER BY frequency
    """,

    "rfm_distribution": """
        SELECT r_score, m_score, COUNT(*) AS customers
        FROM agg_customer_rfm GROUP BY r_score, m_score ORDER BY r_score, m_score
    """,

    "state_revenue": """
        SELECT c.state, c.region, COUNT(DISTINCT v.order_key) AS orders,
               ROUND(SUM(v.item_revenue), 2) AS revenue
        FROM vw_delivered_items v JOIN dim_customer c ON c.customer_key = v.customer_key
        GROUP BY c.state, c.region ORDER BY revenue DESC
    """,

    "top_sellers": """
        SELECT LEFT(seller_id, 10) AS seller, state, total_orders, total_revenue,
               ROUND(avg_review_score, 2) AS avg_review,
               ROUND(100.0 * late_rate, 1) AS late_pct,
               ROUND(avg_delivery_days, 1) AS avg_delivery_days, is_dormant
        FROM vw_seller_scorecard ORDER BY total_revenue DESC LIMIT 10
    """,

    "seller_deciles": """
        SELECT revenue_decile, COUNT(*) AS sellers, ROUND(SUM(total_revenue), 2) AS revenue
        FROM (SELECT total_revenue, NTILE(10) OVER (ORDER BY total_revenue DESC) AS revenue_decile
                FROM agg_seller_performance) t
        GROUP BY revenue_decile ORDER BY revenue_decile
    """,

    "seller_risk": """
        SELECT LEFT(seller_id, 10) AS seller, state, total_orders, total_revenue,
               ROUND(avg_review_score, 2) AS avg_review,
               ROUND(100.0 * late_rate, 1) AS late_pct, days_since_last_sale
        FROM vw_seller_scorecard
        WHERE total_orders >= 20 AND (avg_review_score < 3.5 OR late_rate > 0.15)
        ORDER BY total_revenue DESC LIMIT 10
    """,

    "seller_health": """
        SELECT is_dormant, COUNT(*) AS sellers,
               ROUND(AVG(total_revenue), 0) AS avg_revenue,
               ROUND(AVG(avg_review_score), 2) AS avg_review,
               ROUND(100.0 * AVG(late_rate), 1) AS late_pct
        FROM agg_seller_performance GROUP BY is_dormant
    """,

    "delivery_by_state": """
        SELECT state, region, delivered_orders, avg_delivery_days,
               ROUND(100.0 * late_rate, 2) AS late_pct, avg_review_score
        FROM vw_delivery_by_state
        WHERE delivered_orders >= 200
        ORDER BY avg_delivery_days DESC
    """,

    "delivery_by_region": """
        SELECT c.region, COUNT(*) AS delivered_orders,
               ROUND(AVG(o.delivery_days), 1) AS avg_delivery_days,
               ROUND(100.0 * AVG(o.is_late), 2) AS late_pct,
               ROUND(AVG(r.review_score), 2) AS avg_review
        FROM fact_orders o
        JOIN dim_customer c      ON c.customer_key = o.customer_key
        LEFT JOIN fact_reviews r ON r.order_key    = o.order_key
        WHERE o.is_delivered = 1 AND c.region IS NOT NULL
        GROUP BY c.region ORDER BY avg_delivery_days DESC
    """,

    "delay_impact": """
        SELECT CASE
                 WHEN o.delay_days <= -10 THEN '10+ days early'
                 WHEN o.delay_days <   0  THEN 'Early'
                 WHEN o.delay_days =  0   THEN 'On time'
                 WHEN o.delay_days <=  3  THEN '1-3 days late'
                 WHEN o.delay_days <=  7  THEN '4-7 days late'
                 WHEN o.delay_days <= 15  THEN '8-15 days late'
                 ELSE '15+ days late' END AS delivery_outcome,
               COUNT(*) AS orders,
               ROUND(AVG(r.review_score), 2) AS avg_review_score,
               ROUND(100.0 * AVG(r.is_negative_review), 1) AS negative_pct
        FROM fact_orders o JOIN fact_reviews r ON r.order_key = o.order_key
        WHERE o.is_delivered = 1 AND o.delay_days IS NOT NULL
        GROUP BY delivery_outcome
        ORDER BY MIN(o.delay_days)
    """,

    "review_distribution": """
        SELECT review_score, COUNT(*) AS reviews
        FROM fact_reviews GROUP BY review_score ORDER BY review_score
    """,

    "payment_types": """
        SELECT payment_type, COUNT(*) AS orders, ROUND(SUM(payment_total), 2) AS value
        FROM fact_payments WHERE payment_type IS NOT NULL
        GROUP BY payment_type ORDER BY orders DESC
    """,
}


def extract() -> dict:
    engine = create_engine(cfg.mysql_url(), future=True)
    raw = engine.raw_connection()
    data: dict[str, object] = {
        "generated_from": "ecommerce_dw",
        "reliable_window": {"from": RELIABLE_FROM, "to": RELIABLE_TO},
    }

    try:
        for name, sql in QUERIES.items():
            cur = raw.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            df = pd.DataFrame(rows, columns=[c[0] for c in cur.description])
            cur.close()

            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].map(
                        lambda x: float(x) if hasattr(x, "as_integer_ratio") else x)
            df = df.astype(object).where(df.notna(), None)

            records = df.to_dict("records")
            data[name] = records[0] if name == "kpis" else records
            print(f"  {name:22} {len(records):>4} rows")
    finally:
        raw.close()
        engine.dispose()

    return data


def main() -> int:
    print("extracting dashboard data from ecommerce_dw ...")
    data = extract()
    OUT.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
