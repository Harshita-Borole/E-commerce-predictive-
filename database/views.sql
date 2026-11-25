USE ecommerce_dw;

CREATE OR REPLACE VIEW vw_delivered_items AS
SELECT
    i.order_item_key,
    i.order_key,
    i.order_id,
    i.customer_key,
    i.product_key,
    i.seller_key,
    i.purchase_date_key,
    i.price,
    i.freight_value,
    i.item_revenue,
    i.seller_customer_distance_km,
    o.delivery_days,
    o.delay_days,
    o.is_late
FROM fact_order_items i
JOIN fact_orders o ON o.order_key = i.order_key
WHERE o.is_delivered = 1;

CREATE OR REPLACE VIEW vw_order_detail AS
SELECT
    o.order_key,
    o.order_id,
    o.order_status,
    o.is_delivered,
    d.full_date            AS purchase_date,
    d.calendar_month,
    d.year,
    d.quarter,
    d.month_name,
    d.day_name,
    d.is_weekend,
    o.purchase_hour,
    o.item_count,
    o.order_revenue,
    o.delivery_days,
    o.estimated_delivery_days,
    o.delay_days,
    o.is_late,
    o.approval_hours,
    c.customer_key,
    c.customer_unique_id,
    c.city                 AS customer_city,
    c.state                AS customer_state,
    c.region               AS customer_region,
    p.payment_type,
    p.payment_installments,
    p.payment_total,
    r.review_score,
    r.is_negative_review,
    r.has_review_text
FROM fact_orders o
JOIN dim_customer  c ON c.customer_key = o.customer_key
JOIN dim_date      d ON d.date_key     = o.purchase_date_key
LEFT JOIN fact_payments p ON p.order_key = o.order_key
LEFT JOIN fact_reviews  r ON r.order_key = o.order_key;

CREATE OR REPLACE VIEW vw_daily_revenue AS
SELECT
    d.full_date                        AS revenue_date,
    d.calendar_month,
    d.day_name,
    d.is_weekend,
    COUNT(DISTINCT v.order_key)        AS orders,
    COUNT(*)                           AS items_sold,
    ROUND(SUM(v.item_revenue), 2)      AS revenue,
    ROUND(SUM(v.price), 2)             AS product_revenue,
    ROUND(SUM(v.freight_value), 2)     AS freight_revenue,
    ROUND(AVG(v.item_revenue), 2)      AS avg_item_value
FROM vw_delivered_items v
JOIN dim_date d ON d.date_key = v.purchase_date_key
GROUP BY d.full_date, d.calendar_month, d.day_name, d.is_weekend;

CREATE OR REPLACE VIEW vw_monthly_revenue AS
SELECT
    d.calendar_month,
    d.year,
    d.month,
    d.month_name,
    COUNT(DISTINCT v.order_key)                    AS orders,
    COUNT(DISTINCT v.customer_key)                 AS customers,
    ROUND(SUM(v.item_revenue), 2)                  AS revenue,
    ROUND(SUM(v.item_revenue)
          / COUNT(DISTINCT v.order_key), 2)        AS avg_order_value
FROM vw_delivered_items v
JOIN dim_date d ON d.date_key = v.purchase_date_key
GROUP BY d.calendar_month, d.year, d.month, d.month_name;

CREATE OR REPLACE VIEW vw_weekly_revenue AS
SELECT
    YEARWEEK(d.full_date, 3)                       AS year_week,
    MIN(d.full_date)                               AS week_start,
    COUNT(DISTINCT v.order_key)                    AS orders,
    ROUND(SUM(v.item_revenue), 2)                  AS revenue
FROM vw_delivered_items v
JOIN dim_date d ON d.date_key = v.purchase_date_key
GROUP BY YEARWEEK(d.full_date, 3);

CREATE OR REPLACE VIEW vw_customer_360 AS
SELECT
    c.customer_key,
    c.customer_unique_id,
    c.city,
    c.state,
    c.region,
    f.recency_days,
    f.frequency,
    f.monetary,
    f.avg_order_value,
    f.tenure_days,
    f.first_purchase,
    f.last_purchase,
    f.avg_delivery_days,
    f.late_orders,
    f.late_order_rate,
    f.avg_review_score,
    f.favourite_category,
    f.distinct_categories,
    f.is_repeat_customer,
    f.r_score,
    f.f_score,
    f.m_score,
    f.rfm_score,
    f.segment
FROM dim_customer c
JOIN agg_customer_rfm f ON f.customer_key = c.customer_key;

CREATE OR REPLACE VIEW vw_seller_scorecard AS
SELECT
    s.seller_key,
    s.seller_id,
    s.city,
    s.state,
    s.region,
    p.total_orders,
    p.total_items,
    p.total_revenue,
    p.avg_order_value,
    p.distinct_products,
    p.distinct_categories,
    p.avg_delivery_days,
    p.late_orders,
    p.late_rate,
    p.avg_distance_km,
    p.avg_review_score,
    p.negative_reviews,
    p.negative_review_rate,
    p.first_sale,
    p.last_sale,
    p.active_days,
    p.days_since_last_sale,
    p.orders_per_active_month,
    p.is_dormant
FROM dim_seller s
JOIN agg_seller_performance p ON p.seller_key = s.seller_key;

CREATE OR REPLACE VIEW vw_product_performance AS
SELECT
    pr.product_key,
    pr.product_id,
    pr.product_category,
    pr.weight_g,
    pr.volume_cm3,
    pr.photos_qty,
    COUNT(*)                              AS items_sold,
    COUNT(DISTINCT v.order_key)           AS orders,
    ROUND(SUM(v.item_revenue), 2)         AS revenue,
    ROUND(AVG(v.price), 2)                AS avg_price,
    ROUND(AVG(v.freight_value), 2)        AS avg_freight,
    ROUND(AVG(v.delivery_days), 1)        AS avg_delivery_days,
    ROUND(AVG(v.is_late), 4)              AS late_rate
FROM vw_delivered_items v
JOIN dim_product pr ON pr.product_key = v.product_key
GROUP BY pr.product_key, pr.product_id, pr.product_category,
         pr.weight_g, pr.volume_cm3, pr.photos_qty;

CREATE OR REPLACE VIEW vw_category_performance AS
SELECT
    pr.product_category,
    COUNT(DISTINCT pr.product_key)        AS products,
    COUNT(DISTINCT v.seller_key)          AS sellers,
    COUNT(*)                              AS items_sold,
    COUNT(DISTINCT v.order_key)           AS orders,
    ROUND(SUM(v.item_revenue), 2)         AS revenue,
    ROUND(AVG(v.price), 2)                AS avg_price,
    ROUND(AVG(v.delivery_days), 1)        AS avg_delivery_days,
    ROUND(AVG(v.is_late), 4)              AS late_rate
FROM vw_delivered_items v
JOIN dim_product pr ON pr.product_key = v.product_key
GROUP BY pr.product_category;

CREATE OR REPLACE VIEW vw_delivery_by_state AS
SELECT
    c.state,
    c.region,
    COUNT(DISTINCT o.order_key)           AS delivered_orders,
    ROUND(AVG(o.delivery_days), 1)        AS avg_delivery_days,
    ROUND(AVG(o.estimated_delivery_days), 1) AS avg_promised_days,
    ROUND(AVG(o.delay_days), 1)           AS avg_days_vs_estimate,
    ROUND(AVG(o.is_late), 4)              AS late_rate,
    ROUND(AVG(r.review_score), 2)         AS avg_review_score
FROM fact_orders o
JOIN dim_customer c      ON c.customer_key = o.customer_key
LEFT JOIN fact_reviews r ON r.order_key    = o.order_key
WHERE o.is_delivered = 1
GROUP BY c.state, c.region;

CREATE OR REPLACE VIEW vw_delivery_review_impact AS
SELECT
    o.is_late,
    COUNT(*)                              AS orders,
    ROUND(AVG(r.review_score), 3)         AS avg_review_score,
    ROUND(AVG(r.is_negative_review), 4)   AS negative_review_rate,
    ROUND(AVG(o.delivery_days), 1)        AS avg_delivery_days,
    ROUND(AVG(o.delay_days), 1)           AS avg_days_vs_estimate
FROM fact_orders o
JOIN fact_reviews r ON r.order_key = o.order_key
WHERE o.is_delivered = 1 AND o.is_late IS NOT NULL
GROUP BY o.is_late;
