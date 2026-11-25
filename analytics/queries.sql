USE ecommerce_dw;

-- QUERY 01 | Monthly revenue, growth and running total | CTE, LAG, SUM OVER, CASE
WITH monthly AS (
    SELECT
        d.calendar_month,
        COUNT(DISTINCT v.order_key)          AS orders,
        COUNT(DISTINCT v.customer_key)       AS customers,
        ROUND(SUM(v.item_revenue), 2)        AS revenue
    FROM vw_delivered_items v
    JOIN dim_date d ON d.date_key = v.purchase_date_key
    GROUP BY d.calendar_month
)
SELECT
    calendar_month,
    orders,
    customers,
    revenue,
    ROUND(revenue / NULLIF(orders, 0), 2)                       AS avg_order_value,
    LAG(revenue) OVER (ORDER BY calendar_month)                 AS prev_month_revenue,
    ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY calendar_month))
          / NULLIF(LAG(revenue) OVER (ORDER BY calendar_month), 0), 1) AS mom_growth_pct,
    ROUND(SUM(revenue) OVER (ORDER BY calendar_month), 2)       AS running_total,
    CASE
        WHEN calendar_month < '2017-01' THEN 'pilot - not comparable'
        WHEN calendar_month > '2018-08' THEN 'truncated - partial month'
        ELSE 'reliable'
    END                                                          AS data_quality
FROM monthly
ORDER BY calendar_month;


-- QUERY 02 | Year-over-year growth for matching months | CTE, self-join, CASE
WITH by_year_month AS (
    SELECT
        d.year,
        d.month,
        ROUND(SUM(v.item_revenue), 2) AS revenue
    FROM vw_delivered_items v
    JOIN dim_date d ON d.date_key = v.purchase_date_key
    WHERE d.month BETWEEN 1 AND 8
    GROUP BY d.year, d.month
)
SELECT
    y2018.month,
    y2017.revenue                                    AS revenue_2017,
    y2018.revenue                                    AS revenue_2018,
    ROUND(y2018.revenue - y2017.revenue, 2)          AS absolute_growth,
    ROUND(100.0 * (y2018.revenue - y2017.revenue)
          / NULLIF(y2017.revenue, 0), 1)             AS yoy_growth_pct,
    CASE
        WHEN y2018.revenue > y2017.revenue * 2   THEN 'more than doubled'
        WHEN y2018.revenue > y2017.revenue       THEN 'growth'
        ELSE 'decline'
    END                                              AS trend
FROM by_year_month y2018
JOIN by_year_month y2017
       ON y2017.month = y2018.month
      AND y2017.year = 2017
WHERE y2018.year = 2018
ORDER BY y2018.month;


-- QUERY 03 | Revenue by category, share and Pareto position | RANK, window SUM
WITH category_rev AS (
    SELECT
        p.product_category,
        COUNT(DISTINCT v.order_key)      AS orders,
        COUNT(*)                         AS items_sold,
        ROUND(SUM(v.item_revenue), 2)    AS revenue,
        ROUND(AVG(v.price), 2)           AS avg_price
    FROM vw_delivered_items v
    JOIN dim_product p ON p.product_key = v.product_key
    GROUP BY p.product_category
)
SELECT
    RANK() OVER (ORDER BY revenue DESC)                              AS revenue_rank,
    product_category,
    orders,
    items_sold,
    revenue,
    avg_price,
    ROUND(100.0 * revenue / SUM(revenue) OVER (), 2)                 AS pct_of_total,
    ROUND(100.0 * SUM(revenue) OVER (ORDER BY revenue DESC)
          / SUM(revenue) OVER (), 2)                                 AS cumulative_pct
FROM category_rev
ORDER BY revenue DESC
LIMIT 15;


-- QUERY 04 | Revenue concentration across sellers (Pareto) | NTILE, window SUM
WITH seller_rev AS (
    SELECT
        seller_key,
        seller_id,
        total_revenue,
        NTILE(10) OVER (ORDER BY total_revenue DESC) AS revenue_decile
    FROM agg_seller_performance
)
SELECT
    revenue_decile,
    COUNT(*)                                                     AS sellers,
    ROUND(SUM(total_revenue), 2)                                 AS revenue,
    ROUND(100.0 * SUM(total_revenue)
          / SUM(SUM(total_revenue)) OVER (), 2)                  AS pct_of_revenue,
    ROUND(AVG(total_revenue), 2)                                 AS avg_seller_revenue
FROM seller_rev
GROUP BY revenue_decile
ORDER BY revenue_decile;

-- QUERY 05 | Top 15 customers by lifetime value | JOIN, window RANK
SELECT
    DENSE_RANK() OVER (ORDER BY c.monetary DESC)     AS value_rank,
    LEFT(c.customer_unique_id, 12)                   AS customer,
    c.state,
    c.frequency                                      AS orders,
    c.monetary                                       AS lifetime_value,
    c.avg_order_value,
    c.favourite_category,
    c.avg_review_score,
    c.segment
FROM vw_customer_360 c
ORDER BY c.monetary DESC
LIMIT 15;


-- QUERY 06 | Repeat-purchase rate | CASE WHEN aggregation
SELECT
    COUNT(*)                                                          AS total_customers,
    SUM(CASE WHEN frequency > 1 THEN 1 ELSE 0 END)                    AS repeat_customers,
    ROUND(100.0 * SUM(CASE WHEN frequency > 1 THEN 1 ELSE 0 END)
          / COUNT(*), 2)                                              AS repeat_rate_pct,
    ROUND(AVG(frequency), 3)                                          AS avg_orders_per_customer,
    ROUND(AVG(monetary), 2)                                           AS avg_lifetime_value,
    ROUND(AVG(CASE WHEN frequency > 1 THEN monetary END), 2)          AS avg_ltv_repeat,
    ROUND(AVG(CASE WHEN frequency = 1 THEN monetary END), 2)          AS avg_ltv_single
FROM agg_customer_rfm;


-- QUERY 07 | Purchase-frequency distribution | GROUP BY, window SUM
SELECT
    frequency                                                     AS orders_placed,
    COUNT(*)                                                      AS customers,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)            AS pct_of_customers,
    ROUND(SUM(monetary), 2)                                       AS revenue,
    ROUND(100.0 * SUM(monetary) / SUM(SUM(monetary)) OVER (), 2)  AS pct_of_revenue,
    ROUND(AVG(monetary), 2)                                       AS avg_lifetime_value
FROM agg_customer_rfm
GROUP BY frequency
ORDER BY frequency;


-- QUERY 08 | RFM segment scorecard | GROUP BY, CASE WHEN, window share
SELECT
    segment,
    COUNT(*)                                                      AS customers,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)            AS pct_customers,
    ROUND(SUM(monetary), 2)                                       AS revenue,
    ROUND(100.0 * SUM(monetary) / SUM(SUM(monetary)) OVER (), 2)  AS pct_revenue,
    ROUND(AVG(monetary), 2)                                       AS avg_value,
    ROUND(AVG(recency_days), 0)                                   AS avg_days_since_purchase,
    ROUND(AVG(avg_review_score), 2)                               AS avg_review,
    ROUND(100.0 * AVG(late_order_rate), 2)                        AS late_order_pct
FROM agg_customer_rfm
GROUP BY segment
ORDER BY revenue DESC;


-- QUERY 09 | Cohort retention — do customers ever come back? | CTE, TIMESTAMPDIFF, CASE
WITH first_purchase AS (
    SELECT
        o.customer_key,
        MIN(d.full_date) AS first_date
    FROM fact_orders o
    JOIN dim_date d ON d.date_key = o.purchase_date_key
    WHERE o.is_delivered = 1
    GROUP BY o.customer_key
),
cohort_orders AS (
    SELECT
        DATE_FORMAT(f.first_date, '%Y-%m')                          AS cohort_month,
        o.customer_key,
        TIMESTAMPDIFF(MONTH, f.first_date, d.full_date)             AS months_since_first
    FROM fact_orders o
    JOIN dim_date d       ON d.date_key = o.purchase_date_key
    JOIN first_purchase f ON f.customer_key = o.customer_key
    WHERE o.is_delivered = 1
)
SELECT
    cohort_month,
    COUNT(DISTINCT CASE WHEN months_since_first = 0 THEN customer_key END) AS cohort_size,
    COUNT(DISTINCT CASE WHEN months_since_first BETWEEN 1 AND 3
                        THEN customer_key END)                             AS returned_m1_m3,
    COUNT(DISTINCT CASE WHEN months_since_first > 3
                        THEN customer_key END)                             AS returned_m4_plus,
    ROUND(100.0 * COUNT(DISTINCT CASE WHEN months_since_first >= 1
                                      THEN customer_key END)
          / NULLIF(COUNT(DISTINCT CASE WHEN months_since_first = 0
                                       THEN customer_key END), 0), 2)      AS retention_pct,
    TIMESTAMPDIFF(MONTH,
                  STR_TO_DATE(CONCAT(cohort_month, '-01'), '%Y-%m-%d'),
                  '2018-10-17')                                            AS months_observed,
    CASE
        WHEN TIMESTAMPDIFF(MONTH,
                           STR_TO_DATE(CONCAT(cohort_month, '-01'), '%Y-%m-%d'),
                           '2018-10-17') < 6
        THEN 'censored - too recent to compare'
        ELSE 'comparable'
    END                                                                    AS window_note
FROM cohort_orders
WHERE cohort_month >= '2017-01'
GROUP BY cohort_month
HAVING cohort_size >= 100
ORDER BY cohort_month;

-- QUERY 10 | Top 15 products by revenue | JOIN, aggregate, subquery in SELECT
SELECT
    ROW_NUMBER() OVER (ORDER BY SUM(v.item_revenue) DESC)   AS rn,
    LEFT(p.product_id, 12)                                  AS product,
    p.product_category,
    COUNT(*)                                                AS units_sold,
    ROUND(SUM(v.item_revenue), 2)                           AS revenue,
    ROUND(AVG(v.price), 2)                                  AS avg_price,
    ROUND(AVG(v.delivery_days), 1)                          AS avg_delivery_days,
    (SELECT ROUND(AVG(r.review_score), 2)
       FROM fact_reviews r
       JOIN fact_order_items oi2 ON oi2.order_key = r.order_key
      WHERE oi2.product_key = p.product_key)                AS avg_review_score
FROM vw_delivered_items v
JOIN dim_product p ON p.product_key = v.product_key
GROUP BY p.product_key, p.product_id, p.product_category
ORDER BY revenue DESC
LIMIT 15;


-- QUERY 11 | Best-selling product within each category | ROW_NUMBER PARTITION BY
WITH ranked AS (
    SELECT
        p.product_category,
        p.product_id,
        COUNT(*)                      AS units_sold,
        ROUND(SUM(v.item_revenue), 2) AS revenue,
        ROW_NUMBER() OVER (PARTITION BY p.product_category
                           ORDER BY SUM(v.item_revenue) DESC) AS rn_in_category
    FROM vw_delivered_items v
    JOIN dim_product p ON p.product_key = v.product_key
    GROUP BY p.product_category, p.product_key, p.product_id
)
SELECT product_category, LEFT(product_id, 12) AS top_product, units_sold, revenue
FROM ranked
WHERE rn_in_category = 1
ORDER BY revenue DESC
LIMIT 15;


-- QUERY 12 | Worst-performing products | HAVING volume floor, CASE WHEN
SELECT
    LEFT(p.product_id, 12)                   AS product,
    p.product_category,
    COUNT(*)                                 AS units_sold,
    ROUND(SUM(v.item_revenue), 2)            AS revenue,
    ROUND(AVG(r.review_score), 2)            AS avg_review_score,
    ROUND(100.0 * AVG(r.is_negative_review), 1) AS negative_review_pct,
    ROUND(AVG(v.delivery_days), 1)           AS avg_delivery_days,
    ROUND(100.0 * AVG(v.is_late), 1)         AS late_pct,
    CASE
        WHEN AVG(r.review_score) < 2.5 AND AVG(v.is_late) > 0.15 THEN 'quality AND logistics'
        WHEN AVG(r.review_score) < 2.5                            THEN 'product quality'
        WHEN AVG(v.is_late) > 0.15                                THEN 'logistics'
        ELSE 'review'
    END                                      AS likely_root_cause
FROM vw_delivered_items v
JOIN dim_product  p ON p.product_key = v.product_key
JOIN fact_reviews r ON r.order_key   = v.order_key
GROUP BY p.product_key, p.product_id, p.product_category
HAVING units_sold >= 20 AND avg_review_score < 3.2
ORDER BY avg_review_score ASC, units_sold DESC
LIMIT 15;

-- QUERY 13 | Top sellers — commercial performance beside service quality | RANK
SELECT
    RANK() OVER (ORDER BY s.total_revenue DESC)     AS revenue_rank,
    LEFT(s.seller_id, 12)                           AS seller,
    s.state,
    s.total_orders,
    s.total_revenue,
    s.avg_order_value,
    s.distinct_categories,
    ROUND(s.avg_review_score, 2)                    AS avg_review,
    ROUND(100.0 * s.late_rate, 1)                   AS late_pct,
    ROUND(s.avg_delivery_days, 1)                   AS avg_delivery_days,
    CASE WHEN s.is_dormant = 1 THEN 'DORMANT' ELSE 'active' END AS status
FROM vw_seller_scorecard s
ORDER BY s.total_revenue DESC
LIMIT 15;


-- QUERY 14 | Sellers to intervene on | HAVING, CASE WHEN, subquery threshold
SELECT
    LEFT(s.seller_id, 12)                    AS seller,
    s.state,
    s.total_orders,
    s.total_revenue,
    ROUND(s.avg_review_score, 2)             AS avg_review,
    ROUND(100.0 * s.late_rate, 1)            AS late_pct,
    ROUND(100.0 * s.negative_review_rate, 1) AS negative_review_pct,
    s.days_since_last_sale,
    CASE
        WHEN s.is_dormant = 1                          THEN 'already churned - win back'
        WHEN s.avg_review_score < 3.5 AND s.late_rate > 0.15 THEN 'critical - quality + delivery'
        WHEN s.late_rate > 0.15                        THEN 'logistics coaching'
        WHEN s.avg_review_score < 3.5                  THEN 'quality review'
        ELSE 'monitor'
    END                                      AS recommended_action
FROM vw_seller_scorecard s
WHERE s.total_orders >= 20
  AND (s.avg_review_score < 3.5 OR s.late_rate > 0.15)
  AND s.total_revenue > (SELECT AVG(total_revenue) FROM agg_seller_performance)
ORDER BY s.total_revenue DESC
LIMIT 15;

-- QUERY 15 | Delivery performance by state | JOIN, HAVING, CASE WHEN
SELECT
    c.state,
    c.region,
    COUNT(*)                                       AS delivered_orders,
    ROUND(AVG(o.delivery_days), 1)                 AS avg_delivery_days,
    ROUND(AVG(o.estimated_delivery_days), 1)       AS avg_promised_days,
    ROUND(AVG(o.delay_days), 1)                    AS avg_vs_promise,
    ROUND(100.0 * AVG(o.is_late), 2)               AS late_pct,
    ROUND(AVG(r.review_score), 2)                  AS avg_review,
    CASE
        WHEN AVG(o.delivery_days) > 20 THEN 'critical'
        WHEN AVG(o.delivery_days) > 15 THEN 'slow'
        WHEN AVG(o.delivery_days) > 10 THEN 'acceptable'
        ELSE 'fast'
    END                                            AS delivery_grade
FROM fact_orders o
JOIN dim_customer c      ON c.customer_key = o.customer_key
LEFT JOIN fact_reviews r ON r.order_key    = o.order_key
WHERE o.is_delivered = 1
GROUP BY c.state, c.region
HAVING delivered_orders >= 200
ORDER BY avg_delivery_days DESC
LIMIT 15;


-- QUERY 16 | Late-delivery rate trend | CTE, LAG, CASE
SELECT
    calendar_month,
    delivered_orders,
    late_orders,
    late_pct,
    LAG(late_pct) OVER (ORDER BY calendar_month)                       AS prev_late_pct,
    ROUND(late_pct - LAG(late_pct) OVER (ORDER BY calendar_month), 2)  AS change_pts,
    CASE
        WHEN late_pct > 15 THEN 'crisis'
        WHEN late_pct > 10 THEN 'elevated'
        WHEN late_pct > 5  THEN 'normal'
        ELSE 'excellent'
    END                                                                AS status
FROM (
    SELECT
        d.calendar_month,
        COUNT(*)                          AS delivered_orders,
        SUM(o.is_late)                    AS late_orders,
        ROUND(100.0 * AVG(o.is_late), 2)  AS late_pct
    FROM fact_orders o
    JOIN dim_date d ON d.date_key = o.purchase_date_key
    WHERE o.is_delivered = 1
    GROUP BY d.calendar_month
) m
WHERE calendar_month >= '2017-01'
ORDER BY calendar_month;


-- QUERY 17 | What a late delivery costs | CASE WHEN buckets, JOIN
SELECT
    CASE
        WHEN o.delay_days <= -10 THEN 'a. 10+ days early'
        WHEN o.delay_days <   0  THEN 'b. early'
        WHEN o.delay_days =  0   THEN 'c. exactly on time'
        WHEN o.delay_days <=  3  THEN 'd. 1-3 days late'
        WHEN o.delay_days <=  7  THEN 'e. 4-7 days late'
        WHEN o.delay_days <= 15  THEN 'f. 8-15 days late'
        ELSE                          'g. 15+ days late'
    END                                             AS delivery_outcome,
    COUNT(*)                                        AS orders,
    ROUND(AVG(r.review_score), 2)                   AS avg_review_score,
    ROUND(100.0 * AVG(r.is_negative_review), 1)     AS negative_review_pct,
    ROUND(100.0 * AVG(CASE WHEN r.review_score = 5 THEN 1 ELSE 0 END), 1) AS five_star_pct
FROM fact_orders o
JOIN fact_reviews r ON r.order_key = o.order_key
WHERE o.is_delivered = 1 AND o.delay_days IS NOT NULL
GROUP BY delivery_outcome
ORDER BY delivery_outcome;

-- QUERY 18 | Rating and negative-review rate by category | HAVING, CASE WHEN
SELECT
    p.product_category,
    COUNT(*)                                        AS reviewed_orders,
    ROUND(AVG(r.review_score), 2)                   AS avg_review_score,
    ROUND(100.0 * AVG(r.is_negative_review), 2)     AS negative_pct,
    ROUND(100.0 * AVG(o.is_late), 2)                AS late_pct,
    ROUND(AVG(o.delivery_days), 1)                  AS avg_delivery_days,
    ROUND(100.0 * AVG(r.has_review_text), 1)        AS pct_with_text,
    CASE
        WHEN AVG(r.review_score) >= 4.2 THEN 'strong'
        WHEN AVG(r.review_score) >= 3.8 THEN 'acceptable'
        ELSE 'needs attention'
    END                                             AS rating_grade
FROM fact_reviews r
JOIN fact_orders  o ON o.order_key   = r.order_key
JOIN fact_order_items i ON i.order_key = o.order_key
JOIN dim_product  p ON p.product_key = i.product_key
WHERE o.is_delivered = 1
GROUP BY p.product_category
HAVING reviewed_orders >= 300
ORDER BY avg_review_score ASC
LIMIT 15;

-- QUERY 19 | Indexes supporting this query layer | information_schema
SELECT
    TABLE_NAME,
    INDEX_NAME,
    GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX) AS columns,
    CASE WHEN NON_UNIQUE = 0 THEN 'UNIQUE' ELSE 'index' END AS kind
FROM information_schema.STATISTICS
WHERE TABLE_SCHEMA = 'ecommerce_dw'
  AND TABLE_NAME IN ('fact_orders', 'fact_order_items', 'fact_reviews', 'dim_date')
GROUP BY TABLE_NAME, INDEX_NAME, NON_UNIQUE
ORDER BY TABLE_NAME, INDEX_NAME;
