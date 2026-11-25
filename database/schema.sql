DROP DATABASE IF EXISTS ecommerce_dw;
CREATE DATABASE ecommerce_dw
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;
USE ecommerce_dw;

SET FOREIGN_KEY_CHECKS = 0;

CREATE TABLE dim_date (
    date_key            INT           NOT NULL,
    full_date           DATE          NOT NULL,
    year                SMALLINT      NOT NULL,
    quarter             TINYINT       NOT NULL,
    month               TINYINT       NOT NULL,
    month_name          VARCHAR(12)   NOT NULL,
    calendar_month          CHAR(7)       NOT NULL,
    day_of_month        TINYINT       NOT NULL,
    day_of_week         TINYINT       NOT NULL,
    day_name            VARCHAR(12)   NOT NULL,
    week_of_year        TINYINT       NOT NULL,
    is_weekend          TINYINT(1)    NOT NULL DEFAULT 0,
    PRIMARY KEY (date_key),
    UNIQUE KEY uq_dim_date_full (full_date),
    KEY ix_dim_date_ym (calendar_month),
    KEY ix_dim_date_calendar_month (year, month)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE dim_geography (
    geography_key       INT           NOT NULL AUTO_INCREMENT,
    zip_code_prefix     INT           NOT NULL,
    city                VARCHAR(100)  NULL,
    state               CHAR(2)       NULL,
    region              VARCHAR(20)   NULL,
    latitude            DECIMAL(10,7) NULL,
    longitude           DECIMAL(10,7) NULL,
    PRIMARY KEY (geography_key),
    UNIQUE KEY uq_geo_zip (zip_code_prefix),
    KEY ix_geo_state (state),
    KEY ix_geo_region (region)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE dim_customer (
    customer_key        INT           NOT NULL AUTO_INCREMENT,
    customer_unique_id  CHAR(32)      NOT NULL,
    geography_key       INT           NULL,
    city                VARCHAR(100)  NULL,
    state               CHAR(2)       NULL,
    region              VARCHAR(20)   NULL,
    PRIMARY KEY (customer_key),
    UNIQUE KEY uq_customer_unique_id (customer_unique_id),
    KEY ix_customer_geo (geography_key),
    KEY ix_customer_state (state),
    CONSTRAINT fk_customer_geo FOREIGN KEY (geography_key)
        REFERENCES dim_geography (geography_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE dim_product (
    product_key         INT           NOT NULL AUTO_INCREMENT,
    product_id          CHAR(32)      NOT NULL,
    product_category    VARCHAR(100)  NOT NULL DEFAULT 'unknown',
    product_category_pt VARCHAR(100)  NULL,
    weight_g            DECIMAL(10,2) NULL,
    length_cm           DECIMAL(10,2) NULL,
    height_cm           DECIMAL(10,2) NULL,
    width_cm            DECIMAL(10,2) NULL,
    volume_cm3          DECIMAL(14,2) NULL,
    photos_qty          SMALLINT      NULL,
    name_length         SMALLINT      NULL,
    description_length  INT           NULL,
    PRIMARY KEY (product_key),
    UNIQUE KEY uq_product_id (product_id),
    KEY ix_product_category (product_category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE dim_seller (
    seller_key          INT           NOT NULL AUTO_INCREMENT,
    seller_id           CHAR(32)      NOT NULL,
    geography_key       INT           NULL,
    city                VARCHAR(100)  NULL,
    state               CHAR(2)       NULL,
    region              VARCHAR(20)   NULL,
    PRIMARY KEY (seller_key),
    UNIQUE KEY uq_seller_id (seller_id),
    KEY ix_seller_geo (geography_key),
    KEY ix_seller_state (state),
    CONSTRAINT fk_seller_geo FOREIGN KEY (geography_key)
        REFERENCES dim_geography (geography_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE fact_orders (
    order_key               INT           NOT NULL AUTO_INCREMENT,
    order_id                CHAR(32)      NOT NULL,
    customer_key            INT           NOT NULL,
    customer_id             CHAR(32)      NOT NULL,
    purchase_date_key       INT           NOT NULL,
    approved_date_key       INT           NULL,
    delivered_date_key      INT           NULL,
    estimated_date_key      INT           NULL,

    order_status            VARCHAR(20)   NOT NULL,
    is_delivered            TINYINT(1)    NOT NULL DEFAULT 0,

    estimated_delivery_days SMALLINT      NULL,
    purchase_hour           TINYINT       NULL,
    is_weekend_purchase     TINYINT(1)    NULL,
    item_count              SMALLINT      NOT NULL DEFAULT 0,

    delivery_days           SMALLINT      NULL,
    delay_days              SMALLINT      NULL,
    is_late                 TINYINT(1)    NULL,
    approval_hours          DECIMAL(10,2) NULL,
    carrier_handover_days   SMALLINT      NULL,

    order_revenue           DECIMAL(12,2) NOT NULL DEFAULT 0.00,

    PRIMARY KEY (order_key),
    UNIQUE KEY uq_order_id (order_id),
    KEY ix_fo_customer (customer_key),
    KEY ix_fo_purchase_date (purchase_date_key),
    KEY ix_fo_status (order_status),
    KEY ix_fo_delivered (is_delivered),
    KEY ix_fo_late (is_late),
    CONSTRAINT fk_fo_customer  FOREIGN KEY (customer_key)      REFERENCES dim_customer (customer_key),
    CONSTRAINT fk_fo_pdate     FOREIGN KEY (purchase_date_key) REFERENCES dim_date (date_key),
    CONSTRAINT fk_fo_adate     FOREIGN KEY (approved_date_key) REFERENCES dim_date (date_key),
    CONSTRAINT fk_fo_ddate     FOREIGN KEY (delivered_date_key)REFERENCES dim_date (date_key),
    CONSTRAINT fk_fo_edate     FOREIGN KEY (estimated_date_key)REFERENCES dim_date (date_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE fact_order_items (
    order_item_key              INT           NOT NULL AUTO_INCREMENT,
    order_id                    CHAR(32)      NOT NULL,
    order_item_id               SMALLINT      NOT NULL,
    order_key                   INT           NOT NULL,
    customer_key                INT           NOT NULL,
    product_key                 INT           NOT NULL,
    seller_key                  INT           NOT NULL,
    purchase_date_key           INT           NOT NULL,

    order_status                VARCHAR(20)   NOT NULL,
    is_delivered                TINYINT(1)    NOT NULL DEFAULT 0,

    review_score                TINYINT       NULL,
    is_negative_review          TINYINT(1)    NULL,

    price                       DECIMAL(10,2) NOT NULL,
    freight_value               DECIMAL(10,2) NOT NULL,
    item_revenue                DECIMAL(10,2) NOT NULL,
    freight_ratio               DECIMAL(10,4) NULL,
    seller_customer_distance_km DECIMAL(10,2) NULL,
    shipping_limit_date         DATETIME      NULL,

    PRIMARY KEY (order_item_key),
    UNIQUE KEY uq_order_item (order_id, order_item_id),
    KEY ix_foi_order (order_key),
    KEY ix_foi_customer (customer_key),
    KEY ix_foi_product (product_key),
    KEY ix_foi_seller (seller_key),
    KEY ix_foi_date (purchase_date_key),
    KEY ix_foi_delivered (is_delivered),
    CONSTRAINT fk_foi_order    FOREIGN KEY (order_key)         REFERENCES fact_orders (order_key),
    CONSTRAINT fk_foi_customer FOREIGN KEY (customer_key)      REFERENCES dim_customer (customer_key),
    CONSTRAINT fk_foi_product  FOREIGN KEY (product_key)       REFERENCES dim_product (product_key),
    CONSTRAINT fk_foi_seller   FOREIGN KEY (seller_key)        REFERENCES dim_seller (seller_key),
    CONSTRAINT fk_foi_date     FOREIGN KEY (purchase_date_key) REFERENCES dim_date (date_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE fact_payments (
    payment_key          INT           NOT NULL AUTO_INCREMENT,
    order_key            INT           NOT NULL,
    order_id             CHAR(32)      NOT NULL,
    payment_type         VARCHAR(30)   NULL,
    payment_installments SMALLINT      NULL,
    payment_total        DECIMAL(12,2) NOT NULL,
    payment_slices       SMALLINT      NULL,
    payment_methods_used TINYINT       NULL,
    is_installment_plan  TINYINT(1)    NULL,
    PRIMARY KEY (payment_key),
    UNIQUE KEY uq_payment_order (order_id),
    KEY ix_fp_order (order_key),
    KEY ix_fp_type (payment_type),
    CONSTRAINT fk_fp_order FOREIGN KEY (order_key) REFERENCES fact_orders (order_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE fact_reviews (
    review_key              INT           NOT NULL AUTO_INCREMENT,
    review_id               CHAR(32)      NOT NULL,
    order_key               INT           NOT NULL,
    order_id                CHAR(32)      NOT NULL,
    customer_key            INT           NOT NULL,
    review_creation_date_key INT          NULL,

    review_score            TINYINT       NOT NULL,
    is_negative_review      TINYINT(1)    NOT NULL DEFAULT 0,
    is_positive_review      TINYINT(1)    NOT NULL DEFAULT 0,
    has_review_text         TINYINT(1)    NOT NULL DEFAULT 0,
    review_text_length      INT           NULL,
    response_days           DECIMAL(10,2) NULL,
    review_comment_title    VARCHAR(255)  NULL,
    review_comment_message  TEXT          NULL,

    PRIMARY KEY (review_key),
    UNIQUE KEY uq_review_order (order_id),
    KEY ix_fr_order (order_key),
    KEY ix_fr_customer (customer_key),
    KEY ix_fr_score (review_score),
    KEY ix_fr_negative (is_negative_review),
    CONSTRAINT fk_fr_order    FOREIGN KEY (order_key)                REFERENCES fact_orders (order_key),
    CONSTRAINT fk_fr_customer FOREIGN KEY (customer_key)             REFERENCES dim_customer (customer_key),
    CONSTRAINT fk_fr_date     FOREIGN KEY (review_creation_date_key) REFERENCES dim_date (date_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE agg_customer_rfm (
    customer_key         INT           NOT NULL,
    customer_unique_id   CHAR(32)      NOT NULL,
    recency_days         INT           NULL,
    frequency            INT           NULL,
    monetary             DECIMAL(12,2) NULL,
    avg_order_value      DECIMAL(12,2) NULL,
    total_items          INT           NULL,
    avg_item_price       DECIMAL(10,2) NULL,
    total_freight        DECIMAL(12,2) NULL,
    tenure_days          INT           NULL,
    first_purchase       DATETIME      NULL,
    last_purchase        DATETIME      NULL,
    avg_delivery_days    DECIMAL(10,2) NULL,
    late_orders          INT           NULL,
    late_order_rate      DECIMAL(6,4)  NULL,
    avg_review_score     DECIMAL(4,2)  NULL,
    distinct_categories  INT           NULL,
    favourite_category   VARCHAR(100)  NULL,
    is_repeat_customer   TINYINT(1)    NULL,
    r_score              TINYINT       NULL,
    f_score              TINYINT       NULL,
    m_score              TINYINT       NULL,
    rfm_score            TINYINT       NULL,
    segment              VARCHAR(30)   NULL,
    PRIMARY KEY (customer_key),
    UNIQUE KEY uq_rfm_unique_id (customer_unique_id),
    KEY ix_rfm_segment (segment),
    CONSTRAINT fk_rfm_customer FOREIGN KEY (customer_key) REFERENCES dim_customer (customer_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE agg_seller_performance (
    seller_key              INT           NOT NULL,
    seller_id               CHAR(32)      NOT NULL,
    total_orders            INT           NULL,
    total_items             INT           NULL,
    total_revenue           DECIMAL(14,2) NULL,
    avg_order_value         DECIMAL(12,2) NULL,
    avg_item_price          DECIMAL(10,2) NULL,
    avg_freight             DECIMAL(10,2) NULL,
    avg_delivery_days       DECIMAL(10,2) NULL,
    late_orders             INT           NULL,
    late_rate               DECIMAL(6,4)  NULL,
    avg_distance_km         DECIMAL(10,2) NULL,
    distinct_products       INT           NULL,
    distinct_categories     INT           NULL,
    first_sale              DATETIME      NULL,
    last_sale               DATETIME      NULL,
    active_days             INT           NULL,
    days_since_last_sale    INT           NULL,
    orders_per_active_month DECIMAL(10,2) NULL,
    avg_review_score        DECIMAL(4,2)  NULL,
    negative_reviews        INT           NULL,
    total_reviews           INT           NULL,
    negative_review_rate    DECIMAL(6,4)  NULL,
    is_dormant              TINYINT(1)    NULL,
    PRIMARY KEY (seller_key),
    UNIQUE KEY uq_perf_seller_id (seller_id),
    KEY ix_perf_dormant (is_dormant),
    CONSTRAINT fk_perf_seller FOREIGN KEY (seller_key) REFERENCES dim_seller (seller_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

SET FOREIGN_KEY_CHECKS = 1;
