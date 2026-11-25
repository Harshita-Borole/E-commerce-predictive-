from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

RAW_DIR        = ROOT / os.getenv("RAW_DATA_PATH", "data/raw")
PROCESSED_DIR  = ROOT / os.getenv("PROCESSED_DATA_PATH", "data/processed")
QUARANTINE_DIR = PROCESSED_DIR / "quarantine"
REPORT_DIR     = ROOT / "docs"

for _d in (PROCESSED_DIR, QUARANTINE_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RAW_TABLES = [
    "customers", "orders", "order_items", "payments",
    "reviews", "products", "sellers", "geolocation", "category_translation",
]

EXPECTED_COLUMNS: dict[str, list[str]] = {
    "customers":   ["customer_id", "customer_unique_id", "customer_zip_code_prefix",
                    "customer_city", "customer_state"],
    "orders":      ["order_id", "customer_id", "order_status", "order_purchase_timestamp",
                    "order_approved_at", "order_delivered_carrier_date",
                    "order_delivered_customer_date", "order_estimated_delivery_date"],
    "order_items": ["order_id", "order_item_id", "product_id", "seller_id",
                    "shipping_limit_date", "price", "freight_value"],
    "payments":    ["order_id", "payment_sequential", "payment_type",
                    "payment_installments", "payment_value"],
    "reviews":     ["review_id", "order_id", "review_score", "review_comment_title",
                    "review_comment_message", "review_creation_date",
                    "review_answer_timestamp"],
    "products":    ["product_id", "product_category_name", "product_name_lenght",
                    "product_description_lenght", "product_photos_qty",
                    "product_weight_g", "product_length_cm", "product_height_cm",
                    "product_width_cm"],
    "sellers":     ["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"],
    "geolocation": ["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng",
                    "geolocation_city", "geolocation_state"],
    "category_translation": ["product_category_name", "product_category_name_english"],
}

PERSON_KEY      = "customer_unique_id"
REVENUE_STATUS  = "delivered"

ORDER_DATE_COLS = [
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

VALID_ORDER_STATUS = {
    "delivered", "shipped", "canceled", "unavailable",
    "invoiced", "processing", "created", "approved",
}
REVIEW_SCORE_RANGE = (1, 5)

SNAPSHOT_DATE   = "2018-10-17"
DATA_START_DATE = "2016-09-04"

CHURN_WINDOW_DAYS  = 90
REPEAT_WINDOW_DAYS = 180

CATEGORY_TRANSLATION_PATCH = {
    "pc_gamer": "pc_gamer",
    "portateis_cozinha_e_preparadores_de_alimentos": "kitchen_portables_food_preparers",
}
UNKNOWN_CATEGORY = "unknown"

BASELINE = {
    "raw_rows": {
        "customers": 99_441, "orders": 99_441, "order_items": 112_650,
        "payments": 103_886, "reviews": 99_224, "products": 32_951,
        "sellers": 3_095, "geolocation": 1_000_163, "category_translation": 71,
    },
    "total_orders":       99_441,
    "delivered_orders":   96_478,
    "unique_customers":   96_096,
    "sellers":            3_095,
    "delivered_items":    110_197,
    "total_revenue":      15_419_773.75,
    "price_sum":          13_221_498.11,
    "freight_sum":        2_198_275.64,
    "geo_zip_prefixes":   19_015,
    "late_orders":        6_534,
    "avg_delivery_days":  12.09,
}
REVENUE_TOLERANCE = 0.01

MYSQL = {
    "host":     os.getenv("MYSQL_HOST", "localhost"),
    "port":     int(os.getenv("MYSQL_PORT", 3306)),
    "user":     os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "ecommerce_dw"),
}


def mysql_url(include_db: bool = True) -> str:
    db = MYSQL["database"] if include_db else ""
    return (f"mysql+pymysql://{MYSQL['user']}:{MYSQL['password']}"
            f"@{MYSQL['host']}:{MYSQL['port']}/{db}?charset=utf8mb4")
