from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine

from etl import config as cfg

OUT_DIR = cfg.ROOT / "dashboard" / "powerbi_csv"

TABLES = [
    "dim_date", "dim_geography", "dim_customer", "dim_product", "dim_seller",
    "fact_orders", "fact_order_items", "fact_payments", "fact_reviews",
    "agg_customer_rfm", "agg_seller_performance",
]

DROP_COLUMNS = {"fact_reviews": ["review_comment_message", "review_comment_title"]}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    engine = create_engine(cfg.mysql_url(), future=True)
    raw = engine.raw_connection()

    total = 0
    try:
        for table in TABLES:
            cur = raw.cursor()
            cur.execute(f"SELECT * FROM {table}")
            df = pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])
            cur.close()

            df = df.drop(columns=DROP_COLUMNS.get(table, []), errors="ignore")
            path = OUT_DIR / f"{table}.csv"
            df.to_csv(path, index=False, encoding="utf-8-sig")
            total += len(df)
            print(f"  {table:24} {len(df):>8,} rows -> {path.name}")
    finally:
        raw.close()
        engine.dispose()

    print(f"\n{total:,} rows exported to {OUT_DIR}")
    print("Power BI: Get Data > Text/CSV, once per file (11 imports).")
    print("Do NOT use Get Data > Folder > Combine — it appends all 11 tables into one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
