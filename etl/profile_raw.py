import pandas as pd, os, warnings
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
files = ["customers","orders","order_items","payments","reviews","products","sellers","geolocation","category_translation"]
df = {f: pd.read_csv(os.path.join(RAW, f + ".csv")) for f in files}

print("="*70)
print("TABLE SHAPES")
print("="*70)
for f in files:
    print(f"{f:22} rows={len(df[f]):>8,}  cols={df[f].shape[1]}")

print("\n" + "="*70)
print("COLUMNS / DTYPES / NULLS / DUPES")
print("="*70)
for f in files:
    d = df[f]
    print(f"\n--- {f} ({len(d):,} rows, {d.duplicated().sum():,} full-dup rows) ---")
    for c in d.columns:
        n = d[c].isna().sum()
        pct = 100*n/len(d)
        print(f"  {c:35} {str(d[c].dtype):10} nulls={n:>7,} ({pct:5.2f}%)  uniq={d[c].nunique():>7,}")

print("\n" + "="*70)
print("KEY CARDINALITY / RELATIONSHIPS")
print("="*70)
o, c, oi, p, r = df["orders"], df["customers"], df["order_items"], df["payments"], df["reviews"]
print(f"customers.customer_id unique      : {c.customer_id.nunique():,}  (rows {len(c):,})")
print(f"customers.customer_unique_id uniq : {c.customer_unique_id.nunique():,}  <-- REAL people")
print(f"orders.order_id unique            : {o.order_id.nunique():,}  (rows {len(o):,})")
print(f"order_items rows per order        : {len(oi)/oi.order_id.nunique():.2f} avg")
print(f"payments rows per order           : {len(p)/p.order_id.nunique():.2f} avg")
print(f"reviews.review_id unique          : {r.review_id.nunique():,} (rows {len(r):,})")
print(f"reviews.order_id unique           : {r.order_id.nunique():,}")

print("\n-- Orphan / referential integrity checks --")
print(f"orders w/ customer_id not in customers : {(~o.customer_id.isin(c.customer_id)).sum():,}")
print(f"order_items w/ order_id not in orders  : {(~oi.order_id.isin(o.order_id)).sum():,}")
print(f"orders with NO order_items            : {(~o.order_id.isin(oi.order_id)).sum():,}")
print(f"orders with NO payment                : {(~o.order_id.isin(p.order_id)).sum():,}")
print(f"orders with NO review                 : {(~o.order_id.isin(r.order_id)).sum():,}")
print(f"order_items w/ product not in products: {(~oi.product_id.isin(df['products'].product_id)).sum():,}")
print(f"order_items w/ seller not in sellers  : {(~oi.seller_id.isin(df['sellers'].seller_id)).sum():,}")

print("\n" + "="*70)
print("ORDER STATUS + DATE COVERAGE")
print("="*70)
print(o.order_status.value_counts().to_string())
for col in [c for c in o.columns if "date" in c or "timestamp" in c]:
    s = pd.to_datetime(o[col], errors="coerce")
    print(f"{col:36} min={str(s.min())[:10]}  max={str(s.max())[:10]}  nulls={o[col].isna().sum():,}")

print("\n" + "="*70)
print("REPEAT PURCHASE BEHAVIOUR (churn feasibility)")
print("="*70)
oc = o.merge(c[["customer_id","customer_unique_id"]], on="customer_id", how="left")
delivered = oc[oc.order_status == "delivered"]
cnt = delivered.groupby("customer_unique_id").order_id.nunique()
print(f"unique customers (delivered orders): {len(cnt):,}")
print("orders per customer distribution:")
print(cnt.value_counts().sort_index().head(8).to_string())
print(f"\n% customers with >1 order: {100*(cnt>1).mean():.2f}%")

print("\n" + "="*70)
print("REVIEW SCORES + TEXT AVAILABILITY (NLP feasibility)")
print("="*70)
print(r.review_score.value_counts().sort_index().to_string())
print(f"\nreviews with comment text : {r.review_comment_message.notna().sum():,} ({100*r.review_comment_message.notna().mean():.1f}%)")
print(f"reviews with title        : {r.review_comment_title.notna().sum():,}")
neg = r[r.review_score <= 2]
print(f"negative reviews (1-2*)   : {len(neg):,}, with text: {neg.review_comment_message.notna().sum():,}")

print("\n" + "="*70)
print("DELIVERY / DELAY FEASIBILITY")
print("="*70)
d2 = o[o.order_status=="delivered"].copy()
for cc in ["order_purchase_timestamp","order_delivered_customer_date","order_estimated_delivery_date","order_approved_at","order_delivered_carrier_date"]:
    d2[cc] = pd.to_datetime(d2[cc], errors="coerce")
d2["delivery_days"] = (d2.order_delivered_customer_date - d2.order_purchase_timestamp).dt.days
d2["late_days"] = (d2.order_delivered_customer_date - d2.order_estimated_delivery_date).dt.days
d2["is_late"] = (d2.late_days > 0).astype(int)
print(f"delivered orders                : {len(d2):,}")
print(f"delivery_days  mean={d2.delivery_days.mean():.1f}  median={d2.delivery_days.median():.0f}  p95={d2.delivery_days.quantile(.95):.0f}  max={d2.delivery_days.max():.0f}")
print(f"LATE orders (past estimate)     : {d2.is_late.sum():,} ({100*d2.is_late.mean():.2f}%)  <-- ML target base rate")
print(f"null delivered_customer_date    : {d2.order_delivered_customer_date.isna().sum():,}")

print("\n" + "="*70)
print("REVENUE / SALES FORECAST FEASIBILITY")
print("="*70)
oi2 = oi.merge(o[["order_id","order_purchase_timestamp","order_status"]], on="order_id")
oi2["order_purchase_timestamp"] = pd.to_datetime(oi2.order_purchase_timestamp)
oi2["revenue"] = oi2.price + oi2.freight_value
deliv = oi2[oi2.order_status=="delivered"]
m = deliv.set_index("order_purchase_timestamp").revenue.resample("MS").sum()
print(f"total GMV (delivered): R$ {deliv.revenue.sum():,.0f}")
print(f"months of history: {len(m)}")
print(m.to_string(float_format=lambda x: f"{x:,.0f}"))

print("\n" + "="*70)
print("PRODUCTS / CATEGORIES")
print("="*70)
pr = df["products"]
print(f"products: {len(pr):,}  categories: {pr.product_category_name.nunique()}  translation rows: {len(df['category_translation'])}")
missing_cat = set(pr.product_category_name.dropna()) - set(df["category_translation"].product_category_name)
print(f"categories missing English translation: {missing_cat}")
print(f"products with null category: {pr.product_category_name.isna().sum():,}")

print("\n" + "="*70)
print("GEOLOCATION")
print("="*70)
g = df["geolocation"]
print(f"rows={len(g):,}  unique zip prefixes={g.geolocation_zip_code_prefix.nunique():,}  full dupes={g.duplicated().sum():,}")
print(f"customer zips not in geo: {(~c.customer_zip_code_prefix.isin(g.geolocation_zip_code_prefix)).sum():,}")
print(f"seller   zips not in geo: {(~df['sellers'].seller_zip_code_prefix.isin(g.geolocation_zip_code_prefix)).sum():,}")
