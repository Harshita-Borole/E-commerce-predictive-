# Predictive E-Commerce Business Intelligence Platform

Turns raw Olist marketplace data into business insights and risk
predictions — across three layers:

| Layer | Question | Stack |
|---|---|---|
| Data Engineering | How do we organise the raw data? | Python · Pandas · MySQL |
| Data Analytics | What happened? | SQL · Power BI |
| Data Science | What is likely to happen? | XGBoost · K-Means · TF-IDF + Logistic Regression, served through a Streamlit app |

## Dataset

Brazilian E-Commerce Public Dataset by Olist — 99,441 orders, 96,096 unique
customers, 3,095 sellers, 32,951 products, **2016-09 → 2018-10**,
R$ 15.4M GMV on delivered orders.

Raw files live in `data/raw/` and are **never modified**. They are not
committed to this repository (the geolocation file alone is ~60 MB). Download
the dataset from
[Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and save
the files into `data/raw/` under these names:

| Kaggle file | Save as |
|---|---|
| `olist_customers_dataset.csv` | `customers.csv` |
| `olist_orders_dataset.csv` | `orders.csv` |
| `olist_order_items_dataset.csv` | `order_items.csv` |
| `olist_order_payments_dataset.csv` | `payments.csv` |
| `olist_order_reviews_dataset.csv` | `reviews.csv` |
| `olist_products_dataset.csv` | `products.csv` |
| `olist_sellers_dataset.csv` | `sellers.csv` |
| `olist_geolocation_dataset.csv` | `geolocation.csv` |
| `product_category_name_translation.csv` | `category_translation.csv` |

## Definitions (single source of truth)

Every number in SQL, Power BI and the ML layer must reconcile to these:

- **Person key** is `customer_unique_id`, *not* `customer_id` (Olist issues a new
  `customer_id` per order — keying on it makes repeat rate read 0%).
- **Revenue** = `SUM(price + freight_value)` where `order_status = 'delivered'`.
- **Order total** = `SUM(payment_value)` grouped by `order_id` (payments are
  per-instalment rows).
- **Late delivery** = `order_delivered_customer_date > order_estimated_delivery_date`.

## Project structure

```
predictive-ecommerce/
├── data/raw/            # untouched Olist CSVs
├── data/processed/      # ETL output, warehouse-ready
├── notebooks/           # EDA + model development
├── etl/                 # extract → transform → validate → load
├── database/            # star schema DDL + views
├── analytics/           # KPI queries
├── ml/                  # churn, forecast, delivery risk, segmentation, review sentiment
├── app/                 # Streamlit front end — 5 pages, reads ml/models/ directly
├── dashboard/           # Power BI DAX measures + relationships, HTML dashboard
└── docs/                # ETL report, ML metrics, EDA figures, dashboard charts
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # then fill in your MySQL credentials
```

Profile the raw data:

```bash
python etl/profile_raw.py
```

## Running the ETL pipeline

```bash
python -m etl.run_etl              # extract -> transform -> validate -> load
python -m etl.run_etl --no-write   # dry run: validate without writing
python -m etl.run_etl --quiet      # report only
```

```bash
python -m etl.run_etl --warehouse   # also build + load the MySQL star schema
```

Reads `data/raw/`, writes the curated layer to `data/processed/`, and saves a
reconciliation report to `docs/etl_report.json`. **37 validation checks must pass
before anything is written** — including a revenue tie-out to R$ 15,419,773.75 to
the cent. Rejected rows are quarantined to `data/processed/quarantine/` with the
rule that rejected them, never silently dropped.

## Running the ML models

```bash
python -m ml.run_all                  # train all six
python -m ml.churn_model              # or one at a time
python -m ml.customer_churn_model
python -m ml.delivery_model
python -m ml.sales_forecast
python -m ml.segmentation
python -m ml.review_sentiment
```

Reads `data/processed/`, writes trained models + scored predictions to
`ml/models/` and a metrics report per model to `docs/ml_reports/`. XGBoost is
the primary algorithm (seller churn, customer churn, delivery delay, sales
forecast); K-Means handles segmentation; TF-IDF + Logistic Regression handles
review sentiment (Portuguese text — no off-the-shelf English model applies).
Model results:

| Model | Metric | Result |
|---|---|---|
| Seller churn | ROC-AUC / PR-AUC | 0.82 / 0.80 |
| Customer churn | ROC-AUC / top-decile lift | 0.55 / 1.99x |
| Delivery delay | ROC-AUC / Recall / F1 | 0.68 / 0.31 / 0.10 |
| Sales forecast | MAE / RMSE / MAPE | R$38.3K / R$44.7K / 16.1% — does not beat naive baseline (reported honestly) |
| Segmentation | k / silhouette | 4 / 0.57 (extended) vs 4 / 0.49 (classic RFM) |
| Review sentiment | accuracy / macro-F1 | 0.79 / 0.66 |

**Two churn models, deliberately.** Module 1's EDA found only 3.00% of
customers ever place a second order — a naive customer-churn classifier
predicts "churn" for everyone and scores ~97% accuracy while catching
nothing. **Seller churn** (dormancy, 49.7% base rate) is the strong, headline
model built on that finding. **Customer churn** is also built — with the
target properly engineered via a cutoff-date + forward-window method
(`ml/customer_churn_model.py`), trained on the minority-positive framing to
avoid the same 97%-majority trap the naive version falls into, and evaluated
honestly: ROC-AUC 0.55 is weak, and the report says so. The one number that
holds up is top-decile lift (1.99x) — customers ranked highest-risk do return
at roughly twice the base rate, which is what the HIGH-risk output band is
built from.

**Sales forecast produces a genuine forward number, not just a backtest.**
`ml/sales_forecast.py` reports MAE/RMSE/MAPE via one-week-ahead walk-forward
evaluation, then recursively forecasts 4 weeks past the end of all available
data (each predicted week feeds the next as a lag input) to produce a
Next-4-Week Forecast and Expected Growth % — while carrying forward the same
"does not beat naive" caveat rather than presenting the forward number as
more precise than the backtest showed it to be.

**Delivery delay adds a leakage-safe `seller_historical_late_rate`.** A
seller's overall late rate would leak that seller's future performance into
predictions for their own earlier orders, so it's computed as an expanding
window shifted by one order — each order sees only that seller's track
record from strictly before it (cold-start sellers fall back to the dataset
average). It's now the #4 most important feature. `product_category` (also
newly added, per this module's feature list) was tested at full cardinality
first — 74 one-hot columns measurably hurt ROC-AUC (0.655 vs 0.677 without)
— then bucketed to the top 12 categories + "other", which recovered the
score AND lifted recall from 22.9% to 30.6%, the metric that matters most
for an early-warning system. Output is a scored table (Order # / Delay
Probability / Risk) with quantile-based risk bands, same reasoning as the
customer-churn model's bands.

**Segmentation now runs classic RFM for real, not just as an assertion.**
`ml/segmentation.py` fits textbook 3-feature (Recency/Frequency/Monetary)
K-Means first, with both elbow (inertia) and silhouette reported per k. It
does NOT collapse into one 90%+ blob the way the 97%-frequency=1 finding
alone might suggest — recency and monetary carry enough real variation on
their own (largest cluster: 54.3%). But that was never the actual
limitation: R, F and M contain zero delivery or review information by
construction, so no choice of k lets classic RFM ever find a
delivery-quality segment. The extended 9-feature version (adds delivery
days, late rate, review score) finds one anyway — 6.6% of customers with a
99% late-delivery rate and 2.26★ average review, invisible to RFM at any k.
Clusters are named to match the requested archetypes (High-Value Loyal,
Frequent Buyers, Occasional Buyers, At-Risk) wherever a cluster's own
profile actually supports that label; "Poor Delivery Experience" keeps its
own name rather than being forced into the template, since it's the one
finding classic RFM structurally cannot produce.

**Review sentiment is a real classifier trained on text, not a rating
lookup.** Only 41.3% of reviews have any text at all, and it's Portuguese,
so `ml/review_sentiment.py` trains TF-IDF + Logistic Regression using
star-rating-derived labels (1-2★→Negative, 3★→Neutral, 4-5★→Positive) as
weak supervision, then predicts sentiment from the TEXT alone — the
percentages reported (57.0% / 27.3% / 15.7%) are the model's predictions on
the 40,396 reviews with text, not a restatement of their existing ratings.
83.7% agreement between predicted-from-text and rating-derived sentiment
validates the model; the 1,269 disagreement cases (positive rating, negative
text) are flagged as likely sarcasm or a problem later resolved. NMF topic
modelling over the text-predicted-negative reviews independently confirms
the delivery-drives-dissatisfaction finding from Module 5's SQL layer — 5 of
6 discovered topics are unmistakably about non-delivery ("não recebi" /
didn't receive, "ainda aguardando" / still waiting), this time as direct
textual evidence rather than a statistical correlation. Category and seller
breakdowns use a minimum-review-count floor (the same discipline as every
"worst N" ranking elsewhere in this project) and exclude the "unknown"
category (no product category recorded — a data gap, not a finding) from
the headline list.

## Running the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Run the ETL pipeline first: the Overview page reads `data/processed/`, which
is not committed. The model outputs in `ml/models/` are included.

Five pages (Streamlit's native `pages/` convention — a real sidebar
navigator, no custom routing code): Overview, Churn Prediction, Sales
Forecast, Delivery Risk, Segmentation. Every page reads directly from
`ml/models/*.csv` and `docs/ml_reports/*.json` — the actual output of the
training scripts above, not a hand-copied snapshot — so re-running any model
and refreshing the app shows the new numbers with no other step. No MySQL
dependency, consistent with the rest of the ML layer.

Why this app exists alongside Power BI rather than replacing it: Power BI
answers *what happened*; this app answers *what's likely next* — the two
things a static dashboard can't do, since it can't call a trained model live.
The Delivery Risk page includes a live order lookup (select an Order ID, see
its prediction render on its own); every page keeps the same honest-metrics
discipline as the training scripts — the churn page's ROC-AUC 0.55 caveat and
the forecast page's "does not beat naive baseline" warning are shown in the
app itself, not just in the docs.

## Build status

- [x] **Module 1** — dataset understanding, project scaffold, raw data staged
- [x] **Module 2** — EDA ([notebooks/01_eda.ipynb](notebooks/01_eda.ipynb), 12 charts in [docs/figures/](docs/figures/))
- [x] **Module 3** — ETL pipeline ([etl/](etl/)) · 37/37 validation checks pass, revenue reconciles to R$0.00
- [x] **Module 4** — MySQL star schema ([database/](database/)) · 11 tables + 11 views loaded, 10/10 warehouse checks pass
- [x] **Module 5** — SQL analytics ([analytics/queries.sql](analytics/queries.sql)) · 19 queries, 19/19 pass (`python -m analytics.run_queries --report` writes the results to `docs/kpi_report.md`)
- [x] **Module 6** — Dashboards ([dashboard/](dashboard/)) · 4-page Power BI report ([charts](docs/PowerBI_Dashboard_Charts.pdf)) + DAX library + live HTML dashboard
- [x] **Module 7** — ML core ([ml/](ml/)) · seller churn, delivery delay, weekly sales forecast, K-Means segmentation
- [x] **Module 8** — Customer churn ([ml/customer_churn_model.py](ml/customer_churn_model.py)) · cutoff-date target engineering, ROC-AUC 0.55 (honest), top-decile lift 1.99x
- [x] **Module 9** — Sales forecast extended ([ml/sales_forecast.py](ml/sales_forecast.py)) · MAE/RMSE/MAPE + genuine forward forecast (Next-4-Week + growth %)
- [x] **Module 10** — Delivery delay extended ([ml/delivery_model.py](ml/delivery_model.py)) · +seller_historical_late_rate (leakage-safe) +product_category, recall 22.9%→30.6%
- [x] **Module 11** — Segmentation extended ([ml/segmentation.py](ml/segmentation.py)) · elbow + silhouette, classic RFM run and compared against the extended feature set
- [x] **Module 12** — Review sentiment ([ml/review_sentiment.py](ml/review_sentiment.py)) · TF-IDF + LogReg (Portuguese), 79% accuracy, NMF complaint topics
- [x] **Streamlit app** ([app/](app/)) · 5 pages — Overview, Churn Prediction, Sales Forecast, Delivery Risk (with live order lookup), Segmentation

No AI advisor — the LLM layer was intentionally cut. Everything above stands
on its own: SQL/Power BI for descriptive analytics, the ML layer for
prediction, Streamlit for a live interface to it.
