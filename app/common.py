from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "ml" / "models"
REPORTS = ROOT / "docs" / "ml_reports"

COLORS = {
    "primary": "#2A78D6", "secondary": "#EB6834", "good": "#0CA30C",
    "warn": "#FAB219", "critical": "#D03B3B", "muted": "#898781",
}
RISK_COLORS = {"LOW": COLORS["good"], "MEDIUM": COLORS["warn"], "HIGH": COLORS["critical"]}
CLUSTER_PALETTE = ["#2A78D6", "#EB6834", "#1BAF7A", "#EDA100", "#E87BA4", "#4A3AA7"]


@st.cache_data
def load_processed(table: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED / f"{table}.csv", low_memory=False)


@st.cache_data
def load_model_output(name: str) -> pd.DataFrame:
    return pd.read_csv(MODELS / f"{name}.csv", low_memory=False)


@st.cache_data
def load_report(name: str) -> dict:
    path = REPORTS / f"{name}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def compute_kpis() -> dict:
    orders = load_processed("orders")
    items = load_processed("order_items")
    reviews = load_processed("reviews")

    delivered_orders = orders[orders.order_status == "delivered"]
    delivered_ids = set(delivered_orders.order_id)
    delivered_items = items[items.order_id.isin(delivered_ids)]

    revenue = float(delivered_items.item_revenue.sum())
    order_value = delivered_items.groupby("order_id").item_revenue.sum()

    return {
        "total_revenue": revenue,
        "total_orders": int(orders.order_id.nunique()),
        "delivered_orders": int(len(delivered_orders)),
        "total_customers": int(load_processed("customers").customer_unique_id.nunique()),
        "avg_order_value": float(order_value.mean()),
        "avg_review_score": float(reviews.review_score.mean()),
        "on_time_pct": float(100 * (1 - delivered_orders.is_late.fillna(0).mean())),
        "avg_delivery_days": float(delivered_orders.delivery_days.mean()),
    }


def brl(x: float) -> str:
    if abs(x) >= 1e6:
        return f"R$ {x/1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"R$ {x/1e3:.0f}K"
    return f"R$ {x:.0f}"


def risk_badge(risk: str) -> str:
    color = RISK_COLORS.get(str(risk).upper(), COLORS["muted"])
    return f'<span style="background:{color}22;color:{color};padding:2px 10px;' \
           f'border-radius:12px;font-weight:600;font-size:0.85em">{risk}</span>'


def inject_base_style() -> None:
    st.markdown("""
        <style>
        div[data-testid="stMetric"] {
            background: var(--background-color, #fff);
            border: 1px solid rgba(128,128,128,0.2);
            border-radius: 10px;
            padding: 14px 16px 10px;
        }
        div[data-testid="stMetricLabel"] { font-size: 0.8rem; }
        </style>
    """, unsafe_allow_html=True)
