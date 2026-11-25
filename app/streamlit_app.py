from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import COLORS, brl, compute_kpis, inject_base_style, load_processed

st.set_page_config(page_title="E-Commerce BI Platform", page_icon="📊", layout="wide")
inject_base_style()

st.title("📊 Predictive E-Commerce BI Platform")
st.caption("Olist marketplace · Sep 2016 – Oct 2018 · Same KPI definitions as the Power BI dashboard")

kpis = compute_kpis()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Revenue", brl(kpis["total_revenue"]))
c2.metric("Total Orders", f"{kpis['total_orders']:,}")
c3.metric("Total Customers", f"{kpis['total_customers']:,}")
c4.metric("Avg Order Value", f"R$ {kpis['avg_order_value']:.2f}")
c5.metric("Avg Rating", f"{kpis['avg_review_score']:.2f} / 5")
c6.metric("On-Time Delivery", f"{kpis['on_time_pct']:.1f}%")

st.divider()

col_a, col_b = st.columns([3, 2])

with col_a:
    st.subheader("Monthly Revenue Trend")
    orders = load_processed("orders")
    items = load_processed("order_items")
    orders["order_purchase_timestamp"] = pd.to_datetime(orders.order_purchase_timestamp)

    delivered = orders[orders.order_status == "delivered"]
    d = items.merge(delivered[["order_id", "order_purchase_timestamp"]], on="order_id")
    monthly = (d.set_index("order_purchase_timestamp").item_revenue
                .resample("MS").sum())
    monthly = monthly[(monthly.index >= "2017-01-01") & (monthly.index <= "2018-08-01")]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly.index, y=monthly.values, mode="lines+markers",
        line=dict(color=COLORS["primary"], width=2.5),
        marker=dict(size=6, color=COLORS["primary"]),
        hovertemplate="%{x|%b %Y}<br>R$ %{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Revenue (R$)", xaxis_title=None,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("What this app adds")
    st.markdown(f"""
    Power BI answers **what happened**. This app answers **what's likely next**
    and **what to do about it** — the two things a static dashboard can't do.

    | Page | Model |
    |---|---|
    | 🎯 Churn Prediction | XGBoost — customer return probability |
    | 📈 Sales Forecast | XGBoost — weekly revenue, forward-looking |
    | 🚚 Delivery Risk | XGBoost — late-delivery probability per order |
    | 👥 Segmentation | K-Means — 4 data-driven customer clusters |

    Delivered orders only ({kpis['delivered_orders']:,} of {kpis['total_orders']:,}) count
    toward revenue and delivery metrics, matching the definition used everywhere
    else in this project.
    """)

st.divider()
st.caption(
    "Every number on this page is computed directly from `data/processed/` — "
    "no cached snapshot, no MySQL dependency. Model pages read from `ml/models/`, "
    "the output of the training scripts in `ml/`."
)
