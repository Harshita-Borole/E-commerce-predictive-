from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go
import streamlit as st

from common import RISK_COLORS, inject_base_style, load_model_output, load_report, risk_badge

st.set_page_config(page_title="Delivery Risk", page_icon="🚚", layout="wide")
inject_base_style()

st.title("🚚 Delivery Delay Risk")
st.caption("XGBoost Classifier · seller_historical_late_rate computed as a leakage-safe expanding window "
          "(see docs/ml_reports/delivery_delay.json)")

report = load_report("delivery_delay")
scores = load_model_output("delivery_delay_scores")
m = report.get("metrics", {})

c1, c2, c3, c4 = st.columns(4)
c1.metric("Orders Scored", f"{len(scores):,}")
c2.metric("ROC-AUC", f"{m.get('roc_auc', 0):.2f}")
c3.metric("Recall (late orders caught)", f"{m.get('recall', 0)*100:.1f}%")
c4.metric("HIGH Risk Orders", f"{(scores.risk == 'HIGH').sum():,}")

with st.expander("ℹ️ Why not accuracy — read before trusting a single number here"):
    st.markdown(report.get("why_not_accuracy", ""))

st.divider()

st.subheader("🔍 Look up an order")
lc1, lc2 = st.columns([2, 3])
with lc1:
    order_id = st.selectbox(
        "Select an Order ID (or type to search)",
        options=scores.order_id.tolist(), index=None,
        placeholder="Start typing an order ID…",
    )

if order_id:
    row = scores.loc[scores.order_id == order_id].iloc[0]
    with lc2:
        b1, b2, b3 = st.columns(3)
        b1.metric("Order #", order_id[:14] + "…")
        b2.metric("Delay Probability", f"{row.delay_probability*100:.0f}%")
        b3.markdown(
            f"<div style='padding-top:8px'><span style='font-size:0.8rem;color:#898781'>Risk</span><br>"
            f"{risk_badge(row.risk)}</div>", unsafe_allow_html=True)
else:
    st.caption("Pick an order above to see its individual prediction.")

st.divider()

fc1, fc2 = st.columns([2, 1])
with fc1:
    search = st.text_input("Search Order ID", placeholder="e.g. 74d91d0ed6…")
with fc2:
    risk_filter = st.multiselect("Risk level", ["HIGH", "MEDIUM", "LOW"],
                                 default=["HIGH", "MEDIUM", "LOW"], key="delivery_risk_filter")

view = scores[scores.risk.isin(risk_filter)]
if search:
    view = view[view.order_id.str.contains(search, case=False, na=False)]

col_table, col_chart = st.columns([3, 1])
with col_table:
    st.subheader(f"Ranked Risk List  ({len(view):,} orders)")
    display = view[["order_id", "delay_probability", "risk"]].copy()
    display["delay_probability"] = (display["delay_probability"] * 100).round(1)
    st.dataframe(
        display.rename(columns={"order_id": "Order #",
                                "delay_probability": "Delay Probability (%)",
                                "risk": "Risk"}),
        use_container_width=True, height=480, hide_index=True,
        column_config={
            "Delay Probability (%)": st.column_config.ProgressColumn(
                "Delay Probability (%)", min_value=0, max_value=100, format="%.1f%%"),
        },
    )

with col_chart:
    st.subheader("Risk Distribution")
    counts = scores.risk.value_counts().reindex(["HIGH", "MEDIUM", "LOW"]).fillna(0)
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=[RISK_COLORS[r] for r in counts.index],
        text=counts.values, texttemplate="%{text:,}", textposition="outside",
    ))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      yaxis=dict(gridcolor="rgba(128,128,128,0.15)"), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
