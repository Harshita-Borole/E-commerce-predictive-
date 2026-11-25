from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import CLUSTER_PALETTE, inject_base_style, load_model_output, load_processed, load_report

st.set_page_config(page_title="Segmentation", page_icon="👥", layout="wide")
inject_base_style()

st.title("👥 Customer Segmentation")
st.caption("K-Means, k=4 chosen via elbow + silhouette · classic RFM vs. extended feature set both evaluated "
          "(see docs/ml_reports/segmentation.json)")

report = load_report("segmentation")
scores = load_model_output("segmentation_scores")
rfm = load_processed("customer_rfm")[["customer_unique_id", "recency_days", "monetary"]]

data = scores.merge(rfm, on="customer_unique_id", how="left").dropna(subset=["recency_days", "monetary"])

m = report.get("metrics", {})
c1, c2, c3 = st.columns(3)
c1.metric("Customers Segmented", f"{len(data):,}")
c2.metric("Clusters (k)", m.get("k_selected", "—"))
c3.metric("Silhouette Score", f"{m.get('silhouette_selected', 0):.2f}")

st.divider()

st.subheader("Recency vs. Monetary, colored by cluster")

SAMPLE_N = 8000
plot_data = data.sample(n=min(SAMPLE_N, len(data)), random_state=42)

labels = sorted(plot_data.cluster_label.unique())
color_map = {lbl: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] for i, lbl in enumerate(labels)}

fig = go.Figure()
for lbl in labels:
    sub = plot_data[plot_data.cluster_label == lbl]
    fig.add_trace(go.Scatter(
        x=sub.recency_days, y=sub.monetary, mode="markers", name=lbl,
        marker=dict(size=5, color=color_map[lbl], opacity=0.55, line=dict(width=0)),
        hovertemplate="Recency: %{x:.0f}d<br>Monetary: R$ %{y:,.0f}<extra>" + lbl + "</extra>",
    ))
fig.update_layout(
    height=500, margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="Recency (days since last purchase)", yaxis_title="Monetary (R$, log scale)",
    yaxis_type="log", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    yaxis=dict(gridcolor="rgba(128,128,128,0.15)"), xaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
    legend=dict(orientation="h", y=1.1, x=0),
)
st.plotly_chart(fig, use_container_width=True)
st.caption(f"Showing a random sample of {len(plot_data):,} of {len(data):,} customers for chart responsiveness. "
          "Monetary axis is log-scaled — a few high-value customers would otherwise compress everyone else "
          "into the bottom of the chart.")

st.divider()

st.subheader("What each cluster means")
profile = report.get("cluster_profile", [])
cols = st.columns(len(profile)) if profile else []
for col, row in zip(cols, profile):
    with col:
        lbl = row.get("label", "—")
        swatch = color_map.get(lbl, "#898781")
        st.markdown(f"""
        <div style="border:1px solid rgba(128,128,128,0.25);border-radius:10px;padding:14px;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
          <span style="width:12px;height:12px;border-radius:3px;background:{swatch};display:inline-block"></span>
          <b style="font-size:0.95rem">{lbl}</b>
        </div>
        <span style="font-size:1.6rem;font-weight:700">{row.get('pct_of_base', 0):.1f}%</span>
        <span style="color:#898781;font-size:0.85rem"> of customers</span>
        <hr style="margin:8px 0;opacity:0.2">
        <table style="font-size:0.82rem;width:100%">
          <tr><td style="color:#898781">Avg recency</td><td style="text-align:right">{row.get('avg_recency_days',0):.0f}d</td></tr>
          <tr><td style="color:#898781">Avg value</td><td style="text-align:right">R$ {row.get('avg_monetary',0):,.0f}</td></tr>
          <tr><td style="color:#898781">Avg rating</td><td style="text-align:right">{row.get('avg_review_score',0):.2f}★</td></tr>
          <tr><td style="color:#898781">Late rate</td><td style="text-align:right">{row.get('avg_late_rate',0)*100:.0f}%</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

with st.expander("ℹ️ Classic RFM vs. extended feature set — what the comparison found"):
    st.markdown(report.get("comparison", ""))
