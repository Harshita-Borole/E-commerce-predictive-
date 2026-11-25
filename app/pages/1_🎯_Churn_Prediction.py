from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go
import streamlit as st

from common import RISK_COLORS, inject_base_style, load_model_output, load_report

st.set_page_config(page_title="Churn Prediction", page_icon="🎯", layout="wide")
inject_base_style()

st.title("🎯 Customer Churn Prediction")
st.caption("XGBoost Classifier · trained on cutoff-date target engineering (see docs/ml_reports/customer_churn.json)")

report = load_report("customer_churn")
scores = load_model_output("customer_churn_scores")

m = report.get("metrics", {})
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers Scored", f"{len(scores):,}")
c2.metric("ROC-AUC", f"{m.get('roc_auc', 0):.2f}", help="0.50 = random guessing. Reported honestly, not inflated.")
c3.metric("Top-Decile Lift", f"{m.get('top_decile_lift', 0):.2f}x",
         help="The riskiest 10% (by this model's ranking) actually return at ~2x the base rate. This is the real, usable signal.")
c4.metric("HIGH Risk Customers", f"{(scores.risk == 'HIGH').sum():,}")

with st.expander("ℹ️ Why ROC-AUC is only 0.55 — read before trusting this at face value"):
    st.markdown(report.get("honest_finding", "").replace("\n", " "))

st.divider()

fc1, fc2 = st.columns([2, 1])
with fc1:
    search = st.text_input("Search Customer ID", placeholder="e.g. c27b3b3188…")
with fc2:
    risk_filter = st.multiselect("Risk level", ["HIGH", "MEDIUM", "LOW"],
                                 default=["HIGH", "MEDIUM", "LOW"])

view = scores[scores.risk.isin(risk_filter)]
if search:
    view = view[view.customer_id.str.contains(search, case=False, na=False)]

col_table, col_chart = st.columns([3, 1])

with col_table:
    st.subheader(f"Ranked Risk List  ({len(view):,} customers)")
    display = view[["customer_id", "churn_probability", "risk"]].copy()
    display["churn_probability"] = (display["churn_probability"] * 100).round(1)
    st.dataframe(
        display.rename(columns={"customer_id": "Customer ID",
                                "churn_probability": "Churn Probability (%)",
                                "risk": "Risk"}),
        use_container_width=True, height=520, hide_index=True,
        column_config={
            "Churn Probability (%)": st.column_config.ProgressColumn(
                "Churn Probability (%)", min_value=0, max_value=100, format="%.1f%%"),
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

    st.caption(
        "Risk bands are quantile-based (top 10% / next 30% / rest), not fixed "
        "probability cutoffs — the predicted range is too compressed for "
        "absolute thresholds to mean much (see the ROC-AUC note above)."
    )
