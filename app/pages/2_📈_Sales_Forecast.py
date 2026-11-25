from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import COLORS, brl, inject_base_style, load_model_output, load_report

st.set_page_config(page_title="Sales Forecast", page_icon="📈", layout="wide")
inject_base_style()

st.title("📈 Sales Forecast")
st.caption("XGBoost Regressor · weekly grain · recursive forward forecast (see docs/ml_reports/sales_forecast.json)")

report = load_report("sales_forecast")
fwd_meta = report.get("forward_forecast", {})
m = report.get("metrics", {})

hist = load_model_output("sales_forecast_features")
fwd = load_model_output("sales_forecast_forward")
hist["week_start"] = pd.to_datetime(hist.week_start)
fwd["week_start"] = pd.to_datetime(fwd.week_start)

next_total = fwd_meta.get("next_period_forecast", 0)
prior_total = fwd_meta.get("prior_period_actual", 0)
growth = fwd_meta.get("expected_growth_pct", 0)

h1, h2, h3 = st.columns([2, 2, 3])
h1.metric("Next 4-Week Forecast", brl(next_total))
h2.metric("Expected Growth", f"{growth:+.1f}%",
         delta=f"{growth:+.1f}%", delta_color="normal")
with h3:
    st.markdown(f"""
    <div style="padding-top:6px">
    <span style="font-size:0.8rem;color:{COLORS['muted']}">MAE</span>
    &nbsp;<b>R$ {m.get('mae', 0):,.0f}</b>&nbsp;&nbsp;·&nbsp;&nbsp;
    <span style="font-size:0.8rem;color:{COLORS['muted']}">RMSE</span>
    &nbsp;<b>R$ {m.get('rmse', 0):,.0f}</b>&nbsp;&nbsp;·&nbsp;&nbsp;
    <span style="font-size:0.8rem;color:{COLORS['muted']}">MAPE</span>
    &nbsp;<b>{m.get('mape', 0)*100:.1f}%</b>
    </div>
    """, unsafe_allow_html=True)

if not m.get("beats_naive_baseline", True):
    st.warning(
        f"⚠️ This model does **not** beat a naive \"same as last week\" baseline "
        f"under walk-forward evaluation (model MAE R$ {m.get('mae',0):,.0f} vs. "
        f"naive R$ {m.get('naive_baseline_mae',0):,.0f}). The forecast above is "
        "genuine — it's produced by the trained model on real future weeks — "
        "but should be read as **indicative, not precise**, and uncertainty "
        "compounds further with each of the 4 recursive steps.",
        icon="⚠️",
    )

st.divider()

st.subheader("Weekly Revenue — History + Forecast")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hist.week_start, y=hist.revenue, mode="lines+markers", name="Actual",
    line=dict(color=COLORS["primary"], width=2.5), marker=dict(size=5),
    hovertemplate="%{x|%d %b %Y}<br>R$ %{y:,.0f}<extra>Actual</extra>",
))
bridge_x = [hist.week_start.iloc[-1]] + fwd.week_start.tolist()
bridge_y = [hist.revenue.iloc[-1]] + fwd.forecast_revenue.tolist()
fig.add_trace(go.Scatter(
    x=bridge_x, y=bridge_y, mode="lines+markers", name="Forecast",
    line=dict(color=COLORS["secondary"], width=2.5, dash="dash"), marker=dict(size=6, symbol="diamond"),
    hovertemplate="%{x|%d %b %Y}<br>R$ %{y:,.0f}<extra>Forecast</extra>",
))
fig.update_layout(
    height=420, margin=dict(l=10, r=10, t=10, b=10),
    yaxis_title="Revenue (R$)", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    yaxis=dict(gridcolor="rgba(128,128,128,0.15)"), legend=dict(orientation="h", y=1.08, x=0),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

col_a, col_b = st.columns([1, 1])
with col_a:
    st.subheader("Weekly Forecast Breakdown")
    disp = fwd.copy()
    disp["forecast_revenue"] = disp.forecast_revenue.round(0)
    st.dataframe(
        disp.rename(columns={"week_start": "Week", "forecast_revenue": "Forecast Revenue (R$)"}),
        use_container_width=True, hide_index=True,
    )
with col_b:
    st.subheader("Why weekly grain")
    grains = report.get("grain_comparison", {})
    if grains:
        gdf = pd.DataFrame(grains).T
        gdf.index.name = "Grain"
        st.dataframe(
            gdf.rename(columns={"points": "Points", "mean_revenue": "Mean Revenue",
                                "cv": "Noise (CV)"}).round(2),
            use_container_width=True,
        )
        st.caption(
            "Month has the least noise per point but only 20 points — too few to "
            "learn seasonality from. Day has enough points but weekday/weekend "
            "swings dominate each one. Week is where both problems are smallest "
            "at once."
        )
