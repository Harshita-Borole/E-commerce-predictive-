from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

from ml.common import MODELS_DIR, RANDOM_STATE, get_logger, load, print_metric_block, save_report

warnings.filterwarnings("ignore")
log = get_logger("ml.forecast")

RELIABLE_FROM, RELIABLE_TO = "2017-01-01", "2018-08-19"
LAGS = [1, 2, 4]
ROLLING_WINDOWS = [4]
TEST_WEEKS = 12
FORECAST_WEEKS_AHEAD = 4


def _delivered_items_in_window() -> pd.DataFrame:
    orders = load("orders")
    items = load("order_items")
    delivered_ids = orders.loc[orders.order_status == "delivered",
                               ["order_id", "order_purchase_timestamp"]]
    delivered_ids["order_purchase_timestamp"] = pd.to_datetime(delivered_ids.order_purchase_timestamp)
    d = items.merge(delivered_ids, on="order_id", how="inner")
    return d[(d.order_purchase_timestamp >= RELIABLE_FROM) & (d.order_purchase_timestamp <= RELIABLE_TO)]


def compare_grains() -> dict:
    d = _delivered_items_in_window()
    counts = {}
    for label, freq in [("day", "D"), ("week", "W-MON"), ("month", "MS")]:
        s = d.set_index("order_purchase_timestamp").resample(freq).item_revenue.sum()
        counts[label] = {"points": int(len(s)), "mean_revenue": float(s.mean()),
                         "cv": float(s.std() / s.mean())}
    return counts


def build_weekly_series() -> pd.DataFrame:
    d = _delivered_items_in_window()
    weekly = (d.set_index("order_purchase_timestamp")
                .resample("W-MON", label="left", closed="left")
                .agg(revenue=("item_revenue", "sum"), orders=("order_id", "nunique"))
                .reset_index()
                .rename(columns={"order_purchase_timestamp": "week_start"}))
    return weekly


def add_time_features(weekly: pd.DataFrame) -> pd.DataFrame:
    df = weekly.copy()
    df["year"] = df.week_start.dt.year
    df["week_of_year"] = df.week_start.dt.isocalendar().week.astype(int)
    df["month"] = df.week_start.dt.month
    df["quarter"] = df.week_start.dt.quarter
    df["time_index"] = np.arange(len(df))
    df["is_black_friday_week"] = ((df.month == 11) & (df.week_start.dt.day <= 30)
                                  & (df.week_start.dt.day >= 20)).astype(int)

    for lag in LAGS:
        df[f"revenue_lag_{lag}"] = df.revenue.shift(lag)
    for w in ROLLING_WINDOWS:
        df[f"revenue_rollmean_{w}"] = df.revenue.shift(1).rolling(w).mean()
        df[f"revenue_rollstd_{w}"] = df.revenue.shift(1).rolling(w).std()

    return df


def make_model() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=80,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=4,
        reg_lambda=4.0,
        random_state=RANDOM_STATE,
    )


def walk_forward_eval(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    n = len(df)
    for i in range(n - TEST_WEEKS, n):
        train_df = df.iloc[:i]
        X_train, y_train = train_df[feature_cols], train_df["revenue"]
        X_next = df.iloc[[i]][feature_cols]
        y_actual = df["revenue"].iloc[i]

        model = make_model()
        model.fit(X_train, y_train)
        pred = float(model.predict(X_next)[0])
        naive = float(df["revenue"].iloc[i - 1])

        rows.append({
            "week": df.week_start.iloc[i].date(),
            "actual": y_actual, "predicted": pred, "naive": naive,
            "abs_err_model": abs(pred - y_actual),
            "abs_err_naive": abs(naive - y_actual),
        })
    return pd.DataFrame(rows)


def forecast_forward(df: pd.DataFrame, feature_cols: list[str], weeks_ahead: int) -> pd.DataFrame:
    model = make_model()
    model.fit(df[feature_cols], df["revenue"])

    history = df["revenue"].tolist()
    last_week = df["week_start"].iloc[-1]
    rows = []

    for step in range(1, weeks_ahead + 1):
        wk = last_week + pd.Timedelta(weeks=step)
        feat = {
            "year": wk.year,
            "week_of_year": int(wk.isocalendar().week),
            "month": wk.month,
            "quarter": wk.quarter,
            "time_index": len(df) + step - 1,
            "is_black_friday_week": int(wk.month == 11 and 20 <= wk.day <= 30),
        }
        for lag in LAGS:
            feat[f"revenue_lag_{lag}"] = history[-lag]
        for w in ROLLING_WINDOWS:
            recent = history[-w:]
            feat[f"revenue_rollmean_{w}"] = float(np.mean(recent))
            feat[f"revenue_rollstd_{w}"] = float(np.std(recent, ddof=1)) if len(recent) > 1 else 0.0

        X_next = pd.DataFrame([feat])[feature_cols]
        pred = float(model.predict(X_next)[0])
        history.append(pred)
        rows.append({"week_start": wk, "forecast_revenue": pred})

    return pd.DataFrame(rows)


def train() -> dict:
    grains = compare_grains()
    print_metric_block("GRAIN COMPARISON (why weekly)", {
        f"{g}: points / mean revenue / CV": f"{v['points']} / R${v['mean_revenue']:,.0f} / {v['cv']:.2f}"
        for g, v in grains.items()
    })
    print("CV = coefficient of variation (std/mean) — lower means less noise per point.")
    print("month: too few points (20) to learn seasonality from.")
    print("day: enough points, but weekday/weekend swings dominate each single point.")
    print("week: the grain where both problems are smallest at once. -> used below.\n")

    weekly = build_weekly_series()
    log.info("weekly observations in reliable window: %d", len(weekly))

    df = add_time_features(weekly).dropna().reset_index(drop=True)
    log.info("usable rows after lag warm-up: %d", len(df))

    feature_cols = [c for c in df.columns if c not in ("week_start", "revenue", "orders")]
    X, y = df[feature_cols], df["revenue"]

    wf = walk_forward_eval(df, feature_cols)

    metrics = {
        "total_weeks": int(len(weekly)),
        "usable_weeks_after_lags": int(len(df)),
        "walk_forward_weeks": int(len(wf)),
        "mae": float(wf.abs_err_model.mean()),
        "rmse": float(root_mean_squared_error(wf.actual, wf.predicted)),
        "mape": float((wf.abs_err_model / wf.actual).mean()),
        "naive_baseline_mae": float(wf.abs_err_naive.mean()),
        "naive_baseline_rmse": float(root_mean_squared_error(wf.actual, wf.naive)),
        "naive_baseline_mape": float((wf.abs_err_naive / wf.actual).mean()),
        "beats_naive_baseline": bool(wf.abs_err_model.mean() < wf.abs_err_naive.mean()),
        "weeks_model_beat_naive": int((wf.abs_err_model < wf.abs_err_naive).sum()),
    }
    print_metric_block("SALES FORECAST — XGBoost Regressor (weekly, walk-forward)", metrics)

    print(f"\n{'week':<12}{'actual':>14}{'predicted':>14}{'naive':>14}{'error %':>10}")
    for _, r in wf.iterrows():
        print(f"{str(r.week):<12}{r.actual:>14,.0f}{r.predicted:>14,.0f}{r.naive:>14,.0f}"
              f"{100*(r.predicted - r.actual)/r.actual:>9.1f}%")

    model = make_model()
    model.fit(X, y)
    importance = (pd.Series(model.feature_importances_, index=X.columns)
                    .sort_values(ascending=False))
    print("\nTop 10 features by importance (final model, fit on all data):")
    print(importance.head(10).to_string())

    model.save_model(MODELS_DIR / "sales_forecast_xgb.json")
    df.to_csv(MODELS_DIR / "sales_forecast_features.csv", index=False)
    wf.to_csv(MODELS_DIR / "sales_forecast_walk_forward.csv", index=False)

    fwd = forecast_forward(df, feature_cols, FORECAST_WEEKS_AHEAD)
    fwd.to_csv(MODELS_DIR / "sales_forecast_forward.csv", index=False)

    next_period_total = float(fwd.forecast_revenue.sum())
    prior_period_total = float(df["revenue"].tail(FORECAST_WEEKS_AHEAD).sum())
    growth_pct = 100 * (next_period_total - prior_period_total) / prior_period_total

    print(f"\n{'-'*70}\nFORWARD FORECAST — next {FORECAST_WEEKS_AHEAD} weeks "
          f"({fwd.week_start.iloc[0].date()} to {fwd.week_start.iloc[-1].date()})\n{'-'*70}")
    for _, r in fwd.iterrows():
        print(f"  {r.week_start.date()}   R$ {r.forecast_revenue:>12,.0f}")
    print(f"\nNext-{FORECAST_WEEKS_AHEAD}-Week Forecast:  R$ {next_period_total:,.0f}")
    print(f"Prior {FORECAST_WEEKS_AHEAD} Weeks (actual):  R$ {prior_period_total:,.0f}")
    print(f"Expected Growth:          {growth_pct:+.1f}%")
    print(f"\nCaveat: walk-forward MAPE is {metrics['mape']*100:.0f}% and the model "
          f"does {'beat' if metrics['beats_naive_baseline'] else 'NOT beat'} a naive "
          "baseline one week ahead — treat this forward number as indicative, and "
          "widening further with each of the 4 recursive steps, not a precise target.")

    report = {
        "model": "XGBoost Regressor",
        "target": "weekly revenue (delivered orders, price + freight)",
        "reliable_window": f"{RELIABLE_FROM} to {RELIABLE_TO}",
        "grain_comparison": grains,
        "eval_method": "walk-forward, one week ahead, retrained each step",
        "metrics": metrics,
        "top_features": importance.head(10).round(4).to_dict(),
        "forward_forecast": {
            "weeks_ahead": FORECAST_WEEKS_AHEAD,
            "period": f"{fwd.week_start.iloc[0].date()} to {fwd.week_start.iloc[-1].date()}",
            "next_period_forecast": round(next_period_total, 2),
            "prior_period_actual": round(prior_period_total, 2),
            "expected_growth_pct": round(growth_pct, 2),
            "weekly_breakdown": fwd.assign(
                week_start=fwd.week_start.astype(str)).round(2).to_dict("records"),
        },
        "honest_finding": (
            "Not enough data for XGBoost to reliably beat a naive last-week "
            "forecast at this grain (~70-80 weekly points). Reported as-is "
            "rather than tuned until it wins one lucky split. The forward "
            "forecast above should be read the same way — indicative, not precise."
            if not metrics["beats_naive_baseline"] else
            "XGBoost beats the naive baseline under walk-forward evaluation."
        ),
    }
    save_report("sales_forecast", report)
    return report


if __name__ == "__main__":
    train()
