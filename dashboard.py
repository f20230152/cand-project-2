from __future__ import annotations

import itertools
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from robustness_engine import SurvivalConfig, run_survival_framework
from research_engine import (
    ResearchConfig,
    build_strategy_library,
    compute_metrics,
    detect_regimes,
    evaluate_strategy_library,
    generate_feature_library,
    load_price_data,
    run_research,
)

TRADING_DAYS = 252
OUTPUT_DIR = Path("outputs")

KPI_EXPLAINERS = {
    "Sharpe": (
        "How much return you earned per unit of total risk. Higher is generally better.",
        "Annualized mean return divided by annualized volatility.",
    ),
    "Sortino": (
        "Like Sharpe, but only penalizes downside volatility.",
        "Annualized mean return divided by downside deviation.",
    ),
    "Calmar": (
        "Return earned relative to worst peak-to-trough loss.",
        "Annual return divided by absolute max drawdown.",
    ),
    "Ann Return": (
        "Estimated average yearly growth rate.",
        "Geometric annualized return from daily strategy returns.",
    ),
    "Max Drawdown": (
        "Worst percentage fall from a previous peak in equity curve.",
        "Minimum of cumulative return / running peak - 1.",
    ),
    "Ann Vol": (
        "Typical annualized ups-and-downs in returns.",
        "Daily std dev multiplied by sqrt(252).",
    ),
    "Hit Rate": (
        "How often the strategy has positive daily returns.",
        "Fraction of days where daily PnL > 0.",
    ),
    "Turnover (daily)": (
        "How much position changes day to day. Higher means more trading costs.",
        "Average absolute change in signal each day.",
    ),
    "Avg Hold (days)": (
        "Average length of a non-zero position before switching/closing.",
        "Average run length of consecutive active signal states.",
    ),
    "Profit Factor": (
        "Total gains divided by total losses. Above 1 means gains exceed losses.",
        "Sum of positive returns divided by absolute sum of negative returns.",
    ),
    "VaR 95% (daily)": (
        "A bad-day threshold: 5% of days are expected to be worse than this.",
        "5th percentile of daily return distribution.",
    ),
    "CVaR 95% (daily)": (
        "Average loss on the worst 5% days.",
        "Mean return conditional on returns <= VaR 95%.",
    ),
    "Tail Ratio": (
        "Compares upside tail to downside tail. Higher means upside tail is stronger.",
        "95th percentile gain divided by absolute 5th percentile loss.",
    ),
    "Long Exposure": (
        "Fraction of time strategy is net long.",
        "Share of days where signal > 0.",
    ),
    "Short Exposure": (
        "Fraction of time strategy is net short.",
        "Share of days where signal < 0.",
    ),
    "Trades (Total)": (
        "How many times the strategy changed position across the full dataset.",
        "Count of days where absolute signal change is non-zero.",
    ),
    "Trades/Year": (
        "Average number of position changes per year.",
        "Total trade events divided by number of calendar years in sample.",
    ),
    "Trade Day %": (
        "Percent of days when a trade happens.",
        "Share of dates where absolute signal change is non-zero.",
    ),
    "Years With Trades": (
        "How many years had at least one trade.",
        "Count of calendar years with trade events > 0.",
    ),
    "Activity-Adjusted Sharpe": (
        "Sharpe scaled by how active the strategy is, so sparse traders are penalized.",
        "Sharpe multiplied by sqrt((total trades + 1) / (median trades + 1)).",
    ),
}

FAMILY_EXPLAINERS = {
    "trend_following": (
        "Follows price direction. If market is rising, it tends to stay long; if falling, short.",
        "Signals from momentum, moving average spreads, slopes, and breakout logic.",
    ),
    "mean_reversion": (
        "Assumes short-term overreaction snaps back toward normal levels.",
        "Signals from z-scores, RSI thresholds, and short-horizon reversal rules.",
    ),
    "volatility_regime": (
        "Changes behavior depending on whether market volatility is calm or stressed.",
        "Conditional signal activation using volatility percentile and regime thresholds.",
    ),
    "time_calendar": (
        "Trades seasonal/time effects such as day-of-week or month patterns.",
        "Calendar dummy-based signals and turn-of-month/summer proxies.",
    ),
    "stat_structure": (
        "Uses statistical fingerprints like autocorrelation or randomness persistence.",
        "Signals from autocorr, variance ratio, entropy, and Hurst-based switching.",
    ),
    "risk_managed_alpha": (
        "Same directional view, but smarter position sizing to control risk.",
        "Vol-targeting, drawdown scaling, and Kelly-proxy allocation overlays.",
    ),
    "hybrid": (
        "Combines multiple ideas, e.g., trend only in low vol or MR only in low trend.",
        "Rule-combination strategies with context filters.",
    ),
    "ml_like": (
        "Combines many features into a score and maps score to trade direction.",
        "Linear composite signal with thresholded decision boundaries.",
    ),
    "time_series": (
        "Classical time-series rules that use persistence, trend strength, and adaptive momentum.",
        "Signals from multi-horizon momentum agreement, EWMA edge, and drawdown/volatility gates.",
    ),
    "event_flow": (
        "Event-window logic around month/quarter timing and weekday flow effects.",
        "Calendar-event conditioned directional rules, including quarter-turn and turn-of-month structures.",
    ),
    "microstructure_proxy": (
        "Short-horizon behavior proxies for bounce, reversal, and post-shock reactions.",
        "Lag-return pattern rules and volatility-conditioned short-term state transitions.",
    ),
    "time_series_adaptive": (
        "Adaptive time-series models that blend fast and slow trend persistence.",
        "Dual-horizon momentum agreement plus EWMA risk-adjusted edge signals.",
    ),
    "risk_overlay": (
        "Risk-budget overlays that dynamically scale exposure by volatility and drawdown.",
        "Vol-target and drawdown cutback overlays applied on directional base signals.",
    ),
}


def pct(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x:.2%}"


def fmt(x: float, n: int = 3) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x:.{n}f}"


def tutor_popover(title: str, simple: str, pro: str, enabled: bool) -> None:
    if not enabled:
        return
    with st.popover("(i)", help=title):
        st.markdown(f"**{title}**")
        st.markdown(f"Simple: {simple}")
        st.markdown(f"Technical: {pro}")


def section_header(title: str, simple: str, pro: str, show_tutor: bool) -> None:
    c1, c2 = st.columns([18, 1])
    with c1:
        st.subheader(title)
    with c2:
        tutor_popover(title, simple, pro, show_tutor)


def drawdown_series(returns: pd.Series) -> pd.Series:
    wealth = (1 + returns.fillna(0.0)).cumprod()
    return wealth / wealth.cummax() - 1


def rolling_sharpe(returns: pd.Series, window: int = 126) -> pd.Series:
    return np.sqrt(TRADING_DAYS) * returns.rolling(window).mean() / (returns.rolling(window).std() + 1e-12)


def monthly_return_table(returns: pd.Series) -> pd.DataFrame:
    monthly = (1 + returns.dropna()).resample("M").prod() - 1
    out = monthly.to_frame("ret")
    out["year"] = out.index.year
    out["month"] = out.index.month
    pivot = out.pivot(index="year", columns="month", values="ret").sort_index()
    pivot.columns = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    return pivot


def extended_kpis(returns: pd.Series, signal: pd.Series) -> Dict[str, float]:
    metrics = compute_metrics(returns, signal)
    r = returns.dropna()
    s = signal.reindex(r.index).fillna(0.0)

    var_95 = float(r.quantile(0.05)) if not r.empty else np.nan
    cvar_95 = float(r[r <= var_95].mean()) if not r.empty else np.nan
    gross_profit = float(r[r > 0].sum()) if not r.empty else np.nan
    gross_loss = float(-r[r < 0].sum()) if not r.empty else np.nan
    profit_factor = float(gross_profit / (gross_loss + 1e-12)) if not r.empty else np.nan
    tail_ratio = float(r.quantile(0.95) / (abs(r.quantile(0.05)) + 1e-12)) if not r.empty else np.nan
    long_exposure = float((s > 0).mean())
    short_exposure = float((s < 0).mean())
    flat_exposure = float((s == 0).mean())

    metrics.update(
        {
            "var_95": var_95,
            "cvar_95": cvar_95,
            "profit_factor": profit_factor,
            "tail_ratio": tail_ratio,
            "long_exposure": long_exposure,
            "short_exposure": short_exposure,
            "flat_exposure": flat_exposure,
        }
    )
    return metrics


def trade_event_series(signal: pd.Series) -> pd.Series:
    s = signal.fillna(0.0)
    return (s.diff().abs().fillna(0.0) > 1e-12).astype(int)


def universe_trade_stats(signal_df: pd.DataFrame) -> pd.DataFrame:
    s = signal_df.fillna(0.0)
    changes = s.diff().abs().fillna(0.0)
    events = (changes > 1e-12).astype(int)
    n_years = max(int(s.index.year.nunique()), 1)

    yearly_events = events.groupby(s.index.year).sum()
    years_with_trades = yearly_events.gt(0).sum(axis=0)

    out = pd.DataFrame(
        {
            "trade_events_total": events.sum(axis=0).astype(float),
            "trade_day_pct": events.mean(axis=0),
            "turnover_units_total": changes.sum(axis=0),
            "active_day_pct": s.ne(0.0).mean(axis=0),
            "years_with_trades": years_with_trades.astype(float),
        }
    )
    out["trades_per_year"] = out["trade_events_total"] / n_years
    median_trades = float(out["trade_events_total"].median()) if not out.empty else 0.0
    out["trade_weight"] = np.sqrt((out["trade_events_total"] + 1.0) / (median_trades + 1.0))
    out.index.name = "strategy"
    return out


def yearly_breakdown(returns: pd.Series, signal: pd.Series | None = None) -> pd.DataFrame:
    r = returns.dropna()
    if r.empty:
        return pd.DataFrame()

    trade_events = None
    turnover_units = None
    days_in_market = None
    if signal is not None:
        s = signal.reindex(r.index).fillna(0.0)
        trade_events = trade_event_series(s)
        turnover_units = s.diff().abs().fillna(0.0)
        days_in_market = s.ne(0.0).astype(float)

    rows = []
    for year, grp in r.groupby(r.index.year):
        m = compute_metrics(grp)
        row = {
            "year": int(year),
            "sharpe": m["sharpe"],
            "ann_return": m["ann_return"],
            "ann_vol": m["ann_vol"],
            "max_drawdown": m["max_drawdown"],
            "hit_rate": m["hit_rate"],
        }
        if trade_events is not None and turnover_units is not None and days_in_market is not None:
            y_idx = grp.index
            row["trades"] = int(trade_events.loc[y_idx].sum())
            row["turnover_units"] = float(turnover_units.loc[y_idx].sum())
            row["days_in_market_pct"] = float(days_in_market.loc[y_idx].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("year")


def strategy_one_liner(strategy: str) -> str:
    s = strategy
    if s.startswith("trend_mom_"):
        w = s.split("_")[-1]
        return f"Trend momentum ({w}d): long when the last {w} days were up, short when they were down."
    if s.startswith("trend_slope_"):
        w = s.split("_")[-1]
        return f"Trend slope ({w}d): uses rolling log-price slope; positive slope goes long, negative goes short."
    if s.startswith("trend_donchian_"):
        w = s.split("_")[-1]
        return f"Donchian breakout ({w}d): long on breakout above prior {w}-day high, short below prior {w}-day low."
    if s.startswith("trend_ma_cross_"):
        _, _, _, short, long = s.split("_")
        return f"MA crossover ({short}/{long}): long when short MA is above long MA, short when below."
    if s.startswith("mr_z_"):
        _, _, w, thr = s.split("_")
        return f"Z-score mean reversion ({w}d, thr {thr}): fades extremes and bets price moves back toward average."
    if s.startswith("mr_rsi_"):
        _, _, w, overbought, oversold = s.split("_")
        return f"RSI mean reversion ({w}d): short above RSI {overbought}, long below RSI {oversold}."
    if s.startswith("mr_st_rev_"):
        w = s.split("_")[-1]
        return f"Short-term reversal ({w}d): takes the opposite side of recent {w}-day move."
    if s.startswith("vol_lowtrend_"):
        _, _, base_w, vol_w, q = s.split("_")
        return f"Volatility filter: trend({base_w}d) is traded only when vol({vol_w}d) is in the calmer bottom {float(q):.0%} bucket."
    if s.startswith("vol_highmr_"):
        _, _, base_w, vol_w, q = s.split("_")
        return f"High-volatility reversal: flips trend({base_w}d) when vol({vol_w}d) is in the stressed top {float(q):.0%} bucket."
    if s.startswith("time_dow_"):
        _, _, d, side = s.split("_")
        day_map = {"0": "Monday", "1": "Tuesday", "2": "Wednesday", "3": "Thursday", "4": "Friday"}
        pos = "long" if side == "1" else "short"
        return f"Day-of-week effect: {pos} only on {day_map.get(d, d)}, flat otherwise."
    if s.startswith("time_month_"):
        _, _, m, side = s.split("_")
        pos = "long" if side == "1" else "short"
        return f"Month effect: {pos} only during month {m}, flat otherwise."
    if s.startswith("time_turnmonth_"):
        side = s.split("_")[-1]
        pos = "long" if side == "1" else "short"
        return f"Turn-of-month effect: {pos} near month-end/start window, flat outside that window."
    if s.startswith("time_summer_"):
        side = s.split("_")[-1]
        pos = "long" if side == "1" else "short"
        return f"Seasonal effect: {pos} during summer-driving months (May-Aug), flat otherwise."
    if s.startswith("stat_ac_switch_"):
        w = s.split("_")[-1]
        return f"Autocorr switch ({w}d): uses momentum when autocorr is positive, reversal when autocorr is negative."
    if s.startswith("stat_vr_"):
        w = s.split("_")[-1]
        return f"Variance-ratio regime ({w}d): long in trending variance-ratio states, short in reverting states."
    if s.startswith("stat_hurst_"):
        w = s.split("_")[-1]
        return f"Hurst switch ({w}d): follows trend in persistent regimes, fades moves in mean-reverting regimes."
    if s.startswith("stat_entropy_"):
        w = s.split("_")[-1]
        return f"Entropy filter ({w}d): trades trend only when return randomness is relatively low."
    if s.startswith("risk_voltarget_"):
        _, _, w, target = s.split("_")
        return f"Vol-targeted trend ({w}d): trend direction with position size scaled to ~{target}% target annualized volatility."
    if s.startswith("risk_ddscale_"):
        cut = s.split("_")[-1]
        return f"Drawdown scaling ({cut}%): reduces exposure after deep drawdown to control downside risk."
    if s.startswith("risk_kellyproxy_"):
        w = s.split("_")[-1]
        return f"Kelly-proxy sizing ({w}d): size grows when rolling edge is stronger and shrinks when edge weakens."
    if s == "hybrid_trend_lowvol":
        return "Hybrid: trades trend only in low-volatility periods; stands down when volatility is elevated."
    if s == "hybrid_mr_lowtrend":
        return "Hybrid: uses mean reversion only when trend strength is weak, avoiding trend-dominant phases."
    if s == "hybrid_trend_highvol_off":
        return "Hybrid: follows trend but switches off in high-volatility stress regimes."
    if s.startswith("ml_linear_score_"):
        w = s.split("_")[-1]
        return f"Linear multi-factor score ({w}d): combines momentum, z-score, autocorr, and vol state into one directional signal."
    if s.startswith("ml_linear_thr_"):
        _, _, _, w, thr = s.split("_")
        return f"Thresholded linear score ({w}d, thr {thr}): only takes trades when composite score conviction exceeds threshold."
    if s.startswith("ts_dual_mom_"):
        _, _, _, fast, slow = s.split("_")
        return f"Time-series dual momentum ({fast}/{slow}d): trades only when fast and slow trend directions agree."
    if s.startswith("ts_ewm_edge_"):
        w = s.split("_")[-1]
        return f"EWMA edge model ({w}d): position size follows exponentially weighted mean/volatility edge."
    if s.startswith("ts_strength_"):
        _, _, w, thr = s.split("_")
        return f"Trend-strength gate ({w}d, thr {thr}): trades only when normalized trend strength is strong enough."
    if s.startswith("ts_drawdown_vol_gate_"):
        w = s.split("_")[-1]
        return f"Drawdown-volatility gated trend ({w}d): trend trades are muted during deep drawdown or stressed volatility."
    if s.startswith("time_series_adaptive_"):
        return "Adaptive time-series rule from Survival Mode using multi-horizon persistence signals."
    if s.startswith("event_flow_"):
        return "Event-flow rule from Survival Mode using quarter/month timing windows."
    if s.startswith("risk_overlay_"):
        return "Risk-overlay rule from Survival Mode applying volatility/drawdown-aware position scaling."
    if s.startswith("event_"):
        return "Event-flow strategy: calendar/event window logic conditions directional exposure."
    if s.startswith("micro_"):
        return "Microstructure-proxy strategy: short-horizon lag and shock patterns drive tactical positioning."
    if s.startswith("time_based_"):
        return "Time-based rule from Survival Mode: uses calendar structure (month/day/event timing) to set directional exposure."
    if s.startswith("vol_based_"):
        return "Volatility-conditioned rule from Survival Mode: behavior changes when realized volatility enters stress/calm states."
    if s.startswith("regime_"):
        return "Regime-switching rule from Survival Mode: toggles between trend/reversion logic by detected market state."
    if s.startswith("stat_"):
        return "Statistical-state rule from Survival Mode: trades based on distribution shape, z-scores, or shift diagnostics."
    if s.startswith("ml_light_"):
        return "Interpretable ML-light rule from Survival Mode using linear/logistic/tree-vote style predictors."
    return "Rule-based strategy from the research factory; inspect KPIs and charts below for behavior details."


@st.cache_data(show_spinner=False)
def load_output_tables() -> Dict[str, pd.DataFrame]:
    # Rebuild outputs to keep walkforward/robustness tables synced with the live strategy factory.
    run_research(ResearchConfig())

    tables = {
        "strategy_metrics": pd.read_csv(OUTPUT_DIR / "strategy_metrics.csv"),
        "walkforward_grid": pd.read_csv(OUTPUT_DIR / "walkforward_grid.csv"),
        "rebalance_details": pd.read_csv(OUTPUT_DIR / "rebalance_details.csv"),
        "robustness_summary": pd.read_csv(OUTPUT_DIR / "robustness_summary.csv"),
        "noise_injection_test": pd.read_csv(OUTPUT_DIR / "noise_injection_test.csv"),
        "ensemble_metrics": pd.read_csv(OUTPUT_DIR / "ensemble_metrics.csv"),
        "walkforward_portfolio": pd.read_csv(
            OUTPUT_DIR / "walkforward_portfolio.csv", parse_dates=["date"], index_col="date"
        ),
    }
    return tables


@st.cache_data(show_spinner=True)
def build_universe() -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = ResearchConfig()
    close, rets = load_price_data(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        features = generate_feature_library(close)
    pnl_df, signal_df, family_map = build_strategy_library(close, rets, features, cfg)
    family_df = pd.Series(family_map, name="family").to_frame()
    regimes = detect_regimes(features)
    return close, rets, pnl_df, signal_df, family_df, regimes


@st.cache_data(show_spinner=True)
def load_survival_layer(reject_for_missing_logic: bool = True) -> Dict[str, object]:
    cfg = SurvivalConfig(reject_for_missing_logic=reject_for_missing_logic)
    return run_survival_framework(ResearchConfig(), cfg)


def render_survival_mode(show_tutor: bool, reject_for_missing_logic: bool) -> None:
    payload = load_survival_layer(reject_for_missing_logic=reject_for_missing_logic)
    diagnostics = payload["diagnostics"].copy()
    pnl_df = payload["pnl_df"]
    signal_df = payload["signal_df"]
    top_strategy = payload["top_strategy"]
    wf_summary = payload["walkforward_summary"].copy()
    wf_split = payload["walkforward_splits"].copy()
    wf_portfolio = payload["walkforward_portfolio"].fillna(0.0)
    regime_breakdown = payload["regime_breakdown"].copy()
    parameter_surface = payload["parameter_surface"].copy()
    graveyard = payload["graveyard"].copy()
    blending = payload["blending"]

    st.subheader("Quant Robustness & Survival Framework v2")
    profile_label = "Strict profile (economic-logic rejection ON)" if reject_for_missing_logic else "Statistical profile (economic-logic rejection OFF)"
    st.caption(f"Survival-first validation: reject fragile ideas, then rank what survives. Active filter profile: {profile_label}.")

    tabs = st.tabs(
        [
            "Robustness Diagnostics",
            "Walk-Forward Survival",
            "Parameter Surfaces",
            "Feature Blending Lab",
            "Strategy Graveyard",
        ]
    )

    with tabs[0]:
        section_header(
            "Robustness Diagnostics",
            "This panel auto-critiques strategies and highlights fragility.",
            "Hard-filtered diagnostics with automatic red flags, overfit-risk tags, and robustness scoring.",
            show_tutor,
        )
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Strategies", f"{len(diagnostics)}")
        d2.metric("Rejected", f"{int(diagnostics['rejected'].sum())}")
        d3.metric("Median Robustness", fmt(diagnostics["final_robustness_score"].median(), 1))
        d4.metric("High Overfit Risk", f"{int((diagnostics['OVERFIT_RISK'] == 'HIGH').sum())}")
        d5.metric("Median Inactive %", pct(diagnostics["inactive_percent"].median() / 100.0))

        display_cols = [
            "strategy",
            "family",
            "final_robustness_score",
            "OVERFIT_RISK",
            "red_flags",
            "rejection_reason",
            "trades_total",
            "trades_per_year",
            "exposure_percent",
            "inactive_percent",
            "parameter_stability_score",
            "parameter_surface_gradient",
            "plateau_width",
            "in_sample_sharpe",
            "out_sample_sharpe",
            "is_oos_sharpe_delta",
            "regime_robustness_score",
            "walk_forward_score",
        ]
        st.dataframe(
            diagnostics[display_cols].sort_values("final_robustness_score", ascending=False).round(4),
            use_container_width=True,
            hide_index=True,
        )

        strategy = st.selectbox(
            "Inspect strategy robustness profile",
            diagnostics["strategy"].tolist(),
            index=max(diagnostics["strategy"].tolist().index(top_strategy), 0) if top_strategy in diagnostics["strategy"].tolist() else 0,
        )
        row = diagnostics.loc[diagnostics["strategy"] == strategy].iloc[0]
        ts = build_strategy_timeseries(strategy, pnl_df, signal_df)
        ts["rolling_sharpe_1y"] = rolling_sharpe(ts["returns"], window=252)
        ts["rolling_drawdown_1y"] = ts["drawdown"].rolling(252).min()
        ts["rolling_exposure_1y"] = ts["signal"].ne(0.0).rolling(252).mean() * 100.0
        ts["rolling_trade_freq_1y"] = (ts["signal"].diff().abs().fillna(0.0) > 1e-12).astype(float).rolling(252).sum()
        ts["rolling_cagr_1y"] = (1.0 + ts["returns"]).rolling(252).apply(
            lambda arr: (np.prod(arr) ** (252.0 / max(len(arr), 1)) - 1.0), raw=True
        )

        st.markdown(
            f"**Red Flags:** `{row['red_flags']}`  |  **Rejection Reason:** `{row['rejection_reason']}`  |  "
            f"**OVERFIT_RISK:** `{row['OVERFIT_RISK']}`"
        )

        c1, c2 = st.columns(2)
        c1.plotly_chart(
            px.line(
                ts.reset_index(),
                x="date",
                y=["rolling_sharpe_1y", "rolling_cagr_1y", "rolling_drawdown_1y"],
                title="Rolling Sharpe / Rolling CAGR / Drawdown",
            ),
            use_container_width=True,
        )
        c2.plotly_chart(
            px.line(
                ts.reset_index(),
                x="date",
                y=["rolling_exposure_1y", "rolling_trade_freq_1y"],
                title="Exposure % and Trade Frequency Over Time",
            ),
            use_container_width=True,
        )

        delta_df = pd.DataFrame(
            {
                "metric": ["Sharpe", "CAGR", "Drawdown"],
                "in_sample": [row["in_sample_sharpe"], row["in_sample_cagr"], row["in_sample_drawdown"]],
                "out_sample": [row["out_sample_sharpe"], row["out_sample_cagr"], row["out_sample_drawdown"]],
            }
        )
        st.plotly_chart(
            px.bar(delta_df.melt(id_vars="metric", var_name="sample", value_name="value"), x="metric", y="value", color="sample", barmode="group", title="In-Sample vs Out-of-Sample Delta"),
            use_container_width=True,
        )

        rb = regime_breakdown[regime_breakdown["strategy"] == strategy].sort_values("ann_sharpe", ascending=False)
        st.markdown("**Regime Performance Breakdown**")
        st.dataframe(rb.round(4), use_container_width=True, hide_index=True)

    with tabs[1]:
        section_header(
            "Walk-Forward Survival Engine",
            "Rolling train/test survival check: does quality persist out-of-sample?",
            "Rolling window train-select-test validation with split-level stability diagnostics.",
            show_tutor,
        )
        if wf_summary.empty:
            st.warning("Walk-forward summary unavailable.")
        else:
            r = wf_summary.iloc[0]
            rebalance_freq_m = np.nan
            if len(wf_split) > 1:
                test_start = pd.to_datetime(wf_split["test_start"], errors="coerce").sort_values()
                delta_days = test_start.diff().dt.days.dropna()
                if not delta_days.empty:
                    rebalance_freq_m = float(delta_days.median() / 30.44)
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("WF Sharpe", fmt(r["walk_forward_sharpe"]))
            c2.metric("WF CAGR", pct(r["walk_forward_cagr"]))
            c3.metric("WF MaxDD", pct(r["walk_forward_drawdown"]))
            c4.metric("Stability Score", fmt(r["stability_score"], 3))
            c5.metric("Consistency Index", fmt(r["consistency_index"], 3))
            c6.metric("Rebalance Freq", "N/A" if pd.isna(rebalance_freq_m) else f"{rebalance_freq_m:.1f}M")

        if not wf_split.empty:
            st.dataframe(wf_split.round(4), use_container_width=True, hide_index=True)
            st.plotly_chart(
                px.line(
                    wf_split,
                    x="test_start",
                    y=["is_sharpe", "oos_sharpe"],
                    markers=True,
                    title="In-Sample vs Out-of-Sample Sharpe Across Splits",
                ),
                use_container_width=True,
            )

        st.plotly_chart(
            px.line(
                ((1 + wf_portfolio).cumprod()).to_frame("walkforward_oos").reset_index(),
                x="date",
                y="walkforward_oos",
                title="Walk-Forward OOS Portfolio",
            ),
            use_container_width=True,
        )

    with tabs[2]:
        section_header(
            "Parameter Surface Visualization",
            "Smooth plateaus are preferred; sharp spikes are overfit warnings.",
            "2D/3D parameter-response surfaces with smoothness and peak-concentration diagnostics.",
            show_tutor,
        )
        if parameter_surface.empty:
            st.info("No parameterized surface records available.")
        else:
            templates = sorted(parameter_surface["parameter_template"].dropna().unique().tolist())
            template = st.selectbox("Parameter template", templates, index=0)
            surf = parameter_surface[parameter_surface["parameter_template"] == template].copy()

            n_params = len([c for c in surf.columns if c.startswith("param_")])
            if n_params >= 2:
                st.plotly_chart(
                    px.scatter_3d(
                        surf,
                        x="param_1",
                        y="param_2",
                        z="sharpe",
                        color="cagr",
                        hover_data=["strategy", "drawdown"],
                        title="3D Surface: Parameter 1 vs Parameter 2 vs Sharpe",
                    ),
                    use_container_width=True,
                )
            else:
                st.plotly_chart(
                    px.line(
                        surf.sort_values("param_1"),
                        x="param_1",
                        y=["sharpe", "cagr", "drawdown"],
                        markers=True,
                        title="2D Surface: Parameter vs Performance",
                    ),
                    use_container_width=True,
                )
            st.dataframe(surf.round(4), use_container_width=True, hide_index=True)

    with tabs[3]:
        section_header(
            "Feature Blending Lab",
            "Blend top robust strategies as features instead of betting on one winner.",
            "OLS/Ridge/Lasso/WeightedVoting blends with correlation pruning guidance.",
            show_tutor,
        )
        selected = blending["selected_features"]
        st.markdown(f"**Selected robust features:** `{', '.join(selected) if selected else 'None'}`")
        if not blending["metrics"].empty:
            st.dataframe(blending["metrics"].round(4), use_container_width=True, hide_index=True)
            st.dataframe(blending["coefficients"].round(4), use_container_width=True, hide_index=True)
            if not blending["correlation"].empty:
                corr = blending["correlation"]
                st.plotly_chart(
                    px.imshow(
                        corr.values,
                        x=corr.columns,
                        y=corr.index,
                        color_continuous_scale="RdBu",
                        zmin=-1,
                        zmax=1,
                        title="Strategy Correlation Matrix",
                    ),
                    use_container_width=True,
                )
            if not blending["suggestions"].empty:
                st.dataframe(blending["suggestions"], use_container_width=True, hide_index=True)
            if not blending["blended_pnl"].empty:
                st.plotly_chart(
                    px.line(
                        ((1 + blending["blended_pnl"]).cumprod()).reset_index(),
                        x="date",
                        y=blending["blended_pnl"].columns.tolist(),
                        title="Blended Model Equity Curves",
                    ),
                    use_container_width=True,
                )
        else:
            st.info("Not enough robust strategies passed filters for blending.")

    with tabs[4]:
        section_header(
            "Strategy Graveyard",
            "Rejected strategies are logged with explicit reasons.",
            "Auto-generated rejection report with hard-filter reasons for auditability.",
            show_tutor,
        )
        st.caption(f"Current rejection profile: `{profile_label}`")
        if graveyard.empty:
            st.success("No strategies rejected by current hard filters.")
        else:
            st.dataframe(
                graveyard[
                    [
                        "strategy",
                        "family",
                        "final_robustness_score",
                        "OVERFIT_RISK",
                        "red_flags",
                        "rejection_reason",
                        "trades_total",
                        "trades_per_year",
                        "exposure_percent",
                        "inactive_percent",
                    ]
                ].sort_values("final_robustness_score", ascending=True),
                use_container_width=True,
                hide_index=True,
            )
            reason_counts = graveyard["rejection_reason"].value_counts().reset_index()
            reason_counts.columns = ["reason", "count"]
            st.plotly_chart(px.bar(reason_counts, x="reason", y="count", title="Rejection Reason Distribution"), use_container_width=True)


def build_strategy_timeseries(strategy: str, pnl_df: pd.DataFrame, signal_df: pd.DataFrame) -> pd.DataFrame:
    ret = pnl_df[strategy].fillna(0.0)
    sig = signal_df[strategy].fillna(0.0)
    out = pd.DataFrame(index=ret.index)
    out["returns"] = ret
    out["signal"] = sig
    out["cum"] = (1 + ret).cumprod()
    out["drawdown"] = drawdown_series(ret)
    out["rolling_sharpe_6m"] = rolling_sharpe(ret, window=126)
    out["rolling_vol_3m"] = ret.rolling(63).std() * np.sqrt(TRADING_DAYS)
    return out


def build_hero_table(frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    valid = frame.dropna(subset=[score_col]).copy()
    if valid.empty:
        return pd.DataFrame()
    idx = valid.groupby("family")[score_col].idxmax()
    cols = [
        "family",
        "strategy",
        "sharpe",
        "activity_adjusted_sharpe",
        "ann_return",
        "max_drawdown",
        "trade_events_total",
        "trades_per_year",
    ]
    out = valid.loc[idx, cols].sort_values(score_col, ascending=False).reset_index(drop=True)
    return out


def combine_strategy_universe(
    base_pnl: pd.DataFrame,
    base_signal: pd.DataFrame,
    base_family_df: pd.DataFrame,
    include_survival_universe: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not include_survival_universe:
        fam = base_family_df.reindex(base_pnl.columns).copy()
        fam["family"] = fam["family"].fillna("unknown")
        return base_pnl.copy(), base_signal.copy(), fam

    payload = load_survival_layer(reject_for_missing_logic=False)
    surv_pnl = payload["pnl_df"].copy()
    surv_signal = payload["signal_df"].copy()
    surv_family = (
        payload["diagnostics"][["strategy", "family"]]
        .dropna(subset=["strategy"])
        .drop_duplicates(subset=["strategy"])
        .set_index("strategy")
    )

    pnl_df = pd.concat([base_pnl, surv_pnl], axis=1)
    pnl_df = pnl_df.loc[:, ~pnl_df.columns.duplicated(keep="first")]
    signal_df = pd.concat([base_signal, surv_signal], axis=1)
    signal_df = signal_df.loc[:, ~signal_df.columns.duplicated(keep="first")]
    signal_df = signal_df.reindex(columns=pnl_df.columns).fillna(0.0)

    family_df = pd.concat([base_family_df, surv_family], axis=0)
    family_df = family_df[~family_df.index.duplicated(keep="first")]
    family_df = family_df.reindex(pnl_df.columns).copy()
    family_df["family"] = family_df["family"].fillna("unknown")
    return pnl_df, signal_df, family_df


def build_metrics_table(
    pnl_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    family_df: pd.DataFrame,
) -> pd.DataFrame:
    family_map = family_df["family"].to_dict()
    metrics_df = evaluate_strategy_library(pnl_df, signal_df, family_map).copy()
    trade_df = universe_trade_stats(signal_df).reset_index()

    for col in [
        "trade_events_total",
        "trade_day_pct",
        "turnover_units_total",
        "active_day_pct",
        "years_with_trades",
        "trades_per_year",
        "trade_weight",
    ]:
        if col in metrics_df.columns:
            metrics_df = metrics_df.drop(columns=col)

    metrics_df = metrics_df.merge(trade_df, on="strategy", how="left")
    metrics_df["activity_adjusted_sharpe"] = metrics_df["sharpe"] * metrics_df["trade_weight"]
    metrics_df["family"] = metrics_df["family"].fillna("unknown")
    return metrics_df


def _fit_ridge_beta(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    p = x.shape[1]
    return np.linalg.pinv(x.T @ x + alpha * np.eye(p)) @ (x.T @ y)


def _fit_gradient_boosting_proxy(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_rounds: int,
    learning_rate: float,
) -> Tuple[float, List[Tuple[int, float]]]:
    base = float(np.nanmean(y_train))
    pred = np.full(len(y_train), base, dtype=float)
    components: List[Tuple[int, float]] = []

    for _ in range(int(max(n_rounds, 1))):
        residual = y_train - pred
        best_j = -1
        best_coef = 0.0
        best_score = 0.0
        for j in range(x_train.shape[1]):
            col = x_train[:, j]
            denom = float(np.dot(col, col) + 1e-12)
            coef = float(np.dot(col, residual) / denom)
            score = abs(coef) * float(np.std(col))
            if score > best_score:
                best_score = score
                best_j = j
                best_coef = coef
        if best_j < 0:
            break
        step = float(learning_rate) * best_coef
        pred = pred + step * x_train[:, best_j]
        components.append((best_j, step))

    return base, components


def _predict_gradient_boosting_proxy(x: np.ndarray, base: float, components: List[Tuple[int, float]]) -> np.ndarray:
    pred = np.full(x.shape[0], base, dtype=float)
    for j, w in components:
        pred = pred + w * x[:, int(j)]
    return pred


def _selection_score(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    return (s - s.mean()) / (s.std() + 1e-12)


def _fit_model_signals_on_slice(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_pred: pd.DataFrame,
    ridge_alpha: float,
    gb_rounds: int,
    gb_learning_rate: float,
) -> Tuple[pd.Series, pd.Series]:
    if x_pred.empty:
        empty = pd.Series(dtype=float, index=x_pred.index)
        return empty, empty

    x_train = x_train.fillna(0.0)
    x_pred = x_pred.fillna(0.0)
    y_train = y_train.reindex(x_train.index).fillna(0.0)
    if x_train.empty:
        zeros = pd.Series(0.0, index=x_pred.index)
        return zeros.copy(), zeros.copy()

    mu = x_train.mean()
    sd = x_train.std().replace(0.0, 1.0)
    x_train_std = ((x_train - mu) / sd).fillna(0.0)
    x_pred_std = ((x_pred - mu) / sd).fillna(0.0)

    x_train_np = x_train_std.to_numpy()
    y_train_np = y_train.to_numpy()
    x_pred_np = x_pred_std.to_numpy()

    beta = _fit_ridge_beta(x_train_np, y_train_np, alpha=float(ridge_alpha))
    ridge_train_score = pd.Series(x_train_np @ beta, index=x_train.index)
    ridge_pred_score = pd.Series(x_pred_np @ beta, index=x_pred.index)

    gb_base, gb_components = _fit_gradient_boosting_proxy(
        x_train_np,
        y_train_np,
        n_rounds=int(gb_rounds),
        learning_rate=float(gb_learning_rate),
    )
    gb_train_score = pd.Series(_predict_gradient_boosting_proxy(x_train_np, gb_base, gb_components), index=x_train.index)
    gb_pred_score = pd.Series(_predict_gradient_boosting_proxy(x_pred_np, gb_base, gb_components), index=x_pred.index)

    ridge_scale = max(float(ridge_train_score.std()), 1e-6)
    gb_scale = max(float(gb_train_score.std()), 1e-6)
    ridge_signal = pd.Series(np.tanh(ridge_pred_score / ridge_scale), index=x_pred.index)
    gb_signal = pd.Series(np.tanh(gb_pred_score / gb_scale), index=x_pred.index)
    return ridge_signal, gb_signal


def _apply_quarterly_oos_rebalance(
    x: pd.DataFrame,
    y: pd.Series,
    split_idx: int,
    base_ridge_signal: pd.Series,
    base_gb_signal: pd.Series,
    ridge_alpha: float,
    gb_rounds: int,
    gb_learning_rate: float,
) -> Tuple[pd.Series, pd.Series]:
    ridge_q = base_ridge_signal.copy()
    gb_q = base_gb_signal.copy()
    if split_idx >= len(x):
        return ridge_q, gb_q

    oos_index = x.index[split_idx:]
    oos_groups = pd.Series(oos_index, index=oos_index).groupby(oos_index.to_period("Q"))
    for _, segment in oos_groups:
        seg_idx = segment.index
        if len(seg_idx) == 0:
            continue
        start_dt = seg_idx.min()
        start_loc = x.index.get_loc(start_dt)
        if isinstance(start_loc, slice):
            start_loc = start_loc.start
        start_loc = int(start_loc)
        if start_loc <= 0:
            continue

        x_train = x.iloc[:start_loc]
        y_train = y.iloc[:start_loc]
        x_pred = x.loc[seg_idx]
        ridge_seg, gb_seg = _fit_model_signals_on_slice(
            x_train=x_train,
            y_train=y_train,
            x_pred=x_pred,
            ridge_alpha=float(ridge_alpha),
            gb_rounds=int(gb_rounds),
            gb_learning_rate=float(gb_learning_rate),
        )
        ridge_q.loc[seg_idx] = ridge_seg
        gb_q.loc[seg_idx] = gb_seg

    return ridge_q.fillna(0.0), gb_q.fillna(0.0)


def _pick_family_diverse_defaults(
    eligible: pd.DataFrame,
    score_col: str,
    top_n: int,
) -> List[str]:
    if eligible.empty or score_col not in eligible.columns:
        return []
    ranked = eligible.sort_values([score_col, "selection_score"], ascending=False).reset_index(drop=True)
    picks: List[str] = []
    used_families = set()
    for _, row in ranked.iterrows():
        strategy = str(row["strategy"])
        family = str(row["family"])
        if family in used_families:
            continue
        picks.append(strategy)
        used_families.add(family)
        if len(picks) >= int(top_n):
            break
    if len(picks) < int(top_n):
        extras = [str(x) for x in ranked["strategy"].tolist() if str(x) not in picks]
        picks.extend(extras[: max(int(top_n) - len(picks), 0)])
    return picks


@st.cache_data(show_spinner=False)
def build_submission_feature_pool(min_trades_per_year: float) -> Dict[str, object]:
    payload = load_survival_layer(reject_for_missing_logic=False)
    diagnostics = payload["diagnostics"].copy()
    diagnostics["sample_sharpe"] = diagnostics["out_sample_sharpe"].fillna(diagnostics["sharpe"])
    diagnostics["walkforward_ann_return"] = diagnostics["out_sample_cagr"].fillna(diagnostics["ann_return"])
    diagnostics["selection_score"] = (
        0.50 * _selection_score(diagnostics["sample_sharpe"])
        + 0.35 * diagnostics["consistency_index"].fillna(0.0)
        + 0.15 * _selection_score(diagnostics["trades_per_year"].fillna(0.0))
    )

    eligible = diagnostics[diagnostics["trades_per_year"].fillna(0.0) >= float(min_trades_per_year)].copy()
    eligible = eligible.sort_values(["selection_score", "walkforward_ann_return"], ascending=False).reset_index(drop=True)
    default_features = _pick_family_diverse_defaults(eligible, score_col="walkforward_ann_return", top_n=3)
    return {
        "eligible": eligible,
        "default_features": default_features,
    }


def _summarize_submission_model(
    name: str,
    pnl: pd.Series,
    signal: pd.Series,
    split_idx: int,
) -> Dict[str, float]:
    full = compute_metrics(pnl, signal)
    ins = compute_metrics(pnl.iloc[:split_idx], signal.iloc[:split_idx])
    oos = compute_metrics(pnl.iloc[split_idx:], signal.iloc[split_idx:])
    trades_total = float((signal.diff().abs().fillna(0.0) > 1e-12).sum())
    n_years = max(int(signal.index.year.nunique()), 1)
    consistency = 1.0 if np.sign(ins["sharpe"]) == np.sign(oos["sharpe"]) else 0.0
    return {
        "model": name,
        "full_sample_sharpe": float(full["sharpe"]),
        "in_sample_sharpe": float(ins["sharpe"]),
        "out_sample_sharpe": float(oos["sharpe"]),
        "full_sample_ann_return": float(full["ann_return"]),
        "out_sample_ann_return": float(oos["ann_return"]),
        "full_sample_max_drawdown": float(full["max_drawdown"]),
        "out_sample_max_drawdown": float(oos["max_drawdown"]),
        "trades_total": trades_total,
        "trades_per_year": trades_total / n_years,
        "consistency_index": consistency,
    }


@st.cache_data(show_spinner=True)
def build_final_submission_payload(
    top_k: int,
    min_trades_per_year: float,
    split_ratio: float,
    ridge_alpha: float,
    gb_rounds: int,
    gb_learning_rate: float,
    selected_features: Tuple[str, ...],
) -> Dict[str, object]:
    pool = build_submission_feature_pool(min_trades_per_year=float(min_trades_per_year))
    eligible = pool["eligible"].copy()
    default_features = list(pool["default_features"])
    payload = load_survival_layer(reject_for_missing_logic=False)
    diagnostics = payload["diagnostics"].copy()
    pnl_df = payload["pnl_df"].copy()
    signal_df = payload["signal_df"].copy()
    rets = payload["returns"].copy()
    tcost = ResearchConfig().tcost

    diagnostics["sample_sharpe"] = diagnostics["out_sample_sharpe"].fillna(diagnostics["sharpe"])
    diagnostics["walkforward_ann_return"] = diagnostics["out_sample_cagr"].fillna(diagnostics["ann_return"])

    if eligible.empty:
        return {
            "error": f"No strategies satisfy trades/year >= {min_trades_per_year}. Lower the threshold.",
            "eligible": eligible,
        }

    x_universe = signal_df[eligible["strategy"].tolist()].fillna(0.0).copy()
    y = rets.shift(-1).fillna(0.0).reindex(x_universe.index)
    split_idx = int(len(x_universe) * float(split_ratio))
    split_idx = min(max(split_idx, 252), max(len(x_universe) - 5, 1))

    benchmark_full = compute_metrics(y)
    benchmark_is = compute_metrics(y.iloc[:split_idx])
    benchmark_oos = compute_metrics(y.iloc[split_idx:])

    eligible["oos_ann_return"] = eligible["out_sample_cagr"].fillna(eligible["ann_return"])
    eligible["oos_sharpe"] = eligible["out_sample_sharpe"].fillna(eligible["sharpe"])
    eligible["oos_excess_ann_return_vs_brent"] = eligible["oos_ann_return"] - float(benchmark_oos["ann_return"])
    eligible["oos_excess_sharpe_vs_brent"] = eligible["oos_sharpe"] - float(benchmark_oos["sharpe"])
    eligible["beats_brent_oos_ann_return"] = eligible["oos_ann_return"] > float(benchmark_oos["ann_return"])
    eligible["beats_brent_oos_sharpe"] = eligible["oos_sharpe"] > float(benchmark_oos["sharpe"])
    eligible["beats_brent_oos"] = eligible["beats_brent_oos_ann_return"] & eligible["beats_brent_oos_sharpe"]
    eligible = eligible.sort_values(["selection_score", "walkforward_ann_return"], ascending=False).reset_index(drop=True)
    recommended = eligible.head(int(max(3, top_k))).copy()

    eligible_set = set(eligible["strategy"].tolist())
    selected_features_clean = [s for s in list(selected_features) if s in eligible_set]
    if not selected_features_clean:
        selected_features_clean = default_features if default_features else recommended["strategy"].head(3).tolist()
    if not selected_features_clean:
        return {
            "error": f"No strategies satisfy trades/year >= {min_trades_per_year}. Lower the threshold.",
            "eligible": eligible,
        }

    selected = eligible[eligible["strategy"].isin(selected_features_clean)].copy()
    selected["selection_order"] = selected["strategy"].map(
        {name: i for i, name in enumerate(selected_features_clean, start=1)}
    )
    selected = selected.sort_values("selection_order").drop(columns=["selection_order"])

    x = signal_df[selected_features_clean].fillna(0.0).copy()
    y = rets.shift(-1).fillna(0.0).reindex(x.index)

    x_train = x.iloc[:split_idx]
    mu = x_train.mean()
    sd = x_train.std().replace(0.0, 1.0)
    x_std = ((x - mu) / sd).fillna(0.0)

    x_train_np = x_std.iloc[:split_idx].to_numpy()
    y_train_np = y.iloc[:split_idx].to_numpy()
    beta = _fit_ridge_beta(x_train_np, y_train_np, alpha=float(ridge_alpha))
    ridge_score = pd.Series(x_std.to_numpy() @ beta, index=x.index, name="ridge_score")

    gb_base, gb_components = _fit_gradient_boosting_proxy(
        x_train_np,
        y_train_np,
        n_rounds=int(gb_rounds),
        learning_rate=float(gb_learning_rate),
    )
    gb_score = pd.Series(
        _predict_gradient_boosting_proxy(x_std.to_numpy(), gb_base, gb_components),
        index=x.index,
        name="gb_score",
    )

    def score_to_signal(score: pd.Series) -> pd.Series:
        scale = float(score.iloc[:split_idx].std())
        scale = max(scale, 1e-6)
        return pd.Series(np.tanh(score / scale), index=score.index)

    ridge_signal = score_to_signal(ridge_score)
    gb_signal = score_to_signal(gb_score)
    ensemble_signal = 0.5 * ridge_signal + 0.5 * gb_signal
    ridge_q_signal, gb_q_signal = _apply_quarterly_oos_rebalance(
        x=x,
        y=y,
        split_idx=split_idx,
        base_ridge_signal=ridge_signal,
        base_gb_signal=gb_signal,
        ridge_alpha=float(ridge_alpha),
        gb_rounds=int(gb_rounds),
        gb_learning_rate=float(gb_learning_rate),
    )
    ensemble_q_signal = 0.5 * ridge_q_signal + 0.5 * gb_q_signal
    eq_signal = x.mean(axis=1).clip(-1.0, 1.0)

    def signal_to_pnl(signal: pd.Series) -> pd.Series:
        turnover = signal.diff().abs().fillna(0.0)
        return signal * y - float(tcost) * turnover

    best_pool_candidates = eligible.sort_values(
        ["beats_brent_oos", "oos_excess_ann_return_vs_brent", "oos_excess_sharpe_vs_brent", "selection_score"],
        ascending=False,
    )
    best_pool_strategy = str(best_pool_candidates.iloc[0]["strategy"])
    best_pool_model_label = f"Best Pool Strategy ({best_pool_strategy})"
    best_pool_signal = signal_df[best_pool_strategy].reindex(x.index).fillna(0.0).rename("best_pool_strategy_signal")
    best_pool_pnl = pnl_df[best_pool_strategy].reindex(x.index).fillna(0.0).rename("best_pool_strategy")

    pnl_map = {
        "Linear Regression": signal_to_pnl(ridge_signal).rename("linear_regression"),
        "Linear Regression (Quarterly Rebalanced OOS)": signal_to_pnl(ridge_q_signal).rename("linear_regression_quarterly_oos"),
        "Gradient Boosting Proxy": signal_to_pnl(gb_signal).rename("gradient_boosting_proxy"),
        "Gradient Boosting Proxy (Quarterly Rebalanced OOS)": signal_to_pnl(gb_q_signal).rename("gradient_boosting_proxy_quarterly_oos"),
        "Ensemble (50/50)": signal_to_pnl(ensemble_signal).rename("ensemble_50_50"),
        "Ensemble (Quarterly Rebalanced OOS)": signal_to_pnl(ensemble_q_signal).rename("ensemble_quarterly_oos"),
        "Equal-Weight Feature Blend": signal_to_pnl(eq_signal).rename("equal_weight_features"),
        best_pool_model_label: best_pool_pnl,
        "Buy & Hold Brent": y.rename("buy_hold_brent"),
    }
    signal_map = {
        "Linear Regression": ridge_signal.rename("linear_regression_signal"),
        "Linear Regression (Quarterly Rebalanced OOS)": ridge_q_signal.rename("linear_regression_quarterly_oos_signal"),
        "Gradient Boosting Proxy": gb_signal.rename("gradient_boosting_proxy_signal"),
        "Gradient Boosting Proxy (Quarterly Rebalanced OOS)": gb_q_signal.rename("gradient_boosting_proxy_quarterly_oos_signal"),
        "Ensemble (50/50)": ensemble_signal.rename("ensemble_signal"),
        "Ensemble (Quarterly Rebalanced OOS)": ensemble_q_signal.rename("ensemble_quarterly_oos_signal"),
        "Equal-Weight Feature Blend": eq_signal.rename("equal_weight_feature_signal"),
        best_pool_model_label: best_pool_signal,
    }

    model_rows = []
    model_names = list(signal_map.keys())
    for model_name in model_names:
        model_rows.append(
            _summarize_submission_model(
                model_name,
                pnl_map[model_name],
                signal_map[model_name],
                split_idx=split_idx,
            )
        )
    model_table = pd.DataFrame(model_rows)
    model_table["oos_excess_ann_return_vs_brent"] = model_table["out_sample_ann_return"] - float(benchmark_oos["ann_return"])
    model_table["oos_excess_sharpe_vs_brent"] = model_table["out_sample_sharpe"] - float(benchmark_oos["sharpe"])
    model_table["full_excess_ann_return_vs_brent"] = model_table["full_sample_ann_return"] - float(benchmark_full["ann_return"])
    model_table["full_excess_sharpe_vs_brent"] = model_table["full_sample_sharpe"] - float(benchmark_full["sharpe"])
    model_table["beats_brent_oos_ann_return"] = model_table["out_sample_ann_return"] > float(benchmark_oos["ann_return"])
    model_table["beats_brent_oos_sharpe"] = model_table["out_sample_sharpe"] > float(benchmark_oos["sharpe"])
    model_table["beats_brent_full_ann_return"] = model_table["full_sample_ann_return"] > float(benchmark_full["ann_return"])
    model_table["beats_brent_full_sharpe"] = model_table["full_sample_sharpe"] > float(benchmark_full["sharpe"])
    model_table["beats_brent_oos"] = model_table["beats_brent_oos_ann_return"] & model_table["beats_brent_oos_sharpe"]
    model_table = model_table.sort_values(
        ["beats_brent_oos", "oos_excess_ann_return_vs_brent", "oos_excess_sharpe_vs_brent", "out_sample_sharpe"],
        ascending=False,
    ).reset_index(drop=True)

    linear_coef = (
        pd.DataFrame({"feature": x.columns, "coefficient": beta})
        .assign(abs_coef=lambda d: d["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns=["abs_coef"])
        .reset_index(drop=True)
    )

    boost_rows = []
    for round_id, (j, w) in enumerate(gb_components, start=1):
        boost_rows.append({"round": round_id, "feature": x.columns[int(j)], "step_weight": float(w)})
    boost_df = pd.DataFrame(boost_rows)
    if not boost_df.empty:
        boost_summary = (
            boost_df.groupby("feature", as_index=False)
            .agg(rounds_used=("round", "count"), cumulative_weight=("step_weight", "sum"))
            .assign(abs_weight=lambda d: d["cumulative_weight"].abs())
            .sort_values("abs_weight", ascending=False)
            .drop(columns=["abs_weight"])
            .reset_index(drop=True)
        )
    else:
        boost_summary = pd.DataFrame(columns=["feature", "rounds_used", "cumulative_weight"])

    variable_glossary = pd.DataFrame(
        [
            {"variable": "selection_top_k", "value": int(top_k), "description": "Auto-ranked shortlist size shown in the dashboard."},
            {"variable": "manual_selected_features", "value": int(len(selected_features_clean)), "description": "Count of manually selected features used for model fitting."},
            {"variable": "test_period_rebalance", "value": "Quarterly", "description": "Linear/GB/Ensemble are re-fitted each calendar quarter in the OOS segment."},
            {"variable": "min_trades_per_year", "value": float(min_trades_per_year), "description": "Minimum trade intensity filter."},
            {"variable": "sample_sharpe", "value": "out_sample_sharpe (fallback sharpe)", "description": "Primary quality metric in ranking."},
            {"variable": "walkforward_ann_return", "value": "out_sample_cagr (fallback ann_return)", "description": "Walkforward annualized return proxy used for default feature picks."},
            {"variable": "consistency_index", "value": "0/1 from IS/OOS Sharpe sign match", "description": "Stability term in feature ranking."},
            {"variable": "selection_score", "value": "0.50*z(sample_sharpe)+0.35*consistency+0.15*z(trades_per_year)", "description": "Feature ranking formula."},
            {"variable": "split_ratio", "value": float(split_ratio), "description": "Chronological train/test split for model fitting."},
            {"variable": "ridge_alpha", "value": float(ridge_alpha), "description": "L2 regularization strength in linear regression."},
            {"variable": "gb_rounds", "value": int(gb_rounds), "description": "Boosting iterations in the proxy gradient booster."},
            {"variable": "gb_learning_rate", "value": float(gb_learning_rate), "description": "Step size for each boosting round."},
            {"variable": "transaction_cost", "value": float(tcost), "description": "Cost applied per unit signal change."},
            {"variable": "target", "value": "next_day_return", "description": "Model predicts next-day Brent return."},
            {"variable": "signal_transform", "value": "tanh(score/std_train)", "description": "Maps model score to [-1, 1] position size."},
        ]
    )

    selected_cols = [
        "strategy",
        "family",
        "sample_sharpe",
        "walkforward_ann_return",
        "oos_ann_return",
        "oos_sharpe",
        "oos_excess_ann_return_vs_brent",
        "oos_excess_sharpe_vs_brent",
        "beats_brent_oos",
        "sharpe",
        "consistency_index",
        "trades_per_year",
        "selection_score",
        "final_robustness_score",
        "OVERFIT_RISK",
    ]
    selected_view = selected[selected_cols].copy().reset_index(drop=True)
    recommended_view = recommended[selected_cols].copy().reset_index(drop=True)
    eligible_view = eligible[selected_cols].copy().reset_index(drop=True)

    beaters = eligible[eligible["beats_brent_oos"]].copy()
    if beaters.empty:
        beaters = eligible.copy()
    index_beaters_view = beaters.sort_values(
        ["oos_excess_ann_return_vs_brent", "oos_excess_sharpe_vs_brent", "selection_score"],
        ascending=False,
    ).head(15)[selected_cols].reset_index(drop=True)

    return {
        "error": "",
        "default_features": default_features,
        "selected_features": selected_features_clean,
        "best_pool_strategy": best_pool_strategy,
        "best_pool_model_label": best_pool_model_label,
        "recommended_table": recommended_view,
        "selected_table": selected_view,
        "eligible_table": eligible_view,
        "index_beaters_table": index_beaters_view,
        "model_table": model_table,
        "linear_coefficients": linear_coef,
        "boosting_summary": boost_summary,
        "variable_glossary": variable_glossary,
        "pnl_map": pnl_map,
        "signal_map": signal_map,
        "split_idx": int(split_idx),
        "returns_target": y,
        "benchmark_metrics": {
            "full_ann_return": float(benchmark_full["ann_return"]),
            "full_sharpe": float(benchmark_full["sharpe"]),
            "is_ann_return": float(benchmark_is["ann_return"]),
            "is_sharpe": float(benchmark_is["sharpe"]),
            "oos_ann_return": float(benchmark_oos["ann_return"]),
            "oos_sharpe": float(benchmark_oos["sharpe"]),
        },
    }


def _pick_model_row(model_table: pd.DataFrame, model_names: List[str]) -> pd.Series:
    if model_table.empty or "model" not in model_table.columns:
        return pd.Series(dtype=float)
    for name in model_names:
        rows = model_table[model_table["model"] == name]
        if not rows.empty:
            return rows.iloc[0]
    return pd.Series(dtype=float)


@st.cache_data(show_spinner=True)
def _build_fixed_feature_combo_candidates(
    eligible: pd.DataFrame,
    max_seed_features: int = 14,
    shortlist_cap: int = 20,
) -> List[Tuple[str, ...]]:
    ranked = eligible.sort_values(
        ["walkforward_ann_return", "sample_sharpe", "selection_score"],
        ascending=False,
    ).reset_index(drop=True)
    seed_features = [str(x) for x in ranked["strategy"].head(int(max_seed_features)).tolist()]
    if len(seed_features) < 3:
        return []

    family_lookup = ranked.set_index("strategy")["family"].astype(str).to_dict()
    valid_sizes = [s for s in [3, 4, 5, 6, 7, 8, 9] if s <= len(seed_features)]
    combo_set: set[Tuple[str, ...]] = set()

    for size in valid_sizes:
        combo_set.add(tuple(seed_features[:size]))

    ranking_variants = [
        ["walkforward_ann_return", "sample_sharpe", "selection_score"],
        ["sample_sharpe", "walkforward_ann_return", "selection_score"],
        ["selection_score", "walkforward_ann_return", "sample_sharpe"],
    ]
    for cols in ranking_variants:
        ordered = (
            ranked.sort_values(cols, ascending=False)["strategy"]
            .astype(str)
            .head(int(max_seed_features))
            .tolist()
        )
        for size in valid_sizes:
            picks: List[str] = []
            used_families = set()
            for strat in ordered:
                fam = family_lookup.get(strat, "")
                if fam not in used_families:
                    picks.append(strat)
                    used_families.add(fam)
                if len(picks) >= size:
                    break
            if len(picks) < size:
                extras = [s for s in ordered if s not in picks]
                picks.extend(extras[: size - len(picks)])
            if len(picks) == size:
                combo_set.add(tuple(picks))

    small_pool = seed_features[: min(9, len(seed_features))]
    for size in [3, 4]:
        if len(small_pool) < size:
            continue
        for combo in itertools.combinations(small_pool, size):
            combo_set.add(tuple(combo))

    rng = np.random.default_rng(20260219)
    for _ in range(240):
        size = int(rng.choice(valid_sizes))
        idx = sorted(rng.choice(len(seed_features), size=size, replace=False))
        combo_set.add(tuple(seed_features[int(i)] for i in idx))

    lookup = ranked.set_index("strategy")[["walkforward_ann_return", "sample_sharpe", "selection_score", "family"]].copy()
    rows: List[Dict[str, object]] = []
    for combo in combo_set:
        names = list(combo)
        if any(name not in lookup.index for name in names):
            continue
        sub = lookup.loc[names]
        rows.append(
            {
                "features_csv": "|".join(names),
                "mean_wf_ann_return": float(sub["walkforward_ann_return"].mean()),
                "mean_sample_sharpe": float(sub["sample_sharpe"].mean()),
                "max_wf_ann_return": float(sub["walkforward_ann_return"].max()),
                "family_diversity": float(sub["family"].nunique() / max(len(names), 1)),
                "feature_count": len(names),
            }
        )
    if not rows:
        return []

    combo_df = pd.DataFrame(rows).drop_duplicates(subset=["features_csv"]).reset_index(drop=True)
    combo_df["heuristic_score"] = (
        0.45 * _selection_score(combo_df["mean_wf_ann_return"])
        + 0.30 * _selection_score(combo_df["mean_sample_sharpe"])
        + 0.15 * _selection_score(combo_df["max_wf_ann_return"])
        + 0.10 * _selection_score(combo_df["family_diversity"])
    )
    combo_df = combo_df.sort_values(
        ["heuristic_score", "mean_wf_ann_return", "mean_sample_sharpe", "family_diversity"],
        ascending=False,
    ).head(int(shortlist_cap))
    return [tuple(str(x) for x in str(v).split("|")) for v in combo_df["features_csv"].tolist()]


@st.cache_data(show_spinner=True)
def build_fixed_submission_proposals(top_n: int = 5) -> pd.DataFrame:
    fixed_top_k = 10
    fixed_min_trades = 20.0
    param_grid = [
        (0.70, 3.0, 80, 0.08),
        (0.70, 6.0, 80, 0.08),
        (0.70, 3.0, 120, 0.08),
        (0.70, 6.0, 120, 0.08),
        (0.75, 3.0, 80, 0.08),
        (0.75, 6.0, 80, 0.08),
        (0.75, 3.0, 120, 0.08),
        (0.75, 6.0, 120, 0.08),
    ]
    empty_cols = [
        "rank",
        "proposal_label",
        "features_csv",
        "features_display",
        "feature_count",
        "final_model",
        "top_k",
        "min_trades_per_year",
        "split_ratio",
        "ridge_alpha",
        "gb_rounds",
        "gb_learning_rate",
        "out_sample_ann_return",
        "out_sample_sharpe",
        "out_sample_max_drawdown",
        "full_sample_ann_return",
        "full_sample_sharpe",
        "oos_excess_ann_return_vs_brent",
        "oos_excess_sharpe_vs_brent",
        "beats_brent_oos",
    ]

    pool = build_submission_feature_pool(min_trades_per_year=float(fixed_min_trades))
    eligible = pool["eligible"].copy()
    if eligible.empty:
        return pd.DataFrame(columns=empty_cols)

    combo_candidates = _build_fixed_feature_combo_candidates(
        eligible=eligible,
        max_seed_features=14,
        shortlist_cap=20,
    )
    if not combo_candidates:
        return pd.DataFrame(columns=empty_cols)

    rows: List[Dict[str, object]] = []
    for combo in combo_candidates:
        for split_ratio, ridge_alpha, gb_rounds, gb_learning_rate in param_grid:
            bundle = build_final_submission_payload(
                top_k=int(fixed_top_k),
                min_trades_per_year=float(fixed_min_trades),
                split_ratio=float(split_ratio),
                ridge_alpha=float(ridge_alpha),
                gb_rounds=int(gb_rounds),
                gb_learning_rate=float(gb_learning_rate),
                selected_features=tuple(combo),
            )
            if bundle.get("error"):
                continue
            model_table = bundle.get("model_table", pd.DataFrame())
            if not isinstance(model_table, pd.DataFrame) or model_table.empty:
                continue
            candidate_models = model_table[
                model_table["model"].isin(
                    [
                        "Linear Regression (Quarterly Rebalanced OOS)",
                        "Gradient Boosting Proxy (Quarterly Rebalanced OOS)",
                        "Ensemble (Quarterly Rebalanced OOS)",
                    ]
                )
            ].copy()
            if candidate_models.empty:
                continue
            candidate_models = candidate_models.sort_values(
                [
                    "beats_brent_oos",
                    "oos_excess_ann_return_vs_brent",
                    "out_sample_ann_return",
                    "out_sample_sharpe",
                    "full_sample_ann_return",
                ],
                ascending=False,
            ).reset_index(drop=True)
            best = candidate_models.iloc[0]
            rows.append(
                {
                    "features_csv": "|".join(combo),
                    "features_display": ", ".join(combo),
                    "feature_count": len(combo),
                    "final_model": str(best["model"]),
                    "top_k": int(fixed_top_k),
                    "min_trades_per_year": float(fixed_min_trades),
                    "split_ratio": float(split_ratio),
                    "ridge_alpha": float(ridge_alpha),
                    "gb_rounds": int(gb_rounds),
                    "gb_learning_rate": float(gb_learning_rate),
                    "out_sample_ann_return": float(best["out_sample_ann_return"]),
                    "out_sample_sharpe": float(best["out_sample_sharpe"]),
                    "out_sample_max_drawdown": float(best["out_sample_max_drawdown"]),
                    "full_sample_ann_return": float(best["full_sample_ann_return"]),
                    "full_sample_sharpe": float(best["full_sample_sharpe"]),
                    "oos_excess_ann_return_vs_brent": float(best["oos_excess_ann_return_vs_brent"]),
                    "oos_excess_sharpe_vs_brent": float(best["oos_excess_sharpe_vs_brent"]),
                    "beats_brent_oos": bool(best["beats_brent_oos"]),
                }
            )

    if not rows:
        return pd.DataFrame(columns=empty_cols)

    proposals = pd.DataFrame(rows)
    proposals = proposals.sort_values(
        [
            "beats_brent_oos",
            "oos_excess_ann_return_vs_brent",
            "out_sample_ann_return",
            "out_sample_sharpe",
            "full_sample_ann_return",
        ],
        ascending=False,
    )
    proposals = proposals.drop_duplicates(
        subset=["features_csv", "final_model", "split_ratio", "ridge_alpha", "gb_rounds", "gb_learning_rate"],
        keep="first",
    )
    beaters = proposals[proposals["beats_brent_oos"]].copy()
    if len(beaters) >= int(top_n):
        proposals = beaters
    proposals = proposals.head(int(top_n)).reset_index(drop=True)
    proposals["rank"] = np.arange(1, len(proposals) + 1)
    proposals["proposal_label"] = proposals.apply(
        lambda r: (
            f"Top {int(r['rank'])} | {str(r['final_model'])} | "
            f"{int(r['feature_count'])}F | OOS Ann {float(r['out_sample_ann_return']):.1%} | Sharpe {float(r['out_sample_sharpe']):.2f} | "
            f"Brent {'Yes' if bool(r['beats_brent_oos']) else 'No'}"
        ),
        axis=1,
    )
    return proposals[empty_cols].copy()


def render_final_submission_page(show_tutor: bool) -> None:
    section_header(
        "Final Submission Strategy Builder",
        "Select robust features, fit transparent models, and publish one final strategy.",
        "Feature-screened meta-model construction (linear + boosting proxy) with full variable audit.",
        show_tutor,
    )

    st.markdown(
        "This page is submission-oriented: it enforces your constraints (sample Sharpe, consistency, trades/year >= 20), "
        "fits model-based final strategies, and documents every variable used. Add/remove features and all model outputs update live. "
        "Model-based strategies are quarterly rebalanced in the test period."
    )

    show_fixed_key = "final_submission_show_fixed_best5"
    forced_model_key = "final_submission_forced_model"
    if show_fixed_key not in st.session_state:
        st.session_state[show_fixed_key] = False

    if st.button("Show Best 5 Proposed Strategies", key="final_submission_show_best5_button"):
        st.session_state[show_fixed_key] = True

    c1, c2, c3, c4 = st.columns(4)
    top_k = c1.slider("Auto shortlist size", min_value=3, max_value=20, value=10, step=1, key="final_submission_top_k")
    min_trades = c2.number_input(
        "Minimum trades/year filter",
        min_value=1.0,
        max_value=100.0,
        value=20.0,
        step=1.0,
        key="final_submission_min_trades",
    )
    split_ratio = c3.slider("Train ratio", min_value=0.60, max_value=0.85, value=0.70, step=0.05, key="final_submission_split_ratio")
    c4.metric("Test Rebalance", "Quarterly")

    m1, m2 = st.columns(2)
    ridge_alpha = m1.number_input("Ridge alpha", min_value=0.1, max_value=20.0, value=4.0, step=0.1, key="final_submission_ridge_alpha")
    gb_rounds = int(m2.slider("Boosting rounds", min_value=20, max_value=200, value=80, step=10, key="final_submission_gb_rounds"))
    gb_lr = st.slider("Boosting learning rate", min_value=0.01, max_value=0.50, value=0.08, step=0.01, key="final_submission_gb_lr")

    feature_pool = build_submission_feature_pool(min_trades_per_year=float(min_trades))
    eligible_pool = feature_pool["eligible"].copy()
    default_features = list(feature_pool["default_features"])
    if eligible_pool.empty:
        st.warning(f"No strategies satisfy trades/year >= {min_trades}. Lower the threshold.")
        return

    fixed_proposals = build_fixed_submission_proposals(top_n=5)
    if st.session_state.get(show_fixed_key, False):
        st.markdown("### Best 5 Proposed Strategies To Implement (Fixed)")
        st.caption(
            "These proposals are precomputed from broad feature-count and alpha-grid search. They are fixed and do not re-rank with page sliders."
        )
        if fixed_proposals.empty:
            st.warning("No fixed proposals are available right now.")
        else:
            st.dataframe(
                fixed_proposals[
                    [
                        "rank",
                        "final_model",
                        "feature_count",
                        "split_ratio",
                        "ridge_alpha",
                        "gb_rounds",
                        "gb_learning_rate",
                        "out_sample_ann_return",
                        "out_sample_sharpe",
                        "oos_excess_ann_return_vs_brent",
                        "beats_brent_oos",
                        "features_display",
                    ]
                ].round(4),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("### Feature Basket Control")
    st.caption(
        "Default basket = top 3 by walkforward annualized return, each from different families when available."
    )
    selection_key = "final_submission_feature_basket"
    proposal_choice_key = "final_submission_fixed_proposal_choice"
    proposal_applied_key = "final_submission_fixed_proposal_applied"
    options = eligible_pool["strategy"].tolist()
    if selection_key not in st.session_state:
        st.session_state[selection_key] = default_features
    st.session_state[selection_key] = [s for s in st.session_state[selection_key] if s in options]
    if not st.session_state[selection_key]:
        st.session_state[selection_key] = default_features

    def _apply_proposal_row(row: pd.Series) -> None:
        chosen_features = [s for s in str(row["features_csv"]).split("|") if s in options]
        if chosen_features:
            st.session_state[selection_key] = chosen_features
        st.session_state["final_submission_top_k"] = int(row["top_k"])
        st.session_state["final_submission_min_trades"] = float(row["min_trades_per_year"])
        st.session_state["final_submission_split_ratio"] = float(row["split_ratio"])
        st.session_state["final_submission_ridge_alpha"] = float(row["ridge_alpha"])
        st.session_state["final_submission_gb_rounds"] = int(row["gb_rounds"])
        st.session_state["final_submission_gb_lr"] = float(row["gb_learning_rate"])
        st.session_state[forced_model_key] = str(row["final_model"])
        st.session_state[show_fixed_key] = True

    with st.sidebar:
        st.markdown("---")
        st.markdown("### Best 5 Proposed Strategies (Fixed)")
        st.caption(
            "Fixed proposals from broad combinations of features and alphas. Use these for client-facing submission candidates."
        )
        if fixed_proposals.empty:
            st.caption("No fixed proposals available.")
        else:
            proposal_options = fixed_proposals["proposal_label"].astype(str).tolist()
            if proposal_choice_key not in st.session_state or st.session_state[proposal_choice_key] not in proposal_options:
                st.session_state[proposal_choice_key] = proposal_options[0]
            selected_proposal = st.selectbox(
                "Best 5 submission proposals",
                options=proposal_options,
                key=proposal_choice_key,
                help="Select a fixed proposal and apply it to load its features, model, and alpha settings.",
            )
            selected_row = fixed_proposals[fixed_proposals["proposal_label"] == selected_proposal].iloc[0]

            if st.button("Apply Selected Proposed Strategy", key="final_submission_apply_fixed_choice"):
                _apply_proposal_row(selected_row)
                st.session_state[proposal_applied_key] = selected_proposal
                st.rerun()

            if st.button("Apply Best Of Best (Top 1)", key="final_submission_apply_fixed_top1"):
                _apply_proposal_row(fixed_proposals.iloc[0])
                st.session_state[proposal_applied_key] = str(fixed_proposals.iloc[0]["proposal_label"])
                st.rerun()

            with st.expander("View fixed top 5", expanded=False):
                st.dataframe(
                    fixed_proposals[
                        [
                            "rank",
                            "final_model",
                            "feature_count",
                            "split_ratio",
                            "ridge_alpha",
                            "gb_rounds",
                            "gb_learning_rate",
                            "features_display",
                            "out_sample_ann_return",
                            "out_sample_sharpe",
                            "oos_excess_ann_return_vs_brent",
                            "beats_brent_oos",
                        ]
                    ].round(4),
                    use_container_width=True,
                    hide_index=True,
                )

    if st.button("Reset Basket To Default Top 3"):
        st.session_state[selection_key] = default_features
        st.session_state.pop(forced_model_key, None)

    selected_features = st.multiselect(
        "Add or remove features for the final strategy submission",
        options=options,
        key=selection_key,
        help="Changing this list recalculates coefficients, model metrics, and all charts on this page.",
    )
    if not selected_features:
        st.warning("Select at least one feature to build the final strategy.")
        return

    bundle = build_final_submission_payload(
        top_k=int(top_k),
        min_trades_per_year=float(min_trades),
        split_ratio=float(split_ratio),
        ridge_alpha=float(ridge_alpha),
        gb_rounds=int(gb_rounds),
        gb_learning_rate=float(gb_lr),
        selected_features=tuple(selected_features),
    )

    if bundle.get("error"):
        st.warning(str(bundle["error"]))
        if "eligible" in bundle and isinstance(bundle["eligible"], pd.DataFrame):
            st.dataframe(bundle["eligible"], use_container_width=True, hide_index=True)
        return

    st.markdown("### Feature Selection Results")
    st.caption(
        "Ranking formula: selection_score = 0.50*z(sample_sharpe) + 0.35*consistency_index + 0.15*z(trades_per_year). "
        "Default basket uses highest walkforward annualized return (out_sample_cagr proxy) with family diversification."
    )
    st.markdown(f"**Default top-3 basket:** `{', '.join(bundle['default_features'])}`")
    st.dataframe(bundle["recommended_table"].round(4), use_container_width=True, hide_index=True)
    st.markdown(f"**Selected features ({len(bundle['selected_features'])}):** `{', '.join(bundle['selected_features'])}`")
    st.dataframe(bundle["selected_table"].round(4), use_container_width=True, hide_index=True)
    st.markdown(f"**Best pool strategy vs Brent (auto-picked):** `{bundle['best_pool_strategy']}`")

    st.markdown("### Top Strategies That Beat Brent (From Eligible Pool)")
    st.dataframe(bundle["index_beaters_table"].round(4), use_container_width=True, hide_index=True)

    with st.expander("See full eligible feature pool", expanded=False):
        st.dataframe(bundle["eligible_table"].round(4), use_container_width=True, hide_index=True)

    st.markdown("### Model Inputs, Variables, and Hyperparameters")
    st.dataframe(bundle["variable_glossary"], use_container_width=True, hide_index=True)

    v1, v2 = st.columns(2)
    with v1:
        st.markdown("**Linear Regression Coefficients**")
        st.dataframe(bundle["linear_coefficients"].round(6), use_container_width=True, hide_index=True)
    with v2:
        st.markdown("**Gradient Boosting Proxy Feature Contributions**")
        st.dataframe(bundle["boosting_summary"].round(6), use_container_width=True, hide_index=True)

    st.markdown("### Model Performance (Full Sample + IS/OOS)")
    st.dataframe(bundle["model_table"].round(4), use_container_width=True, hide_index=True)

    model_options = [m for m in bundle["model_table"]["model"].tolist() if m in bundle["signal_map"]]
    if not model_options:
        st.warning("No model outputs available for final strategy selection.")
        return
    forced_model = str(st.session_state.get(forced_model_key, ""))
    if forced_model in model_options:
        default_model = forced_model
    elif "Ensemble (Quarterly Rebalanced OOS)" in model_options:
        default_model = "Ensemble (Quarterly Rebalanced OOS)"
    else:
        default_model = model_options[0]
    final_model = st.selectbox(
        "Final strategy model",
        options=model_options,
        index=max(model_options.index(default_model), 0),
    )

    final_pnl = bundle["pnl_map"][final_model]
    final_signal = bundle["signal_map"][final_model]
    target_rets = bundle["returns_target"]
    split_idx = int(bundle["split_idx"])

    kpis = extended_kpis(final_pnl, final_signal)
    st.markdown(f"### Final Strategy KPI Scorecard: `{final_model}`")
    show_kpi_tiles(kpis, show_tutor=show_tutor)

    bench = bundle["benchmark_metrics"]
    final_oos = compute_metrics(final_pnl.iloc[split_idx:], final_signal.iloc[split_idx:])
    final_full = compute_metrics(final_pnl, final_signal)
    beats_oos = (final_oos["ann_return"] > bench["oos_ann_return"]) and (final_oos["sharpe"] > bench["oos_sharpe"])
    bx1, bx2, bx3, bx4 = st.columns(4)
    bx1.metric("OOS Excess Ann Return vs Brent", pct(final_oos["ann_return"] - bench["oos_ann_return"]))
    bx2.metric("OOS Excess Sharpe vs Brent", fmt(final_oos["sharpe"] - bench["oos_sharpe"], 3))
    bx3.metric("Full Excess Ann Return vs Brent", pct(final_full["ann_return"] - bench["full_ann_return"]))
    bx4.metric("Beats Brent OOS?", "Yes" if beats_oos else "No")

    compare_curves = {
        "Final Strategy": (1.0 + final_pnl.fillna(0.0)).cumprod(),
        "Buy & Hold Brent": (1.0 + target_rets.fillna(0.0)).cumprod(),
    }
    best_pool_label = str(bundle["best_pool_model_label"])
    if best_pool_label in bundle["pnl_map"] and best_pool_label != final_model:
        compare_curves["Best Pool Strategy"] = (1.0 + bundle["pnl_map"][best_pool_label].fillna(0.0)).cumprod()
    if "Ensemble (Quarterly Rebalanced OOS)" in bundle["pnl_map"] and final_model != "Ensemble (Quarterly Rebalanced OOS)":
        compare_curves["Quarterly Ensemble"] = (1.0 + bundle["pnl_map"]["Ensemble (Quarterly Rebalanced OOS)"].fillna(0.0)).cumprod()
    compare = pd.DataFrame(compare_curves, index=final_pnl.index)
    fig = px.line(compare.reset_index(), x="date", y=compare.columns.tolist(), title="Final Strategy vs Benchmarks")
    if not compare.empty:
        split_loc = min(max(split_idx, 0), len(compare.index) - 1)
        split_dt = pd.to_datetime(compare.index[split_loc]).to_pydatetime()
        fig.add_shape(
            type="line",
            x0=split_dt,
            x1=split_dt,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(dash="dash", color="#7f7f7f"),
        )
        fig.add_annotation(
            x=split_dt,
            y=1.0,
            xref="x",
            yref="paper",
            text="Train/Test Split",
            showarrow=False,
            yshift=10,
            font=dict(size=11, color="#7f7f7f"),
        )
    st.plotly_chart(fig, use_container_width=True)

    is_oos = pd.DataFrame(
        [
            {"segment": "In Sample", "sharpe": compute_metrics(final_pnl.iloc[:split_idx], final_signal.iloc[:split_idx])["sharpe"]},
            {"segment": "Out of Sample", "sharpe": compute_metrics(final_pnl.iloc[split_idx:], final_signal.iloc[split_idx:])["sharpe"]},
        ]
    )
    st.plotly_chart(px.bar(is_oos, x="segment", y="sharpe", title="Final Strategy IS vs OOS Sharpe"), use_container_width=True)

    st.markdown("### Yearly Final Strategy Breakdown")
    st.dataframe(yearly_breakdown(final_pnl, final_signal).round(4), use_container_width=True, hide_index=True)


def show_kpi_tiles(kpis: Dict[str, float], show_tutor: bool) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sharpe", fmt(kpis["sharpe"]))
    c2.metric("Sortino", fmt(kpis["sortino"]))
    c3.metric("Calmar", fmt(kpis["calmar"]))
    c4.metric("Ann Return", pct(kpis["ann_return"]))
    c5.metric("Max Drawdown", pct(kpis["max_drawdown"]))
    if show_tutor:
        i1, i2, i3, i4, i5 = st.columns(5)
        with i1:
            tutor_popover("Sharpe", *KPI_EXPLAINERS["Sharpe"], enabled=True)
        with i2:
            tutor_popover("Sortino", *KPI_EXPLAINERS["Sortino"], enabled=True)
        with i3:
            tutor_popover("Calmar", *KPI_EXPLAINERS["Calmar"], enabled=True)
        with i4:
            tutor_popover("Ann Return", *KPI_EXPLAINERS["Ann Return"], enabled=True)
        with i5:
            tutor_popover("Max Drawdown", *KPI_EXPLAINERS["Max Drawdown"], enabled=True)

    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("Ann Vol", pct(kpis["ann_vol"]))
    c7.metric("Hit Rate", pct(kpis["hit_rate"]))
    c8.metric("Turnover (daily)", fmt(kpis["turnover"], 4))
    c9.metric("Avg Hold (days)", fmt(kpis["avg_holding_period"], 1))
    c10.metric("Profit Factor", fmt(kpis["profit_factor"], 3))
    if show_tutor:
        i6, i7, i8, i9, i10 = st.columns(5)
        with i6:
            tutor_popover("Ann Vol", *KPI_EXPLAINERS["Ann Vol"], enabled=True)
        with i7:
            tutor_popover("Hit Rate", *KPI_EXPLAINERS["Hit Rate"], enabled=True)
        with i8:
            tutor_popover("Turnover (daily)", *KPI_EXPLAINERS["Turnover (daily)"], enabled=True)
        with i9:
            tutor_popover("Avg Hold (days)", *KPI_EXPLAINERS["Avg Hold (days)"], enabled=True)
        with i10:
            tutor_popover("Profit Factor", *KPI_EXPLAINERS["Profit Factor"], enabled=True)

    c11, c12, c13, c14, c15 = st.columns(5)
    c11.metric("VaR 95% (daily)", pct(kpis["var_95"]))
    c12.metric("CVaR 95% (daily)", pct(kpis["cvar_95"]))
    c13.metric("Tail Ratio", fmt(kpis["tail_ratio"], 3))
    c14.metric("Long Exposure", pct(kpis["long_exposure"]))
    c15.metric("Short Exposure", pct(kpis["short_exposure"]))
    if show_tutor:
        i11, i12, i13, i14, i15 = st.columns(5)
        with i11:
            tutor_popover("VaR 95% (daily)", *KPI_EXPLAINERS["VaR 95% (daily)"], enabled=True)
        with i12:
            tutor_popover("CVaR 95% (daily)", *KPI_EXPLAINERS["CVaR 95% (daily)"], enabled=True)
        with i13:
            tutor_popover("Tail Ratio", *KPI_EXPLAINERS["Tail Ratio"], enabled=True)
        with i14:
            tutor_popover("Long Exposure", *KPI_EXPLAINERS["Long Exposure"], enabled=True)
        with i15:
            tutor_popover("Short Exposure", *KPI_EXPLAINERS["Short Exposure"], enabled=True)


def main() -> None:
    st.set_page_config(page_title="Crude Oil Strategy Dashboard", layout="wide")
    st.title("Crude Oil Strategy Research Dashboard")
    st.caption(
        "Complete strategy universe analytics: ranking, family behavior, one-by-one drilldown, walkforward diagnostics, and risk KPIs."
    )

    tables = load_output_tables()
    close, rets, base_pnl_df, base_signal_df, base_family_df, regimes_df = build_universe()

    benchmark = rets.fillna(0.0)
    benchmark_kpi = extended_kpis(benchmark, pd.Series(1.0, index=benchmark.index))

    with st.sidebar:
        st.header("Controls")
        system_mode = st.radio(
            "Engine Mode",
            options=["Discovery Mode", "Survival Mode"],
            index=0,
            help="Discovery Mode preserves the original dashboard. Survival Mode runs the robustness/rejection framework.",
        )
        show_tutor = st.checkbox(
            "Tutor layer (i)",
            value=True,
            help="Turn on plain-language explanations without removing advanced analytics.",
        )
        survival_profile = "Strict (reject missing economic logic)"
        include_survival_universe = True
        if system_mode == "Discovery Mode":
            include_survival_universe = st.checkbox(
                "Include Survival strategy universe",
                value=True,
                help="Adds Survival-Mode generated strategies into Strategy Explorer and Cross-Compare.",
            )
            pnl_df, signal_df, family_df = combine_strategy_universe(
                base_pnl_df,
                base_signal_df,
                base_family_df,
                include_survival_universe=include_survival_universe,
            )
            metrics_df = build_metrics_table(pnl_df, signal_df, family_df)
            family_filter = st.multiselect(
                "Filter families",
                options=sorted(metrics_df["family"].dropna().unique().tolist()),
                default=sorted(metrics_df["family"].dropna().unique().tolist()),
            )
            sort_col = st.selectbox(
                "Sort strategies by",
                options=[
                    "sharpe",
                    "activity_adjusted_sharpe",
                    "sortino",
                    "calmar",
                    "ann_return",
                    "max_drawdown",
                    "trade_events_total",
                    "trades_per_year",
                    "turnover",
                ],
                index=0,
            )
            top_n = st.slider("Top N to show", min_value=5, max_value=100, value=30, step=5)
            if show_tutor:
                st.markdown("### Quick Guide")
                tutor_popover(
                    "What is a strategy?",
                    "A strategy is a rule that converts market data into buy/sell/flat decisions.",
                    "Each strategy here is a daily signal transformed into cost-adjusted PnL.",
                    enabled=True,
                )
                tutor_popover(
                    "What is a family?",
                    "A family groups similar strategy ideas, like trend or mean reversion.",
                    "Families help compare whether one modeling style is structurally stronger.",
                    enabled=True,
                )
                tutor_popover(
                    "What is walkforward?",
                    "Pick the recent winner, then trade it in the next period, repeat.",
                    "Rolling train-select-test loop to reduce lookahead and adapt to regime shifts.",
                    enabled=True,
                )
        else:
            survival_profile = st.radio(
                "Survival filter profile",
                options=[
                    "Strict (reject missing economic logic)",
                    "Statistical-only (do not reject missing economic logic)",
                ],
                index=0,
                help="Strict mode rejects undocumented ideas; statistical-only mode keeps them and only flags them.",
            )
            st.info(
                "Survival Mode is active: hard filters, rejection engine, parameter surfaces, walk-forward survival, and feature blending lab."
            )

    if system_mode == "Survival Mode":
        reject_missing_logic = survival_profile.startswith("Strict")
        render_survival_mode(show_tutor=show_tutor, reject_for_missing_logic=reject_missing_logic)
        return

    if system_mode != "Discovery Mode":
        return

    filtered = metrics_df[metrics_df["family"].isin(family_filter)].copy()
    filtered = filtered.sort_values(sort_col, ascending=False).reset_index(drop=True)
    if filtered.empty:
        st.warning("No strategies match the current family filter. Please select at least one family.")
        return

    tabs = st.tabs(
        ["Universe Overview", "Strategy Explorer", "Cross-Compare", "Walkforward & Robustness", "Final Submission Strategy"]
    )

    with tabs[0]:
        section_header(
            "Universe Summary",
            "This is the full map of all candidate strategies and how they performed.",
            "Cross-sectional summary over the generated strategy universe with ranking stats.",
            show_tutor,
        )
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Strategies", f"{len(filtered)}")
        m2.metric("Families", f"{filtered['family'].nunique()}")
        m3.metric("Best Sharpe", fmt(filtered["sharpe"].max()))
        m4.metric("Median Sharpe", fmt(filtered["sharpe"].median()))
        m5.metric("Median Trades/Year", fmt(filtered["trades_per_year"].median(), 1))
        m6.metric("Median Total Trades", fmt(filtered["trade_events_total"].median(), 0))
        if show_tutor:
            x1, x2, x3, x4, x5, x6 = st.columns(6)
            with x1:
                tutor_popover(
                    "Strategies count",
                    "How many different rule sets are currently in the filtered list.",
                    "Cardinality of candidate strategy columns after filtering.",
                    True,
                )
            with x2:
                tutor_popover(
                    "Families count",
                    "How many idea-types are represented (trend, MR, etc.).",
                    "Unique family labels among filtered strategies.",
                    True,
                )
            with x3:
                tutor_popover(
                    "Best Sharpe",
                    "The strongest risk-adjusted strategy in this filtered group.",
                    "Maximum Sharpe ratio in filtered universe.",
                    True,
                )
            with x4:
                tutor_popover(
                    "Median Sharpe",
                    "Middle-of-the-pack quality of strategies.",
                    "Median Sharpe gives robustness view beyond top outliers.",
                    True,
                )
            with x5:
                tutor_popover("Trades/Year", *KPI_EXPLAINERS["Trades/Year"], enabled=True)
            with x6:
                tutor_popover("Trades (Total)", *KPI_EXPLAINERS["Trades (Total)"], enabled=True)

        section_header(
            "Strategy Ranking Table",
            "Each row is one strategy; columns are KPI outcomes.",
            "Multi-metric snapshot used for selection and diagnostics.",
            show_tutor,
        )
        st.dataframe(
            filtered.head(top_n)[
                [
                    "strategy",
                    "family",
                    "sharpe",
                    "sortino",
                    "calmar",
                    "ann_return",
                    "ann_vol",
                    "max_drawdown",
                    "activity_adjusted_sharpe",
                    "trade_events_total",
                    "trades_per_year",
                    "turnover",
                    "avg_holding_period",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        if show_tutor:
            c1, c2 = st.columns(2)
            with c1:
                tutor_popover(
                    "How to read this table",
                    "Start with Sharpe and Max Drawdown, then check turnover and holding period.",
                    "Prioritize risk-adjusted performance, then implementation realism via trading intensity.",
                    True,
                )
            with c2:
                tutor_popover(
                    "Why multiple KPIs?",
                    "One metric can hide weaknesses. Example: high return but huge drawdown.",
                    "Use a multi-objective lens: return, risk, tail behavior, and trading frictions.",
                    True,
                )

        scatter = px.scatter(
            filtered,
            x="ann_vol",
            y="ann_return",
            color="family",
            size=np.maximum(filtered["sharpe"].fillna(0) + 1.0, 0.1),
            hover_data=["strategy", "sharpe", "sortino", "calmar", "max_drawdown"],
            title="Risk-Return Map (bubble size linked to Sharpe)",
        )
        if show_tutor:
            tutor_popover(
                "Risk-Return Map",
                "Right = more volatile, up = higher return. Bigger bubble = better Sharpe.",
                "Cross-sectional efficient-frontier style view with family clustering.",
                True,
            )
        st.plotly_chart(scatter, use_container_width=True)

        fam = (
            filtered.groupby("family")[
                ["sharpe", "activity_adjusted_sharpe", "ann_return", "max_drawdown", "turnover", "trade_events_total"]
            ]
            .agg(["mean", "median", "max"])
            .round(4)
        )
        section_header(
            "Family-Level Performance Aggregates",
            "Compares idea-types, not just single strategies.",
            "Grouped statistics by strategy family for structural diagnostics.",
            show_tutor,
        )
        st.dataframe(fam, use_container_width=True)

        section_header(
            "Hero Strategies (By Family and Overall)",
            "Shows the best strategy in each family, both pure Sharpe and trade-activity-adjusted Sharpe.",
            "Unweighted hero uses max Sharpe; weighted hero uses Sharpe * activity weight from total trade count.",
            show_tutor,
        )
        heroes_unweighted = build_hero_table(filtered, "sharpe")
        heroes_weighted = build_hero_table(filtered, "activity_adjusted_sharpe")
        h1, h2 = st.columns(2)
        with h1:
            st.markdown("**Family Heroes: Unweighted (Sharpe)**")
            st.dataframe(heroes_unweighted, use_container_width=True, hide_index=True)
        with h2:
            st.markdown("**Family Heroes: Trade-Weighted**")
            st.dataframe(heroes_weighted, use_container_width=True, hide_index=True)
        if not filtered.empty:
            raw_best = filtered.loc[filtered["sharpe"].idxmax()]
            wgt_best = filtered.loc[filtered["activity_adjusted_sharpe"].idxmax()]
            o1, o2 = st.columns(2)
            with o1:
                st.markdown("**Overall Best (Unweighted)**")
                st.write(
                    f"`{raw_best['strategy']}` | Sharpe {fmt(raw_best['sharpe'])} | "
                    f"Trades {fmt(raw_best['trade_events_total'], 0)}"
                )
                st.caption(strategy_one_liner(str(raw_best["strategy"])))
            with o2:
                st.markdown("**Overall Best (Trade-Weighted)**")
                st.write(
                    f"`{wgt_best['strategy']}` | Activity-Adjusted Sharpe {fmt(wgt_best['activity_adjusted_sharpe'])} | "
                    f"Trades {fmt(wgt_best['trade_events_total'], 0)}"
                )
                st.caption(strategy_one_liner(str(wgt_best["strategy"])))
            if show_tutor:
                tutor_popover(
                    "Activity-Adjusted Sharpe",
                    KPI_EXPLAINERS["Activity-Adjusted Sharpe"][0],
                    KPI_EXPLAINERS["Activity-Adjusted Sharpe"][1],
                    True,
                )
        if show_tutor:
            st.markdown("**Family Explainers**")
            fam_cols = st.columns(2)
            fam_items = sorted(filtered["family"].dropna().unique().tolist())
            for i, fam_name in enumerate(fam_items):
                simple, pro = FAMILY_EXPLAINERS.get(
                    fam_name,
                    ("This is a custom family label.", "No static explainer was defined for this label."),
                )
                with fam_cols[i % 2]:
                    tutor_popover(fam_name, simple, pro, True)

    with tabs[1]:
        section_header(
            "One-by-One Strategy Deep Dive",
            "Pick one strategy and inspect everything about it.",
            "Single-strategy path-dependent and regime-conditional diagnostics.",
            show_tutor,
        )
        strategy = st.selectbox("Choose strategy", options=filtered["strategy"].tolist(), index=0)
        st.info(f"Strategy summary: {strategy_one_liner(strategy)}")
        strategy_family = filtered.loc[filtered["strategy"] == strategy, "family"].iloc[0]
        st.caption(f"Selected strategy family: `{strategy_family}`")
        if show_tutor:
            simple, pro = FAMILY_EXPLAINERS.get(
                strategy_family,
                ("This is a custom family label.", "No static explainer was defined for this label."),
            )
            tutor_popover(f"Family: {strategy_family}", simple, pro, True)
        ts = build_strategy_timeseries(strategy, pnl_df, signal_df)
        kpis = extended_kpis(ts["returns"], ts["signal"])
        section_header(
            "KPI Scorecard",
            "These are the core quality checks for this strategy.",
            "Comprehensive risk/return/liquidity/tail diagnostics for the selected signal.",
            show_tutor,
        )
        show_kpi_tiles(kpis, show_tutor=show_tutor)

        strategy_row = metrics_df.loc[metrics_df["strategy"] == strategy].iloc[0]
        section_header(
            "Trading Activity Snapshot",
            "How often this strategy actually trades, so returns are interpreted with execution reality.",
            "Trade-event diagnostics derived from signal state transitions and time coverage.",
            show_tutor,
        )
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Trades (Total)", fmt(strategy_row["trade_events_total"], 0))
        a2.metric("Trades/Year", fmt(strategy_row["trades_per_year"], 1))
        a3.metric("Trade Day %", pct(strategy_row["trade_day_pct"]))
        a4.metric("Years With Trades", fmt(strategy_row["years_with_trades"], 0))
        if show_tutor:
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                tutor_popover("Trades (Total)", *KPI_EXPLAINERS["Trades (Total)"], enabled=True)
            with b2:
                tutor_popover("Trades/Year", *KPI_EXPLAINERS["Trades/Year"], enabled=True)
            with b3:
                tutor_popover("Trade Day %", *KPI_EXPLAINERS["Trade Day %"], enabled=True)
            with b4:
                tutor_popover("Years With Trades", *KPI_EXPLAINERS["Years With Trades"], enabled=True)

        chart = go.Figure()
        chart.add_trace(go.Scatter(x=ts.index, y=ts["cum"], mode="lines", name=strategy))
        chart.add_trace(
            go.Scatter(
                x=ts.index,
                y=(1 + benchmark).cumprod(),
                mode="lines",
                name="Buy & Hold Brent",
                line=dict(dash="dash"),
            )
        )
        chart.update_layout(title="Cumulative Growth (Strategy vs Buy & Hold)")
        if show_tutor:
            tutor_popover(
                "Cumulative Growth Chart",
                "Shows how $1 would grow over time for the strategy vs buy-and-hold.",
                "Pathwise compounded return comparison against baseline exposure.",
                True,
            )
        st.plotly_chart(chart, use_container_width=True)

        c1, c2 = st.columns(2)
        dd_fig = px.area(ts.reset_index(), x="date", y="drawdown", title="Drawdown")
        c1.plotly_chart(dd_fig, use_container_width=True)
        rs_fig = px.line(
            ts.reset_index(), x="date", y=["rolling_sharpe_6m", "rolling_vol_3m"], title="Rolling Sharpe & Volatility"
        )
        c2.plotly_chart(rs_fig, use_container_width=True)
        if show_tutor:
            t1, t2 = st.columns(2)
            with t1:
                tutor_popover(
                    "Drawdown",
                    "How deep losses got from previous highs, at each point in time.",
                    "Underwater equity curve used for capital-risk tolerance checks.",
                    True,
                )
            with t2:
                tutor_popover(
                    "Rolling Sharpe & Vol",
                    "Shows if quality and risk are stable or drifting over time.",
                    "Local performance/risk regime drift using rolling windows.",
                    True,
                )

        reg = regimes_df[["volatility_regime", "trend_regime", "autocorr_regime", "hurst_regime"]].copy()
        reg = reg.join(ts["returns"])
        section_header(
            "Regime Performance",
            "Checks where this strategy works best: calm vs volatile, trending vs choppy, etc.",
            "Conditional return decomposition by market-state labels.",
            show_tutor,
        )
        for col in ["volatility_regime", "trend_regime", "autocorr_regime", "hurst_regime"]:
            grp = reg.groupby(col)["returns"].agg(["mean", "std", "count"])
            grp["ann_return"] = (1 + grp["mean"]) ** TRADING_DAYS - 1
            grp["ann_sharpe"] = np.sqrt(TRADING_DAYS) * grp["mean"] / (grp["std"] + 1e-12)
            st.markdown(f"**{col}**")
            if show_tutor:
                if col == "volatility_regime":
                    tutor_popover(
                        col,
                        "Splits periods into low/mid/high volatility.",
                        "Bucketed by rolling volatility percentile.",
                        True,
                    )
                elif col == "trend_regime":
                    tutor_popover(
                        col,
                        "Splits periods by trend strength.",
                        "Ranked trend-strength proxy buckets.",
                        True,
                    )
                elif col == "autocorr_regime":
                    tutor_popover(
                        col,
                        "Checks whether returns tend to continue or reverse.",
                        "Sign and magnitude buckets of lag-1 autocorrelation.",
                        True,
                    )
                else:
                    tutor_popover(
                        col,
                        "Rough persistence vs mean-reversion state proxy.",
                        "Hurst exponent bucketed into persistent/random/reverting states.",
                        True,
                    )
            st.dataframe(grp.round(4), use_container_width=True)

        monthly = monthly_return_table(ts["returns"]).round(4)
        section_header(
            "Monthly Returns Heatmap",
            "Green months are gains, red months are losses. Shows seasonality and consistency.",
            "Calendar matrix of monthly compounded returns for distribution diagnostics.",
            show_tutor,
        )
        heat = px.imshow(
            monthly.values,
            x=monthly.columns,
            y=monthly.index.astype(str),
            color_continuous_scale="RdYlGn",
            aspect="auto",
            labels=dict(x="Month", y="Year", color="Return"),
        )
        st.plotly_chart(heat, use_container_width=True)
        st.dataframe(monthly, use_container_width=True)

        section_header(
            "Annual Outcome Breakdown",
            "Year-by-year scorecard with both performance and trade count.",
            "Temporal stability decomposition by calendar year, augmented with annual trade activity.",
            show_tutor,
        )
        st.dataframe(yearly_breakdown(ts["returns"], ts["signal"]).round(4), use_container_width=True, hide_index=True)

    with tabs[2]:
        section_header(
            "Cross-Compare Strategies",
            "Compare several strategies side by side before choosing combinations.",
            "Relative performance, risk profile, and dependence structure analysis.",
            show_tutor,
        )
        picks = st.multiselect(
            "Pick strategies to compare",
            options=filtered["strategy"].tolist(),
            default=filtered["strategy"].head(5).tolist(),
        )
        if picks:
            cum_df = (1 + pnl_df[picks].fillna(0.0)).cumprod()
            cfig = px.line(cum_df.reset_index(), x="date", y=picks, title="Cumulative Performance Comparison")
            if show_tutor:
                tutor_popover(
                    "Cumulative Comparison",
                    "Lets you see which strategy stayed stronger over time, not just at the end.",
                    "Path dependency comparison across selected signals.",
                    True,
                )
            st.plotly_chart(cfig, use_container_width=True)

            compare_rows = []
            trade_lookup = metrics_df.set_index("strategy")[
                ["trade_events_total", "trades_per_year", "trade_day_pct", "years_with_trades", "activity_adjusted_sharpe"]
            ]
            for s in picks:
                k = extended_kpis(pnl_df[s], signal_df[s])
                if s in trade_lookup.index:
                    for col, val in trade_lookup.loc[s].items():
                        k[col] = val
                k["strategy"] = s
                compare_rows.append(k)
            compare_df = pd.DataFrame(compare_rows).set_index("strategy")
            st.dataframe(
                compare_df[
                    [
                        "sharpe",
                        "activity_adjusted_sharpe",
                        "sortino",
                        "calmar",
                        "ann_return",
                        "ann_vol",
                        "max_drawdown",
                        "hit_rate",
                        "trade_events_total",
                        "trades_per_year",
                        "trade_day_pct",
                        "years_with_trades",
                        "turnover",
                        "avg_holding_period",
                        "profit_factor",
                        "tail_ratio",
                        "long_exposure",
                        "short_exposure",
                    ]
                ].round(4),
                use_container_width=True,
            )
            if show_tutor:
                tutor_popover(
                    "Comparison KPI Table",
                    "Use this to choose balanced candidates, not only the top return one.",
                    "Multi-criteria ranking for ensemble candidate selection.",
                    True,
                )

            corr = pnl_df[picks].corr().round(2)
            corr_fig = px.imshow(
                corr.values,
                x=corr.columns,
                y=corr.index,
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
                title="Strategy Return Correlation Matrix",
            )
            if show_tutor:
                tutor_popover(
                    "Correlation Matrix",
                    "Lower correlation means better diversification when combining strategies.",
                    "Pairwise return-correlation structure for portfolio construction.",
                    True,
                )
            st.plotly_chart(corr_fig, use_container_width=True)
        else:
            st.info("Select at least one strategy.")

        section_header(
            "Buy & Hold Baseline KPIs",
            "Simple benchmark: always hold Brent. Use this as your baseline.",
            "Unmanaged directional benchmark for relative alpha attribution.",
            show_tutor,
        )
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Sharpe", fmt(benchmark_kpi["sharpe"]))
        b2.metric("Ann Return", pct(benchmark_kpi["ann_return"]))
        b3.metric("Ann Vol", pct(benchmark_kpi["ann_vol"]))
        b4.metric("Max Drawdown", pct(benchmark_kpi["max_drawdown"]))
        if show_tutor:
            y1, y2, y3, y4 = st.columns(4)
            with y1:
                tutor_popover("Sharpe", *KPI_EXPLAINERS["Sharpe"], enabled=True)
            with y2:
                tutor_popover("Ann Return", *KPI_EXPLAINERS["Ann Return"], enabled=True)
            with y3:
                tutor_popover("Ann Vol", *KPI_EXPLAINERS["Ann Vol"], enabled=True)
            with y4:
                tutor_popover("Max Drawdown", *KPI_EXPLAINERS["Max Drawdown"], enabled=True)

    with tabs[3]:
        section_header(
            "Walkforward Grid Results",
            "This shows which train window and rebalance speed worked best out-of-sample.",
            "Grid search over walkforward hyperparameters (lookback/rebalance).",
            show_tutor,
        )
        wf = tables["walkforward_grid"].copy()
        st.dataframe(wf.sort_values("sharpe", ascending=False), use_container_width=True, hide_index=True)

        best = wf.sort_values("sharpe", ascending=False).iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Lookback", f"{int(best['lookback_m'])}M")
        c2.metric("Best Rebalance", f"{int(best['rebalance_m'])}M")
        c3.metric("Best WF Sharpe", fmt(best["sharpe"]))
        c4.metric("Best WF MaxDD", pct(best["max_drawdown"]))
        if show_tutor:
            z1, z2, z3, z4 = st.columns(4)
            with z1:
                tutor_popover(
                    "Best Lookback",
                    "How much past data was used to choose the next strategy.",
                    "Training window length for each walkforward decision point.",
                    True,
                )
            with z2:
                tutor_popover(
                    "Best Rebalance",
                    "How often the chosen strategy is refreshed.",
                    "Selection update frequency for the meta-strategy.",
                    True,
                )
            with z3:
                tutor_popover("Best WF Sharpe", *KPI_EXPLAINERS["Sharpe"], enabled=True)
            with z4:
                tutor_popover("Best WF MaxDD", *KPI_EXPLAINERS["Max Drawdown"], enabled=True)

        wfp = tables["walkforward_portfolio"]["walkforward_portfolio"].fillna(0.0)
        wf_fig = px.line(
            ((1 + wfp).cumprod()).reset_index(),
            x="date",
            y="walkforward_portfolio",
            title="Walkforward Portfolio Cumulative Growth",
        )
        if show_tutor:
            tutor_popover(
                "Walkforward Equity Curve",
                "This is the realistic live-like curve from adaptive strategy switching.",
                "Out-of-sample stitched portfolio path from periodic strategy re-selection.",
                True,
            )
        st.plotly_chart(wf_fig, use_container_width=True)

        section_header(
            "Rebalance Decisions",
            "Shows which strategy was picked at each rebalance date.",
            "Decision log of the walkforward selector over time.",
            show_tutor,
        )
        st.dataframe(tables["rebalance_details"], use_container_width=True, hide_index=True)

        section_header(
            "Robustness Checks",
            "Tests if results survive noise and random baselines.",
            "Bootstrap and perturbation diagnostics for overfit control.",
            show_tutor,
        )
        r1, r2 = st.columns(2)
        r1.dataframe(tables["robustness_summary"], use_container_width=True, hide_index=True)
        r2.dataframe(tables["noise_injection_test"], use_container_width=True, hide_index=True)
        if show_tutor:
            q1, q2 = st.columns(2)
            with q1:
                tutor_popover(
                    "Bootstrap robustness",
                    "Resamples return blocks to check if Sharpe remains reasonable.",
                    "Block bootstrap Sharpe distribution captures path uncertainty.",
                    True,
                )
            with q2:
                tutor_popover(
                    "Noise injection",
                    "Flips some trade directions and checks how much performance degrades.",
                    "Signal perturbation sensitivity; fragile strategies collapse quickly.",
                    True,
                )

        section_header(
            "Ensemble Outcomes",
            "Combines top strategies and shows if diversification improves outcomes.",
            "Composite portfolio diagnostics using allocation rules across selected strategies.",
            show_tutor,
        )
        st.dataframe(tables["ensemble_metrics"], use_container_width=True, hide_index=True)

    with tabs[4]:
        render_final_submission_page(show_tutor=show_tutor)


if __name__ == "__main__":
    main()
