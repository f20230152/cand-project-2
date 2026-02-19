from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Tuple

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
    return "Rule-based strategy from the research factory; inspect KPIs and charts below for behavior details."


@st.cache_data(show_spinner=False)
def load_output_tables() -> Dict[str, pd.DataFrame]:
    if not OUTPUT_DIR.exists() or not (OUTPUT_DIR / "strategy_metrics.csv").exists():
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
def build_universe() -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = ResearchConfig()
    close, rets = load_price_data(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        features = generate_feature_library(close)
    pnl_df, signal_df, family_map = build_strategy_library(close, rets, features, cfg)
    family_df = pd.Series(family_map, name="family").to_frame()
    regimes = detect_regimes(features)
    return close, rets, pnl_df, signal_df, family_df.join(regimes, how="left")


@st.cache_data(show_spinner=True)
def load_survival_layer() -> Dict[str, object]:
    return run_survival_framework(ResearchConfig(), SurvivalConfig())


def render_survival_mode(show_tutor: bool) -> None:
    payload = load_survival_layer()
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
    st.caption("Survival-first validation: reject fragile ideas, then rank what survives.")

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
            "Auto-generated rejection report with strict hard-filter reasons for auditability.",
            show_tutor,
        )
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
    close, rets, pnl_df, signal_df, tags_df = build_universe()

    metrics_df = tables["strategy_metrics"].copy()
    if "family" not in metrics_df.columns:
        if "family_x" in metrics_df.columns or "family_y" in metrics_df.columns:
            left = metrics_df["family_x"] if "family_x" in metrics_df.columns else pd.Series(index=metrics_df.index)
            right = metrics_df["family_y"] if "family_y" in metrics_df.columns else pd.Series(index=metrics_df.index)
            metrics_df["family"] = left.fillna(right)
        else:
            metrics_df = metrics_df.merge(tags_df[["family"]], left_on="strategy", right_index=True, how="left")

    # Backfill missing family tags from live strategy map when needed.
    fill_map = tags_df["family"].to_dict()
    metrics_df["family"] = metrics_df["family"].fillna(metrics_df["strategy"].map(fill_map))

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
        if system_mode == "Discovery Mode":
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
            st.info(
                "Survival Mode is active: hard filters, rejection engine, parameter surfaces, walk-forward survival, and feature blending lab."
            )

    if system_mode == "Survival Mode":
        render_survival_mode(show_tutor=show_tutor)
        return

    filtered = metrics_df[metrics_df["family"].isin(family_filter)].copy()
    filtered = filtered.sort_values(sort_col, ascending=False).reset_index(drop=True)
    if filtered.empty:
        st.warning("No strategies match the current family filter. Please select at least one family.")
        return

    tabs = st.tabs(
        ["Universe Overview", "Strategy Explorer", "Cross-Compare", "Walkforward & Robustness"]
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

        reg = tags_df[["volatility_regime", "trend_regime", "autocorr_regime", "hurst_regime"]].copy()
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


if __name__ == "__main__":
    main()
