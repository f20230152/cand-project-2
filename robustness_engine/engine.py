from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from feature_blending_lab import run_feature_blending_lab
from regime_analysis import (
    build_regime_table,
    compute_regime_robustness,
    rolling_stability_timeseries,
    strategy_regime_breakdown,
)
from research_engine import (
    ResearchConfig,
    build_strategy_library,
    compute_metrics,
    evaluate_strategy_library,
    generate_feature_library,
    load_price_data,
    to_pnl,
)
from strategy_diagnostics import (
    HardFilterConfig,
    apply_hard_filters,
    build_logical_registry,
    build_parameter_surface,
    compute_is_oos_deltas,
    compute_trade_exposure_table,
)
from validation_framework import RollingWalkForwardConfig, run_rolling_walkforward

EPS = 1e-12


@dataclass
class SurvivalConfig:
    output_dir: str = "outputs/survival"
    parameter_variation_pct: float = 0.10
    top_k_blending: int = 5
    train_years_walkforward: int = 5
    test_years_walkforward: int = 1
    w_walk_forward_score: float = 0.28
    w_parameter_stability: float = 0.22
    w_trade_consistency: float = 0.18
    w_exposure_stability: float = 0.14
    w_regime_diversification: float = 0.18
    min_trades_per_year: float = 4.0
    max_corr_keep: float = 0.95


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _add_strategy(
    name: str,
    raw_signal: pd.Series,
    family: str,
    rets: pd.Series,
    tcost: float,
    pnl_dict: Dict[str, pd.Series],
    signal_dict: Dict[str, pd.Series],
    family_map: Dict[str, str],
) -> None:
    if name in pnl_dict:
        return
    pnl, signal = to_pnl(raw_signal, rets, tcost)
    if pnl.dropna().std() <= 0:
        return
    pnl_dict[name] = pnl
    signal_dict[name] = signal
    family_map[name] = family


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -25, 25)))


def _fit_logistic_gd(x: np.ndarray, y: np.ndarray, lr: float = 0.05, l2: float = 0.001, n_iter: int = 300) -> np.ndarray:
    n, p = x.shape
    beta = np.zeros(p, dtype=float)
    for _ in range(n_iter):
        p_hat = _sigmoid(x @ beta)
        grad = (x.T @ (p_hat - y)) / max(n, 1) + l2 * beta
        beta -= lr * grad
    return beta


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 3.0) -> np.ndarray:
    p = x.shape[1]
    return np.linalg.pinv(x.T @ x + alpha * np.eye(p)) @ (x.T @ y)


def _random_forest_proxy_scores(x: pd.DataFrame, y_sign: pd.Series, n_trees: int = 70, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    cols = x.columns.tolist()
    votes = np.zeros(len(x), dtype=float)
    y_arr = y_sign.fillna(0.0).to_numpy()
    q_levels = [0.2, 0.35, 0.5, 0.65, 0.8]

    for _ in range(n_trees):
        col = cols[int(rng.integers(0, len(cols)))]
        q = q_levels[int(rng.integers(0, len(q_levels)))]
        thr = float(x[col].quantile(q))
        split = np.where(x[col].to_numpy() >= thr, 1.0, -1.0)
        corr = np.corrcoef(split, y_arr)[0, 1] if np.std(split) > 0 and np.std(y_arr) > 0 else 0.0
        polarity = 1.0 if corr >= 0 else -1.0
        votes += polarity * split
    return pd.Series(votes / n_trees, index=x.index)


def _build_expanded_strategy_family(
    close: pd.Series,
    rets: pd.Series,
    features: pd.DataFrame,
    cfg: ResearchConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    idx = close.index
    pnl_dict: Dict[str, pd.Series] = {}
    signal_dict: Dict[str, pd.Series] = {}
    family_map: Dict[str, str] = {}

    # Time-Based
    turn_of_month = ((idx.day <= 3) | (idx.day >= 28)).astype(float)
    _add_strategy(
        "time_based_turn_of_month_long",
        pd.Series(turn_of_month, index=idx),
        "time_based",
        rets,
        cfg.tcost,
        pnl_dict,
        signal_dict,
        family_map,
    )

    rank_in_month = idx.to_series().groupby(idx.to_period("M")).cumcount()
    size_in_month = idx.to_series().groupby(idx.to_period("M")).transform("size")
    rank_from_end = size_in_month - rank_in_month - 1
    pre_expiry = (rank_from_end <= 2).astype(float)
    _add_strategy(
        "time_based_pre_expiry_long",
        pd.Series(pre_expiry, index=idx),
        "time_based",
        rets,
        cfg.tcost,
        pnl_dict,
        signal_dict,
        family_map,
    )

    third_wed = ((idx.dayofweek == 2) & (idx.day >= 15) & (idx.day <= 21)).astype(float)
    post_fomc_proxy = pd.Series(third_wed, index=idx).shift(1).rolling(2).max().fillna(0.0)
    _add_strategy(
        "time_based_post_fomc_drift_proxy",
        post_fomc_proxy,
        "time_based",
        rets,
        cfg.tcost,
        pnl_dict,
        signal_dict,
        family_map,
    )

    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=idx.min(), end=idx.max())
    pre_holiday = pd.Series(0.0, index=idx)
    post_holiday = pd.Series(0.0, index=idx)
    date_index = set(idx)
    for h in holidays:
        prev_day = h - pd.tseries.offsets.BDay(1)
        next_day = h + pd.tseries.offsets.BDay(1)
        if prev_day in date_index:
            pre_holiday.loc[prev_day] = 1.0
        if next_day in date_index:
            post_holiday.loc[next_day] = 1.0
    _add_strategy(
        "time_based_holiday_effect",
        pre_holiday + post_holiday,
        "time_based",
        rets,
        cfg.tcost,
        pnl_dict,
        signal_dict,
        family_map,
    )

    for d in range(5):
        dow_sig = pd.Series((idx.dayofweek == d).astype(float), index=idx)
        _add_strategy(
            f"time_based_weekday_bias_{d}",
            dow_sig,
            "time_based",
            rets,
            cfg.tcost,
            pnl_dict,
            signal_dict,
            family_map,
        )

    first_last_proxy = pd.Series(np.where(rank_in_month <= 1, 1.0, np.where(rank_from_end <= 1, -1.0, 0.0)), index=idx)
    _add_strategy(
        "time_based_first_hour_last_hour_proxy",
        first_last_proxy,
        "time_based",
        rets,
        cfg.tcost,
        pnl_dict,
        signal_dict,
        family_map,
    )

    # Volatility-Based
    trend_20 = np.sign(close.pct_change(20))
    vol_pct = features["vol_pct_20"].fillna(0.5)
    vix_proxy = pd.Series(np.where(vol_pct < 0.35, trend_20, np.where(vol_pct > 0.75, -trend_20, 0.0)), index=idx)
    _add_strategy(
        "vol_based_vix_regime_switch_proxy",
        vix_proxy,
        "volatility_based",
        rets,
        cfg.tcost,
        pnl_dict,
        signal_dict,
        family_map,
    )

    atr_proxy = rets.abs().rolling(20).mean()
    atr_pct = atr_proxy.rolling(252).rank(pct=True)
    atr_trig = pd.Series(np.where(atr_pct > 0.7, -np.sign(features["z_price_20"]), np.sign(trend_20)), index=idx)
    _add_strategy(
        "vol_based_atr_percentile_trigger",
        atr_trig,
        "volatility_based",
        rets,
        cfg.tcost,
        pnl_dict,
        signal_dict,
        family_map,
    )

    squeeze = atr_pct < 0.30
    breakout = close > close.rolling(20).max().shift(1)
    contraction_breakout = pd.Series(np.where(squeeze & breakout, 1.0, np.where(squeeze & (~breakout), -1.0, 0.0)), index=idx)
    _add_strategy(
        "vol_based_contraction_breakout",
        contraction_breakout,
        "volatility_based",
        rets,
        cfg.tcost,
        pnl_dict,
        signal_dict,
        family_map,
    )

    expansion_fade = pd.Series(np.where(atr_pct > 0.75, -np.sign(features["z_price_10"]), 0.0), index=idx)
    _add_strategy(
        "vol_based_expansion_fade",
        expansion_fade,
        "volatility_based",
        rets,
        cfg.tcost,
        pnl_dict,
        signal_dict,
        family_map,
    )

    # Regime Detection
    trend_classifier = features["trend_strength_60"].rank(pct=True)
    z20 = features["z_price_20"]
    trend_vs_mr = pd.Series(np.where(trend_classifier > 0.60, np.sign(close.pct_change(40)), -np.sign(z20)), index=idx)
    _add_strategy(
        "regime_trend_vs_mean_reversion_classifier",
        trend_vs_mr,
        "regime_detection",
        rets,
        cfg.tcost,
        pnl_dict,
        signal_dict,
        family_map,
    )

    hurst_sw = pd.Series(np.where(features["hurst_120"] > 0.55, np.sign(close.pct_change(30)), -np.sign(close.pct_change(5))), index=idx)
    _add_strategy(
        "regime_hurst_exponent_switch",
        hurst_sw,
        "regime_detection",
        rets,
        cfg.tcost,
        pnl_dict,
        signal_dict,
        family_map,
    )

    entropy_sw = pd.Series(
        np.where(features["entropy_120"] < features["entropy_120"].rolling(252).median(), np.sign(close.pct_change(20)), 0.0),
        index=idx,
    )
    _add_strategy(
        "regime_rolling_entropy_switch",
        entropy_sw,
        "regime_detection",
        rets,
        cfg.tcost,
        pnl_dict,
        signal_dict,
        family_map,
    )

    rv = rets.rolling(20).std().fillna(0.0)
    rv_state = rv.rolling(252).rank(pct=True).fillna(0.5)
    markov_proxy = pd.Series(np.where(rv_state > 0.65, -np.sign(close.pct_change(10)), np.sign(close.pct_change(60))), index=idx)
    _add_strategy(
        "regime_markov_switch_proxy",
        markov_proxy,
        "regime_detection",
        rets,
        cfg.tcost,
        pnl_dict,
        signal_dict,
        family_map,
    )

    cluster_score = 0.5 * features["vol_pct_20"].fillna(0.5) + 0.5 * trend_classifier.fillna(0.5)
    cluster_signal = pd.Series(
        np.where(cluster_score > 0.70, np.sign(close.pct_change(60)), np.where(cluster_score < 0.30, -np.sign(z20), 0.0)),
        index=idx,
    )
    _add_strategy(
        "regime_clustering_switch",
        cluster_signal,
        "regime_detection",
        rets,
        cfg.tcost,
        pnl_dict,
        signal_dict,
        family_map,
    )

    # Statistical
    z30 = features["z_price_30"]
    stat_z = pd.Series(np.where(z30 > 1.25, -1.0, np.where(z30 < -1.25, 1.0, 0.0)), index=idx)
    _add_strategy(
        "stat_zscore_mean_reversion_30_1p25",
        stat_z,
        "statistical",
        rets,
        cfg.tcost,
        pnl_dict,
        signal_dict,
        family_map,
    )

    skew = features["rolling_skew_20"]
    skew_break = pd.Series(np.where(skew > 0.75, 1.0, np.where(skew < -0.75, -1.0, 0.0)), index=idx)
    _add_strategy(
        "stat_rolling_skewness_breakout",
        skew_break,
        "statistical",
        rets,
        cfg.tcost,
        pnl_dict,
        signal_dict,
        family_map,
    )

    kurt = features["rolling_kurt_20"]
    kurt_break = pd.Series(np.where(kurt > kurt.rolling(252).quantile(0.8), -np.sign(close.pct_change(5)), 0.0), index=idx)
    _add_strategy(
        "stat_kurtosis_breakout",
        kurt_break,
        "statistical",
        rets,
        cfg.tcost,
        pnl_dict,
        signal_dict,
        family_map,
    )

    mu_short = rets.rolling(20).mean()
    mu_long = rets.rolling(120).mean()
    distribution_shift = pd.Series(np.where((mu_short - mu_long).abs() > rets.rolling(120).std(), np.sign(mu_short - mu_long), 0.0), index=idx)
    _add_strategy(
        "stat_distribution_shift_detection",
        distribution_shift,
        "statistical",
        rets,
        cfg.tcost,
        pnl_dict,
        signal_dict,
        family_map,
    )

    # ML-Light (interpretable)
    ml_cols = [
        "momentum_vol_adj_20",
        "z_price_20",
        "ac1_20",
        "vol_pct_20",
        "trend_strength_60",
        "rolling_skew_20",
        "rolling_kurt_20",
    ]
    x = features[ml_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_bin = (rets.shift(-1).fillna(0.0) > 0.0).astype(float)
    y_reg = rets.shift(-1).fillna(0.0)
    split = max(int(len(x) * 0.7), 1)
    x_train = x.iloc[:split]
    x_mu = x_train.mean()
    x_sd = x_train.std().replace(0.0, 1.0)
    x_std = ((x - x_mu) / x_sd).fillna(0.0)

    beta_log = _fit_logistic_gd(x_std.iloc[:split].to_numpy(), y_bin.iloc[:split].to_numpy(), lr=0.05, l2=0.001, n_iter=280)
    p_up = pd.Series(_sigmoid(x_std.to_numpy() @ beta_log), index=idx)
    log_sig = pd.Series(np.where(p_up > 0.55, 1.0, np.where(p_up < 0.45, -1.0, 0.0)), index=idx)
    _add_strategy(
        "ml_light_logistic_regression_classifier",
        log_sig,
        "ml_light",
        rets,
        cfg.tcost,
        pnl_dict,
        signal_dict,
        family_map,
    )

    beta_ridge = _fit_ridge(x_std.iloc[:split].to_numpy(), y_reg.iloc[:split].to_numpy(), alpha=4.0)
    ridge_score = pd.Series(x_std.to_numpy() @ beta_ridge, index=idx)
    ridge_sig = ridge_score.clip(-1.5, 1.5)
    _add_strategy(
        "ml_light_ridge_regression_alpha",
        ridge_sig,
        "ml_light",
        rets,
        cfg.tcost,
        pnl_dict,
        signal_dict,
        family_map,
    )

    rf_vote = _random_forest_proxy_scores(x_std, np.sign(y_reg.fillna(0.0)), n_trees=80, seed=cfg.random_seed)
    rf_sig = pd.Series(np.where(rf_vote > 0.08, 1.0, np.where(rf_vote < -0.08, -1.0, 0.0)), index=idx)
    _add_strategy(
        "ml_light_random_forest_vote",
        rf_sig,
        "ml_light",
        rets,
        cfg.tcost,
        pnl_dict,
        signal_dict,
        family_map,
    )

    pnl_df = pd.DataFrame(pnl_dict).replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    signal_df = pd.DataFrame(signal_dict).reindex(columns=pnl_df.columns).fillna(0.0)
    return pnl_df, signal_df, family_map


def _build_final_score(df: pd.DataFrame, cfg: SurvivalConfig) -> pd.DataFrame:
    out = df.copy()

    walk_forward_score = (
        0.55 * ((out["out_sample_sharpe"].fillna(-2.0) + 2.0) / 4.0)
        + 0.25 * ((out["out_sample_cagr"].fillna(-0.3) + 0.30) / 0.60)
        + 0.20 * out["is_oos_gap_score"].fillna(0.0)
    ).clip(0.0, 1.0)

    parameter_stability = out["parameter_stability_score"].fillna(0.45).clip(0.0, 1.0)
    trade_consistency = (
        0.6 * (out["trades_per_year"].fillna(0.0) / max(cfg.min_trades_per_year, EPS)).clip(0.0, 1.0)
        + 0.4 * out["consistency_index"].fillna(0.0)
    ).clip(0.0, 1.0)
    exposure_stability = (1.0 - (out["inactive_percent"].fillna(100.0) / 100.0 - 0.35).abs() / 0.65).clip(0.0, 1.0)
    regime_div = out["regime_robustness_score"].fillna(0.0).clip(0.0, 1.0)

    raw = (
        cfg.w_walk_forward_score * walk_forward_score
        + cfg.w_parameter_stability * parameter_stability
        + cfg.w_trade_consistency * trade_consistency
        + cfg.w_exposure_stability * exposure_stability
        + cfg.w_regime_diversification * regime_div
    )

    penalty = out["red_flag_count"].fillna(0.0) * 0.06
    penalty += np.where(out["OVERFIT_RISK"] == "HIGH", 0.12, np.where(out["OVERFIT_RISK"] == "MEDIUM", 0.05, 0.0))
    final = (raw - penalty).clip(0.0, 1.0)

    out["walk_forward_score"] = walk_forward_score
    out["trade_consistency"] = trade_consistency
    out["exposure_stability"] = exposure_stability
    out["regime_diversification"] = regime_div
    out["final_robustness_score"] = 100.0 * final
    return out


def run_survival_framework(
    base_cfg: ResearchConfig | None = None,
    survival_cfg: SurvivalConfig | None = None,
) -> Dict[str, object]:
    base_cfg = base_cfg or ResearchConfig()
    survival_cfg = survival_cfg or SurvivalConfig()

    out_dir = Path(survival_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    close, rets = load_price_data(base_cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        features = generate_feature_library(close)

    base_pnl, base_signal, base_family = build_strategy_library(close, rets, features, base_cfg)
    extra_pnl, extra_signal, extra_family = _build_expanded_strategy_family(close, rets, features, base_cfg)

    pnl_df = pd.concat([base_pnl, extra_pnl], axis=1)
    signal_df = pd.concat([base_signal, extra_signal], axis=1)
    family_map = {**base_family, **extra_family}

    metrics_df = evaluate_strategy_library(pnl_df, signal_df, family_map)
    trade_exp = compute_trade_exposure_table(signal_df, pnl_df)
    param_diag, param_surface = build_parameter_surface(metrics_df, variation_pct=survival_cfg.parameter_variation_pct)
    is_oos = compute_is_oos_deltas(pnl_df)

    regime_df = build_regime_table(features, rets)
    regime_breakdown = strategy_regime_breakdown(pnl_df, regime_df)
    regime_scores = compute_regime_robustness(regime_breakdown)

    logic_registry = build_logical_registry(list(pnl_df.columns), out_dir)

    diagnostics = (
        metrics_df.merge(trade_exp, on="strategy", how="left")
        .merge(param_diag, on="strategy", how="left")
        .merge(is_oos, on="strategy", how="left")
        .merge(regime_scores, on="strategy", how="left")
        .merge(logic_registry, on="strategy", how="left")
    )
    diagnostics = apply_hard_filters(
        diagnostics,
        HardFilterConfig(
            trades_per_year_min=survival_cfg.min_trades_per_year,
        ),
    )
    diagnostics = _build_final_score(diagnostics, survival_cfg).sort_values("final_robustness_score", ascending=False)

    graveyard = diagnostics[diagnostics["rejected"]].copy()
    robust_pass = diagnostics[~diagnostics["rejected"]].copy()
    if robust_pass.empty:
        robust_pass = diagnostics.head(max(3, survival_cfg.top_k_blending)).copy()

    wf_split, wf_summary, wf_portfolio = run_rolling_walkforward(
        pnl_df,
        RollingWalkForwardConfig(
            train_years=survival_cfg.train_years_walkforward,
            test_years=survival_cfg.test_years_walkforward,
            min_train_days=252,
        ),
    )
    wf_summary_df = pd.DataFrame([wf_summary]) if wf_summary else pd.DataFrame()

    top_strategy = str(robust_pass.iloc[0]["strategy"]) if not robust_pass.empty else str(diagnostics.iloc[0]["strategy"])
    rolling_ts = rolling_stability_timeseries(signal_df[top_strategy], pnl_df[top_strategy], window=252)
    rolling_df = pd.concat(rolling_ts, axis=1)
    rolling_df.index.name = "date"

    blending = run_feature_blending_lab(
        signal_df=signal_df,
        pnl_df=pnl_df,
        rets=rets,
        robust_rank=robust_pass,
        tcost=base_cfg.tcost,
        top_k=survival_cfg.top_k_blending,
    )

    diagnostics.to_csv(out_dir / "robustness_diagnostics.csv", index=False)
    graveyard.to_csv(out_dir / "strategy_graveyard.csv", index=False)
    param_surface.to_csv(out_dir / "parameter_surface.csv", index=False)
    regime_df.to_csv(out_dir / "regime_table.csv")
    regime_breakdown.to_csv(out_dir / "regime_breakdown.csv", index=False)
    wf_split.to_csv(out_dir / "walkforward_splits.csv", index=False)
    wf_summary_df.to_csv(out_dir / "walkforward_summary.csv", index=False)
    wf_portfolio.to_frame("walkforward_oos_portfolio").to_csv(out_dir / "walkforward_oos_portfolio.csv")
    rolling_df.to_csv(out_dir / "rolling_stability.csv")
    blending["metrics"].to_csv(out_dir / "feature_blending_metrics.csv", index=False)
    blending["coefficients"].to_csv(out_dir / "feature_blending_coefficients.csv", index=False)
    blending["correlation"].to_csv(out_dir / "feature_correlation_matrix.csv")
    blending["suggestions"].to_csv(out_dir / "feature_blending_suggestions.csv", index=False)
    blending["blended_pnl"].to_csv(out_dir / "feature_blended_pnl.csv")
    blending["blended_signal"].to_csv(out_dir / "feature_blended_signals.csv")

    return {
        "close": close,
        "returns": rets,
        "features": features,
        "pnl_df": pnl_df,
        "signal_df": signal_df,
        "regime_table": regime_df,
        "regime_breakdown": regime_breakdown,
        "diagnostics": diagnostics,
        "graveyard": graveyard,
        "parameter_surface": param_surface,
        "walkforward_splits": wf_split,
        "walkforward_summary": wf_summary_df,
        "walkforward_portfolio": wf_portfolio,
        "rolling_stability": rolling_df,
        "blending": blending,
        "top_strategy": top_strategy,
        "output_dir": str(out_dir.resolve()),
    }
