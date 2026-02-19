from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from research_engine import compute_metrics

TRADING_DAYS = 252
EPS = 1e-12


def build_regime_table(features: pd.DataFrame, rets: pd.Series) -> pd.DataFrame:
    reg = pd.DataFrame(index=features.index)
    vol = features["vol_pct_20"].astype(float)
    trend = features["trend_strength_60"].rank(pct=True)
    hurst = features["hurst_120"].astype(float)
    ac1 = features["ac1_60"].astype(float)
    ret_6m = rets.fillna(0.0).rolling(126).sum()

    reg["volatility_regime"] = np.where(vol > 0.66, "high_vol", np.where(vol < 0.33, "low_vol", "mid_vol"))
    reg["trend_regime"] = np.where(trend > 0.66, "trending", np.where(trend < 0.33, "weak_trend", "neutral"))
    reg["mean_reversion_regime"] = np.where(
        (hurst < 0.45) | (ac1 < -0.03),
        "mean_reverting",
        np.where((hurst > 0.55) | (ac1 > 0.03), "persistent", "mixed"),
    )
    reg["market_regime"] = np.where(ret_6m >= 0.0, "bull", "bear")
    reg["combined_regime"] = (
        reg["volatility_regime"].astype(str) + "|" + reg["trend_regime"].astype(str) + "|" + reg["market_regime"].astype(str)
    )
    return reg


def strategy_regime_breakdown(pnl_df: pd.DataFrame, regime_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if pnl_df.empty:
        return pd.DataFrame()
    for strat in pnl_df.columns:
        s = pnl_df[strat].fillna(0.0)
        frame = regime_df.join(s.rename("returns"), how="inner").dropna(subset=["returns"])
        if frame.empty:
            continue
        for label, grp in frame.groupby("combined_regime"):
            m = compute_metrics(grp["returns"])
            rows.append(
                {
                    "strategy": strat,
                    "regime": label,
                    "count": int(len(grp)),
                    "ann_return": m["ann_return"],
                    "ann_sharpe": m["sharpe"],
                    "max_drawdown": m["max_drawdown"],
                }
            )
    return pd.DataFrame(rows)


def compute_regime_robustness(regime_breakdown: pd.DataFrame) -> pd.DataFrame:
    if regime_breakdown.empty:
        return pd.DataFrame(columns=["strategy", "regime_robustness_score"])

    rows = []
    for strat, grp in regime_breakdown.groupby("strategy"):
        sharpes = grp["ann_sharpe"].replace([np.inf, -np.inf], np.nan).dropna()
        if sharpes.empty:
            rows.append({"strategy": strat, "regime_robustness_score": np.nan})
            continue
        pos_ratio = float((sharpes > 0).mean())
        dispersion = float(sharpes.std(ddof=0))
        tail = float(np.percentile(sharpes, 20))
        score = 0.45 * pos_ratio + 0.35 * (1.0 / (1.0 + dispersion)) + 0.20 * ((tail + 2.0) / 4.0)
        rows.append({"strategy": strat, "regime_robustness_score": float(np.clip(score, 0.0, 1.0))})
    return pd.DataFrame(rows)


def rolling_stability_timeseries(signal: pd.Series, returns: pd.Series, window: int = 252) -> Dict[str, pd.Series]:
    s = signal.fillna(0.0)
    r = returns.fillna(0.0)
    trade_events = (s.diff().abs().fillna(0.0) > 1e-12).astype(float)

    rolling_exposure = s.ne(0.0).rolling(window).mean() * 100.0
    rolling_trade_density = trade_events.rolling(window).sum()
    rolling_sharpe = np.sqrt(TRADING_DAYS) * r.rolling(window).mean() / (r.rolling(window).std() + EPS)
    rolling_drawdown = ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1.0).rolling(window).min()
    return {
        "rolling_exposure": rolling_exposure,
        "rolling_trade_density": rolling_trade_density,
        "rolling_sharpe": rolling_sharpe,
        "rolling_drawdown": rolling_drawdown,
    }
