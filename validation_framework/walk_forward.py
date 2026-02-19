from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from research_engine import compute_metrics

TRADING_DAYS = 252
EPS = 1e-12


@dataclass
class RollingWalkForwardConfig:
    train_years: int = 5
    test_years: int = 1
    min_train_days: int = 252


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def run_rolling_walkforward(
    pnl_df: pd.DataFrame,
    cfg: RollingWalkForwardConfig | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.Series]:
    cfg = cfg or RollingWalkForwardConfig()
    if pnl_df.empty:
        return pd.DataFrame(), {}, pd.Series(dtype=float)

    idx = pnl_df.index.sort_values()
    years = sorted(idx.year.unique().tolist())
    if len(years) <= cfg.train_years:
        return pd.DataFrame(), {}, pd.Series(index=idx, dtype=float)

    rows = []
    stitched = pd.Series(index=idx, dtype=float)

    for test_start_year in range(years[0] + cfg.train_years, years[-1] + 1, cfg.test_years):
        train_start = pd.Timestamp(year=test_start_year - cfg.train_years, month=1, day=1)
        train_end = pd.Timestamp(year=test_start_year - 1, month=12, day=31)
        test_start = pd.Timestamp(year=test_start_year, month=1, day=1)
        test_end = pd.Timestamp(year=min(test_start_year + cfg.test_years - 1, years[-1]), month=12, day=31)

        train_df = pnl_df.loc[(pnl_df.index >= train_start) & (pnl_df.index <= train_end)]
        test_df = pnl_df.loc[(pnl_df.index >= test_start) & (pnl_df.index <= test_end)]
        if train_df.shape[0] < cfg.min_train_days or test_df.empty:
            continue

        train_scores = train_df.apply(lambda s: compute_metrics(s.dropna())["sharpe"])
        train_scores = train_scores.replace([np.inf, -np.inf], np.nan).dropna()
        if train_scores.empty:
            continue

        best_strategy = str(train_scores.idxmax())
        is_ret = train_df[best_strategy].dropna()
        oos_ret = test_df[best_strategy].dropna()
        if oos_ret.empty:
            continue

        stitched.loc[oos_ret.index] = oos_ret
        is_metrics = compute_metrics(is_ret)
        oos_metrics = compute_metrics(oos_ret)

        rows.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "best_strategy": best_strategy,
                "is_sharpe": is_metrics["sharpe"],
                "oos_sharpe": oos_metrics["sharpe"],
                "is_cagr": is_metrics["ann_return"],
                "oos_cagr": oos_metrics["ann_return"],
                "is_drawdown": is_metrics["max_drawdown"],
                "oos_drawdown": oos_metrics["max_drawdown"],
                "sharpe_delta": oos_metrics["sharpe"] - is_metrics["sharpe"],
                "cagr_delta": oos_metrics["ann_return"] - is_metrics["ann_return"],
            }
        )

    split_df = pd.DataFrame(rows)
    stitched = stitched.fillna(0.0)
    if split_df.empty:
        return split_df, {}, stitched

    stitched_metrics = compute_metrics(stitched)
    sharpe_delta_std = float(split_df["sharpe_delta"].std(ddof=0))
    oos_sharpe_std = float(split_df["oos_sharpe"].std(ddof=0))
    stability_score = 1.0 / (1.0 + sharpe_delta_std + oos_sharpe_std)

    sign_match = np.sign(split_df["is_sharpe"].fillna(0.0)) == np.sign(split_df["oos_sharpe"].fillna(0.0))
    oos_positive = split_df["oos_sharpe"].fillna(0.0) > 0.0
    consistency_index = 0.5 * float(sign_match.mean()) + 0.5 * float(oos_positive.mean())

    summary = {
        "walk_forward_sharpe": float(stitched_metrics["sharpe"]),
        "walk_forward_cagr": float(stitched_metrics["ann_return"]),
        "walk_forward_drawdown": float(stitched_metrics["max_drawdown"]),
        "stability_score": float(stability_score),
        "consistency_index": float(consistency_index),
        "splits": int(len(split_df)),
    }
    summary["walk_forward_score"] = _clip01(
        0.45 * ((summary["walk_forward_sharpe"] + 2.0) / 4.0)
        + 0.25 * ((summary["walk_forward_cagr"] + 0.30) / 0.60)
        + 0.15 * (1.0 - min(abs(summary["walk_forward_drawdown"]), 0.8) / 0.8)
        + 0.15 * summary["consistency_index"]
    )

    return split_df, summary, stitched
