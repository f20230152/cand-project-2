from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from research_engine import avg_holding_period, compute_metrics

EPS = 1e-12


@dataclass
class HardFilterConfig:
    trades_total_min: float = 30.0
    trades_per_year_min: float = 4.0
    min_exposure_pct: float = 10.0
    max_inactive_pct: float = 50.0
    max_flat_period_days: int = 252
    min_parameter_stability: float = 0.28
    max_surface_gradient: float = 7.5
    reject_for_missing_logic: bool = True


def _safe_div(a: float, b: float) -> float:
    return float(a / (b + EPS))


def parse_strategy_template(strategy: str) -> Tuple[str, Tuple[float, ...]]:
    parts = strategy.split("_")
    params: List[float] = []
    template_parts: List[str] = []
    for token in parts:
        if re.fullmatch(r"-?\d+(?:\.\d+)?", token):
            params.append(float(token))
            template_parts.append("{p}")
        else:
            template_parts.append(token)
    return "_".join(template_parts), tuple(params)


def _longest_stagnation_days(returns: pd.Series) -> int:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    at_peak = wealth >= (peak - 1e-12)
    max_run = 0
    run = 0
    for val in at_peak.to_numpy():
        if val:
            run = 0
        else:
            run += 1
            if run > max_run:
                max_run = run
    return int(max_run)


def compute_trade_exposure_table(signal_df: pd.DataFrame, pnl_df: pd.DataFrame) -> pd.DataFrame:
    if signal_df.empty:
        return pd.DataFrame()

    rows = []
    n_years = max(int(signal_df.index.year.nunique()), 1)
    for col in signal_df.columns:
        sig = signal_df[col].fillna(0.0)
        trades = (sig.diff().abs().fillna(0.0) > 1e-12).astype(float)
        exposure = float(sig.ne(0.0).mean()) * 100.0
        inactive = 100.0 - exposure
        rows.append(
            {
                "strategy": col,
                "trades_total": float(trades.sum()),
                "trades_per_year": float(trades.sum() / n_years),
                "exposure_percent": exposure,
                "inactive_percent": inactive,
                "avg_holding_period": float(avg_holding_period(sig)),
                "max_flat_period_days": float(_longest_stagnation_days(pnl_df[col].fillna(0.0))),
            }
        )
    return pd.DataFrame(rows)


def _relative_distance(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    if len(a) != len(b) or len(a) == 0:
        return np.nan
    rel = [abs(x - y) / max(abs(x), EPS) for x, y in zip(a, b)]
    return float(np.mean(rel))


def build_parameter_surface(
    metrics_df: pd.DataFrame,
    variation_pct: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = metrics_df[["strategy", "sharpe", "ann_return", "max_drawdown"]].copy()
    parsed = base["strategy"].map(parse_strategy_template)
    base["parameter_template"] = parsed.map(lambda x: x[0])
    base["parameters"] = parsed.map(lambda x: x[1])
    base["n_parameters"] = base["parameters"].map(len)

    rows = []
    surface_rows = []

    for _, grp in base.groupby("parameter_template"):
        grp = grp.copy()
        for _, rec in grp.iterrows():
            params = rec["parameters"]
            strat = rec["strategy"]
            sharpe = float(rec["sharpe"])
            ann_ret = float(rec["ann_return"])
            mdd = float(rec["max_drawdown"])

            if len(params) == 0 or len(grp) < 3:
                rows.append(
                    {
                        "strategy": strat,
                        "parameter_stability_score": np.nan,
                        "parameter_surface_gradient": np.nan,
                        "plateau_width": np.nan,
                        "surface_smoothness_metric": np.nan,
                        "peak_concentration_ratio": np.nan,
                    }
                )
                continue

            dist = grp["parameters"].map(lambda p: _relative_distance(params, p))
            neigh = grp.loc[(dist <= variation_pct) & (grp["strategy"] != strat)].copy()
            if neigh.empty:
                rows.append(
                    {
                        "strategy": strat,
                        "parameter_stability_score": np.nan,
                        "parameter_surface_gradient": np.nan,
                        "plateau_width": np.nan,
                        "surface_smoothness_metric": np.nan,
                        "peak_concentration_ratio": np.nan,
                    }
                )
                continue

            neigh_dist = neigh["parameters"].map(lambda p: _relative_distance(params, p)).to_numpy()
            neigh_perf = neigh["sharpe"].astype(float).to_numpy()
            gradient = float(np.mean(np.abs(sharpe - neigh_perf) / (neigh_dist + EPS)))
            local_max = float(max(sharpe, np.nanmax(neigh_perf)))
            plateau_width = float(np.mean(neigh_perf >= (0.90 * local_max)))
            smoothness = float(1.0 / (1.0 + np.nanstd(np.diff(np.sort(np.append(neigh_perf, sharpe))))))
            peak_ratio = _safe_div(local_max, np.nanmean(np.sort(np.append(neigh_perf, sharpe))[-max(2, len(neigh_perf) // 3) :]))
            stability = float(np.clip((1.0 / (1.0 + gradient)) * (0.5 + 0.5 * plateau_width), 0.0, 1.0))

            rows.append(
                {
                    "strategy": strat,
                    "parameter_stability_score": stability,
                    "parameter_surface_gradient": gradient,
                    "plateau_width": plateau_width,
                    "surface_smoothness_metric": smoothness,
                    "peak_concentration_ratio": peak_ratio,
                }
            )

            surf = {"strategy": strat, "parameter_template": rec["parameter_template"], "sharpe": sharpe, "cagr": ann_ret, "drawdown": mdd}
            for i, p in enumerate(params, start=1):
                surf[f"param_{i}"] = p
            surface_rows.append(surf)

    out = pd.DataFrame(rows)
    surface = pd.DataFrame(surface_rows)
    return out, surface


def compute_is_oos_deltas(pnl_df: pd.DataFrame, split_ratio: float = 0.7) -> pd.DataFrame:
    if pnl_df.empty:
        return pd.DataFrame()
    cut = max(int(len(pnl_df) * split_ratio), 1)
    is_df = pnl_df.iloc[:cut]
    oos_df = pnl_df.iloc[cut:]
    rows = []
    for col in pnl_df.columns:
        is_m = compute_metrics(is_df[col].dropna())
        oos_m = compute_metrics(oos_df[col].dropna())
        sharpe_delta = float(oos_m["sharpe"] - is_m["sharpe"])
        cagr_delta = float(oos_m["ann_return"] - is_m["ann_return"])
        consistency = 1.0 if np.sign(oos_m["sharpe"]) == np.sign(is_m["sharpe"]) else 0.0
        gap = abs(sharpe_delta)
        rows.append(
            {
                "strategy": col,
                "in_sample_sharpe": is_m["sharpe"],
                "out_sample_sharpe": oos_m["sharpe"],
                "in_sample_cagr": is_m["ann_return"],
                "out_sample_cagr": oos_m["ann_return"],
                "in_sample_drawdown": is_m["max_drawdown"],
                "out_sample_drawdown": oos_m["max_drawdown"],
                "is_oos_sharpe_delta": sharpe_delta,
                "is_oos_cagr_delta": cagr_delta,
                "consistency_index": consistency,
                "is_oos_gap_score": float(1.0 / (1.0 + gap)),
            }
        )
    return pd.DataFrame(rows)


def build_logical_registry(strategy_names: List[str], output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = output_dir / "strategy_logic_registry.csv"
    cols = ["strategy", "economic_hypothesis", "supporting_statistical_evidence"]

    if registry_path.exists():
        reg = pd.read_csv(registry_path)
        for c in cols:
            if c not in reg.columns:
                reg[c] = ""
        reg = reg[cols]
    else:
        reg = pd.DataFrame(columns=cols)

    base = pd.DataFrame({"strategy": strategy_names})
    merged = base.merge(reg, on="strategy", how="left")
    merged["economic_hypothesis"] = merged["economic_hypothesis"].fillna("")
    merged["supporting_statistical_evidence"] = merged["supporting_statistical_evidence"].fillna("")
    merged.to_csv(registry_path, index=False)
    return merged


def apply_hard_filters(diagnostics_df: pd.DataFrame, cfg: HardFilterConfig | None = None) -> pd.DataFrame:
    cfg = cfg or HardFilterConfig()
    df = diagnostics_df.copy()
    red_flags = []
    reasons = []
    overfit_risks = []

    for _, row in df.iterrows():
        flags: List[str] = []
        reject_reasons: List[str] = []

        if float(row.get("trades_total", np.nan)) < cfg.trades_total_min:
            flags.append("LOW_TRADE_COUNT")
            reject_reasons.append("Low trade count")
        if float(row.get("trades_per_year", np.nan)) < cfg.trades_per_year_min:
            flags.append("LOW_TRADES_PER_YEAR")
            if "Low trade count" not in reject_reasons:
                reject_reasons.append("Low trade count")
        if float(row.get("exposure_percent", np.nan)) < cfg.min_exposure_pct:
            flags.append("LOW_EXPOSURE")
            reject_reasons.append("Exposure too low")
        if float(row.get("inactive_percent", np.nan)) > cfg.max_inactive_pct:
            flags.append("HIGH_INACTIVE_TIME")
            if "Exposure too low" not in reject_reasons:
                reject_reasons.append("Exposure too low")
        if float(row.get("max_flat_period_days", np.nan)) > cfg.max_flat_period_days:
            flags.append("LONG_FLAT_PERIOD")
            reject_reasons.append("Flat performance")

        p_stability = float(row.get("parameter_stability_score", np.nan))
        p_grad = float(row.get("parameter_surface_gradient", np.nan))
        p_plateau = float(row.get("plateau_width", np.nan))
        if (not np.isnan(p_stability) and p_stability < cfg.min_parameter_stability) or (
            not np.isnan(p_grad) and p_grad > cfg.max_surface_gradient
        ):
            flags.append("PARAMETER_INSTABILITY")
            reject_reasons.append("Parameter instability")
        if (not np.isnan(p_grad) and p_grad > cfg.max_surface_gradient * 1.2) or (
            not np.isnan(p_plateau) and p_plateau < 0.15
        ):
            flags.append("SHARP_PERFORMANCE_PEAK")
            if "Parameter instability" not in reject_reasons:
                reject_reasons.append("Parameter instability")

        if abs(float(row.get("is_oos_sharpe_delta", 0.0))) > 0.8:
            flags.append("IS_OOS_GAP")
            reject_reasons.append("Walk-forward inconsistency")

        missing_logic = (not str(row.get("economic_hypothesis", "")).strip()) or (
            not str(row.get("supporting_statistical_evidence", "")).strip()
        )
        if missing_logic:
            flags.append("MISSING_LOGIC_DOC")
            if cfg.reject_for_missing_logic:
                reject_reasons.append("No economic logic")

        red_flags.append(", ".join(flags) if flags else "NONE")
        reasons.append(", ".join(dict.fromkeys(reject_reasons)) if reject_reasons else "NONE")

        if "SHARP_PERFORMANCE_PEAK" in flags or "PARAMETER_INSTABILITY" in flags:
            overfit_risks.append("HIGH")
        elif "IS_OOS_GAP" in flags:
            overfit_risks.append("MEDIUM")
        else:
            overfit_risks.append("LOW")

    df["red_flags"] = red_flags
    df["rejection_reason"] = reasons
    df["OVERFIT_RISK"] = overfit_risks
    df["red_flag_count"] = df["red_flags"].map(lambda x: 0 if x == "NONE" else len(str(x).split(",")))
    df["rejected"] = df["rejection_reason"] != "NONE"
    return df
