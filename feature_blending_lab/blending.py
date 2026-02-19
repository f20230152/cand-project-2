from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from research_engine import compute_metrics

EPS = 1e-12


def _prep_xy(signal_df: pd.DataFrame, rets: pd.Series, features: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    x = signal_df[features].fillna(0.0).astype(float)
    y = rets.fillna(0.0).astype(float)
    aligned = x.join(y.rename("target"), how="inner")
    return aligned[features], aligned["target"]


def _fit_ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xtx = x.T @ x
    xty = x.T @ y
    return np.linalg.pinv(xtx) @ xty


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 2.0) -> np.ndarray:
    xtx = x.T @ x
    n = xtx.shape[0]
    xty = x.T @ y
    return np.linalg.pinv(xtx + alpha * np.eye(n)) @ xty


def _fit_lasso_cd(x: np.ndarray, y: np.ndarray, alpha: float = 0.002, n_iter: int = 200) -> np.ndarray:
    n, p = x.shape
    beta = np.zeros(p, dtype=float)
    x_sq = (x * x).sum(axis=0) + EPS
    for _ in range(n_iter):
        for j in range(p):
            r = y - x @ beta + x[:, j] * beta[j]
            rho = float((x[:, j] * r).sum())
            if rho < -alpha * n:
                beta[j] = (rho + alpha * n) / x_sq[j]
            elif rho > alpha * n:
                beta[j] = (rho - alpha * n) / x_sq[j]
            else:
                beta[j] = 0.0
    return beta


def _to_pnl(signal: pd.Series, rets: pd.Series, tcost: float) -> pd.Series:
    lagged = signal.shift(1).fillna(0.0).clip(-2.0, 2.0)
    turnover = lagged.diff().abs().fillna(0.0)
    return lagged * rets.fillna(0.0) - tcost * turnover


def run_feature_blending_lab(
    signal_df: pd.DataFrame,
    pnl_df: pd.DataFrame,
    rets: pd.Series,
    robust_rank: pd.DataFrame,
    tcost: float,
    top_k: int = 5,
) -> Dict[str, object]:
    if robust_rank.empty:
        return {
            "selected_features": [],
            "metrics": pd.DataFrame(),
            "coefficients": pd.DataFrame(),
            "correlation": pd.DataFrame(),
            "suggestions": pd.DataFrame(),
            "blended_pnl": pd.DataFrame(),
            "blended_signal": pd.DataFrame(),
        }

    selected = robust_rank["strategy"].head(top_k).tolist()
    selected = [s for s in selected if s in signal_df.columns]
    if len(selected) < 2:
        return {
            "selected_features": selected,
            "metrics": pd.DataFrame(),
            "coefficients": pd.DataFrame(),
            "correlation": pd.DataFrame(),
            "suggestions": pd.DataFrame(),
            "blended_pnl": pd.DataFrame(),
            "blended_signal": pd.DataFrame(),
        }

    x_df, y = _prep_xy(signal_df, rets, selected)
    n = len(x_df)
    split = max(int(n * 0.7), 1)

    x_mu = x_df.iloc[:split].mean()
    x_sd = x_df.iloc[:split].std().replace(0.0, 1.0)
    x_std = (x_df - x_mu) / x_sd

    x_train = x_std.iloc[:split].to_numpy()
    y_train = y.iloc[:split].to_numpy()
    x_full = x_std.to_numpy()

    beta_ols = _fit_ols(x_train, y_train)
    beta_ridge = _fit_ridge(x_train, y_train, alpha=2.5)
    beta_lasso = _fit_lasso_cd(x_train, y_train, alpha=0.002, n_iter=220)

    robust_weights = robust_rank.set_index("strategy")["final_robustness_score"].reindex(selected).fillna(0.0)
    robust_weights = robust_weights / (robust_weights.sum() + EPS)
    vote_weights = robust_weights.to_numpy()

    models = {
        "OLS": beta_ols,
        "Ridge": beta_ridge,
        "Lasso": beta_lasso,
    }
    signal_map: Dict[str, pd.Series] = {}
    pnl_map: Dict[str, pd.Series] = {}
    coeff_rows = []

    for model_name, beta in models.items():
        raw = pd.Series(x_full @ beta, index=x_df.index)
        sig = raw.clip(-1.5, 1.5)
        pnl = _to_pnl(sig, y, tcost)
        signal_map[model_name] = sig
        pnl_map[model_name] = pnl
        for feat, b in zip(selected, beta):
            coeff_rows.append({"model": model_name, "feature": feat, "beta": float(b)})

    vote_raw = pd.Series(np.sign(x_df.to_numpy()) @ vote_weights, index=x_df.index)
    vote_sig = vote_raw.clip(-1.0, 1.0)
    vote_pnl = _to_pnl(vote_sig, y, tcost)
    signal_map["WeightedVoting"] = vote_sig
    pnl_map["WeightedVoting"] = vote_pnl
    for feat, w in zip(selected, vote_weights):
        coeff_rows.append({"model": "WeightedVoting", "feature": feat, "beta": float(w)})

    metrics_rows = []
    for name, pnl in pnl_map.items():
        m = compute_metrics(pnl)
        m["model"] = name
        metrics_rows.append(m)
    metrics_df = pd.DataFrame(metrics_rows).sort_values("sharpe", ascending=False)

    corr = pnl_df[selected].corr()
    suggestions = []
    for i, a in enumerate(selected):
        for b in selected[i + 1 :]:
            c = float(corr.loc[a, b])
            if c > 0.95:
                suggestions.append(
                    {
                        "pair": f"{a} vs {b}",
                        "correlation": c,
                        "recommendation": f"Correlation > 0.95, remove one of {a} or {b}.",
                    }
                )
            if c < -0.10:
                suggestions.append(
                    {
                        "pair": f"{a} vs {b}",
                        "correlation": c,
                        "recommendation": "Negative correlation detected, good blending candidate.",
                    }
                )

    return {
        "selected_features": selected,
        "metrics": metrics_df,
        "coefficients": pd.DataFrame(coeff_rows),
        "correlation": corr,
        "suggestions": pd.DataFrame(suggestions),
        "blended_pnl": pd.DataFrame(pnl_map),
        "blended_signal": pd.DataFrame(signal_map),
    }
