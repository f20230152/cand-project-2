from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import warnings

import numpy as np
import pandas as pd

from backtest import WalkforwardBacktester

TRADING_DAYS = 252
EPS = 1e-12


@dataclass
class ResearchConfig:
    data_path: str = "brent_index.xlsx"
    date_col: str = "date"
    price_col: str = "CO1 Comdty"
    tcost: float = 0.00015
    lookback_grid: Tuple[int, ...] = (6, 12, 24, 36)
    rebalance_grid: Tuple[int, ...] = (1, 3)
    min_abs_ic: float = 0.01
    min_ic_stability: float = 0.52
    feature_corr_cap: float = 0.95
    output_dir: str = "outputs"
    random_seed: int = 42


def load_price_data(cfg: ResearchConfig) -> Tuple[pd.Series, pd.Series]:
    df = pd.read_excel(cfg.data_path, parse_dates=[cfg.date_col])
    df = df[[cfg.date_col, cfg.price_col]].dropna().sort_values(cfg.date_col)
    df = df.set_index(cfg.date_col)
    close = df[cfg.price_col].astype(float)
    rets = close.pct_change().replace([np.inf, -np.inf], np.nan)
    return close, rets


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / (std + EPS)


def compute_rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    up_ewm = up.ewm(alpha=1 / window, adjust=False).mean()
    down_ewm = down.ewm(alpha=1 / window, adjust=False).mean()
    rs = up_ewm / (down_ewm + EPS)
    return 100 - (100 / (1 + rs))


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x = x - x.mean()
    denom = np.sum(x * x)

    def _slope(y: np.ndarray) -> float:
        if np.isnan(y).any():
            return np.nan
        y_centered = y - y.mean()
        return float(np.sum(x * y_centered) / (denom + EPS))

    return series.rolling(window).apply(_slope, raw=True)


def rolling_entropy(series: pd.Series, window: int, bins: int = 10) -> pd.Series:
    def _entropy(arr: np.ndarray) -> float:
        if np.isnan(arr).any():
            return np.nan
        counts, _ = np.histogram(arr, bins=bins)
        total = counts.sum()
        if total == 0:
            return np.nan
        p = counts[counts > 0] / total
        return float(-np.sum(p * np.log(p)))

    return series.rolling(window).apply(_entropy, raw=True)


def rolling_hurst(series: pd.Series, window: int) -> pd.Series:
    def _hurst(arr: np.ndarray) -> float:
        if np.isnan(arr).any():
            return np.nan
        lags = [2, 4, 8, 16]
        valid_lags = [lag for lag in lags if lag < len(arr) // 2]
        if len(valid_lags) < 2:
            return np.nan
        tau: List[float] = []
        for lag in valid_lags:
            diff = arr[lag:] - arr[:-lag]
            val = np.std(diff)
            if val <= 0 or np.isnan(val):
                continue
            tau.append(val)
        if len(tau) < 2:
            return np.nan
        return float(np.polyfit(np.log(valid_lags[: len(tau)]), np.log(tau), 1)[0])

    return series.rolling(window).apply(_hurst, raw=True)


def generate_feature_library(close: pd.Series) -> pd.DataFrame:
    features = pd.DataFrame(index=close.index)
    idx = close.index
    rets = close.pct_change()
    log_ret = np.log(close).diff()
    windows = [2, 3, 5, 7, 10, 14, 20, 30, 40, 42, 60, 90, 120, 180, 252]

    # Calendar features
    features["day_of_week"] = idx.dayofweek
    features["week_of_month"] = ((idx.day - 1) // 7 + 1).astype(float)
    features["week_of_year"] = idx.isocalendar().week.astype(float)
    features["month"] = idx.month
    features["quarter"] = idx.quarter
    features["is_month_start"] = idx.is_month_start.astype(float)
    features["is_month_end"] = idx.is_month_end.astype(float)
    features["n_day_in_month"] = idx.day.astype(float)
    features["turn_of_month"] = ((idx.day <= 3) | (idx.day >= 28)).astype(float)
    features["is_first_trading_day"] = idx.to_series().groupby(idx.to_period("M")).cumcount().eq(0).astype(float)
    features["summer_driving_season"] = idx.month.isin([5, 6, 7, 8]).astype(float)
    features["pre_opec_season_proxy"] = idx.month.isin([3, 6, 9, 11]).astype(float)
    features["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    features["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    features["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    features["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    features["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    features["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)

    for d in range(5):
        features[f"is_dow_{d}"] = (idx.dayofweek == d).astype(float)
    for m in range(1, 13):
        features[f"is_month_{m:02d}"] = (idx.month == m).astype(float)
    for q in range(1, 5):
        features[f"is_quarter_{q}"] = (idx.quarter == q).astype(float)

    # Price-derived and statistical features
    features["ret_1d"] = rets
    features["log_ret_1d"] = log_ret
    for lag in [1, 2, 3, 5, 10, 20]:
        features[f"ret_lag_{lag}"] = rets.shift(lag)
        features[f"log_ret_lag_{lag}"] = log_ret.shift(lag)

    for w in windows:
        ma = close.rolling(w).mean()
        ema = close.ewm(span=w, adjust=False).mean()
        vol = rets.rolling(w).std()
        high = close.rolling(w).max()
        low = close.rolling(w).min()
        ret_w = close.pct_change(w)
        compounded = (1 + rets).rolling(w).apply(np.prod, raw=True) - 1

        features[f"ret_{w}d"] = ret_w
        features[f"log_ret_sum_{w}"] = log_ret.rolling(w).sum()
        features[f"comp_ret_{w}"] = compounded
        features[f"vol_{w}"] = vol
        features[f"rv_{w}"] = vol * np.sqrt(TRADING_DAYS)
        features[f"vol_of_vol_{w}"] = vol.rolling(w).std()
        features[f"sma_ratio_{w}"] = close / (ma + EPS) - 1
        features[f"ema_ratio_{w}"] = close / (ema + EPS) - 1
        features[f"z_price_{w}"] = rolling_zscore(close, w)
        features[f"distance_high_{w}"] = close / (high + EPS) - 1
        features[f"distance_low_{w}"] = close / (low + EPS) - 1
        features[f"breakout_pos_{w}"] = (close - low) / (high - low + EPS)
        features[f"drawdown_{w}"] = close / (high + EPS) - 1
        features[f"momentum_vol_adj_{w}"] = ret_w / (vol * np.sqrt(w) + EPS)
        features[f"rolling_sharpe_{w}"] = rets.rolling(w).mean() / (vol + EPS)
        features[f"rolling_skew_{w}"] = rets.rolling(w).skew()
        features[f"rolling_kurt_{w}"] = rets.rolling(w).kurt()
        features[f"ac1_{w}"] = rets.rolling(w).corr(rets.shift(1))
        features[f"ac5_{w}"] = rets.rolling(w).corr(rets.shift(5))
        features[f"slope_{w}"] = rolling_slope(np.log(close), w)

        long_var = close.pct_change(w).rolling(max(w, 20)).var()
        short_var = rets.rolling(max(w, 20)).var()
        features[f"var_ratio_{w}"] = long_var / (w * short_var + EPS)
        features[f"vol_pct_{w}"] = vol.rolling(252).rank(pct=True)

    for w in [5, 10, 14, 20, 30, 60]:
        features[f"rsi_{w}"] = compute_rsi(close, w)

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    features["macd"] = macd
    features["macd_signal"] = macd_signal
    features["macd_hist"] = macd - macd_signal

    for w in [60, 120]:
        features[f"hurst_{w}"] = rolling_hurst(np.log(close), w)
        features[f"entropy_{w}"] = rolling_entropy(rets, w, bins=8)

    features["trend_strength_60"] = features["slope_60"].abs() / (features["rv_20"] + EPS)
    features["trend_strength_120"] = features["slope_120"].abs() / (features["rv_20"] + EPS)
    features["vol_regime_high"] = (features["vol_pct_20"] > 0.7).astype(float)
    features["vol_regime_low"] = (features["vol_pct_20"] < 0.3).astype(float)
    features["trend_regime_high"] = (
        features["trend_strength_60"] > features["trend_strength_60"].rolling(252).median()
    ).astype(float)

    return features.replace([np.inf, -np.inf], np.nan)


def build_feature_diagnostics(features: pd.DataFrame, rets: pd.Series) -> pd.DataFrame:
    target = rets.shift(-1)
    records = []
    for col in features.columns:
        x = features[col]
        aligned = pd.concat([x, target], axis=1).dropna()
        if len(aligned) < 252:
            continue
        ic = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        rolling_ic = aligned.iloc[:, 0].rolling(252).corr(aligned.iloc[:, 1]).dropna()
        if np.isnan(ic) or rolling_ic.empty:
            continue
        sign = np.sign(ic)
        if sign == 0:
            stability = np.nan
        else:
            stability = float((np.sign(rolling_ic) == sign).mean())
        t_stat = float(ic * np.sqrt(max(len(aligned) - 2, 1) / (1 - ic * ic + EPS)))
        records.append(
            {
                "feature": col,
                "ic": float(ic),
                "abs_ic": float(abs(ic)),
                "ic_tstat": t_stat,
                "ic_stability": stability,
                "coverage": float(len(aligned) / len(features)),
            }
        )
    return pd.DataFrame(records).sort_values("abs_ic", ascending=False)


def select_stable_features(features: pd.DataFrame, diagnostics: pd.DataFrame, cfg: ResearchConfig) -> pd.DataFrame:
    selected = diagnostics[
        (diagnostics["abs_ic"] >= cfg.min_abs_ic)
        & (diagnostics["ic_stability"] >= cfg.min_ic_stability)
    ]["feature"].tolist()

    if not selected:
        return features.copy()

    reduced = features[selected].copy()
    corr = reduced.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if (upper[col] > cfg.feature_corr_cap).any()]
    return reduced.drop(columns=to_drop)


def to_pnl(raw_signal: pd.Series, rets: pd.Series, tcost: float) -> Tuple[pd.Series, pd.Series]:
    signal = raw_signal.shift(1).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-2.0, 2.0)
    turnover = signal.diff().abs().fillna(0.0)
    pnl = signal * rets.fillna(0.0) - tcost * turnover
    return pnl, signal


def donchian_signal(close: pd.Series, window: int) -> pd.Series:
    upper = close.rolling(window).max().shift(1)
    lower = close.rolling(window).min().shift(1)
    sig = np.where(close > upper, 1.0, np.where(close < lower, -1.0, np.nan))
    return pd.Series(sig, index=close.index).ffill().fillna(0.0)


def add_strategy(
    name: str,
    raw_signal: pd.Series,
    rets: pd.Series,
    tcost: float,
    family: str,
    pnl_dict: Dict[str, pd.Series],
    signal_dict: Dict[str, pd.Series],
    family_map: Dict[str, str],
) -> None:
    if name in pnl_dict:
        return
    pnl, signal = to_pnl(raw_signal, rets, tcost)
    if pnl.std(skipna=True) <= 0:
        return
    pnl_dict[name] = pnl
    signal_dict[name] = signal
    family_map[name] = family


def build_strategy_library(
    close: pd.Series,
    rets: pd.Series,
    features: pd.DataFrame,
    cfg: ResearchConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    pnl_dict: Dict[str, pd.Series] = {}
    signal_dict: Dict[str, pd.Series] = {}
    family_map: Dict[str, str] = {}
    idx = close.index

    # Trend following
    for w in [5, 10, 20, 40, 60, 90, 120, 180, 252]:
        add_strategy(
            f"trend_mom_{w}",
            np.sign(close.pct_change(w)),
            rets,
            cfg.tcost,
            "trend_following",
            pnl_dict,
            signal_dict,
            family_map,
        )
        add_strategy(
            f"trend_slope_{w}",
            np.sign(features[f"slope_{w}"]),
            rets,
            cfg.tcost,
            "trend_following",
            pnl_dict,
            signal_dict,
            family_map,
        )
        add_strategy(
            f"trend_donchian_{w}",
            donchian_signal(close, w),
            rets,
            cfg.tcost,
            "trend_following",
            pnl_dict,
            signal_dict,
            family_map,
        )

    for short in [5, 10, 20, 30]:
        for long in [40, 60, 90, 120, 180, 252]:
            if short >= long:
                continue
            raw = np.sign(close.rolling(short).mean() - close.rolling(long).mean())
            add_strategy(
                f"trend_ma_cross_{short}_{long}",
                raw,
                rets,
                cfg.tcost,
                "trend_following",
                pnl_dict,
                signal_dict,
                family_map,
            )

    # Mean reversion
    for w in [5, 10, 20, 40, 60]:
        z = features[f"z_price_{w}"]
        for thr in [0.5, 1.0, 1.5, 2.0]:
            raw = pd.Series(np.where(z > thr, -1.0, np.where(z < -thr, 1.0, 0.0)), index=idx)
            add_strategy(
                f"mr_z_{w}_{thr:.1f}",
                raw,
                rets,
                cfg.tcost,
                "mean_reversion",
                pnl_dict,
                signal_dict,
                family_map,
            )

    for w in [5, 10, 14, 20, 30]:
        rsi = features[f"rsi_{w}"]
        for overbought, oversold in [(70, 30), (80, 20), (65, 35)]:
            raw = pd.Series(
                np.where(rsi > overbought, -1.0, np.where(rsi < oversold, 1.0, 0.0)),
                index=idx,
            )
            add_strategy(
                f"mr_rsi_{w}_{overbought}_{oversold}",
                raw,
                rets,
                cfg.tcost,
                "mean_reversion",
                pnl_dict,
                signal_dict,
                family_map,
            )

    for w in [2, 3, 5, 10, 20]:
        add_strategy(
            f"mr_st_rev_{w}",
            -np.sign(close.pct_change(w)),
            rets,
            cfg.tcost,
            "mean_reversion",
            pnl_dict,
            signal_dict,
            family_map,
        )

    # Volatility regime
    for base_w in [20, 60, 120]:
        base_mom = np.sign(close.pct_change(base_w))
        for vol_w in [10, 20, 60]:
            vol_pct = features[f"vol_pct_{vol_w}"]
            for q in [0.4, 0.5, 0.6]:
                low_vol_trend = pd.Series(np.where(vol_pct < q, base_mom, 0.0), index=idx)
                high_vol_mr = pd.Series(np.where(vol_pct > (1 - q), -base_mom, 0.0), index=idx)
                add_strategy(
                    f"vol_lowtrend_{base_w}_{vol_w}_{q:.1f}",
                    low_vol_trend,
                    rets,
                    cfg.tcost,
                    "volatility_regime",
                    pnl_dict,
                    signal_dict,
                    family_map,
                )
                add_strategy(
                    f"vol_highmr_{base_w}_{vol_w}_{q:.1f}",
                    high_vol_mr,
                    rets,
                    cfg.tcost,
                    "volatility_regime",
                    pnl_dict,
                    signal_dict,
                    family_map,
                )

    # Calendar/time-driven
    for d in range(5):
        for side in [-1.0, 1.0]:
            signal = pd.Series(np.where(idx.dayofweek == d, side, 0.0), index=idx)
            add_strategy(
                f"time_dow_{d}_{int(side)}",
                signal,
                rets,
                cfg.tcost,
                "time_calendar",
                pnl_dict,
                signal_dict,
                family_map,
            )

    for m in range(1, 13):
        for side in [-1.0, 1.0]:
            signal = pd.Series(np.where(idx.month == m, side, 0.0), index=idx)
            add_strategy(
                f"time_month_{m:02d}_{int(side)}",
                signal,
                rets,
                cfg.tcost,
                "time_calendar",
                pnl_dict,
                signal_dict,
                family_map,
            )

    turn_of_month = pd.Series(((idx.day <= 3) | (idx.day >= 28)).astype(float), index=idx)
    for side in [-1.0, 1.0]:
        add_strategy(
            f"time_turnmonth_{int(side)}",
            turn_of_month * side,
            rets,
            cfg.tcost,
            "time_calendar",
            pnl_dict,
            signal_dict,
            family_map,
        )

    summer = pd.Series(idx.month.isin([5, 6, 7, 8]).astype(float), index=idx)
    for side in [-1.0, 1.0]:
        add_strategy(
            f"time_summer_{int(side)}",
            summer * side,
            rets,
            cfg.tcost,
            "time_calendar",
            pnl_dict,
            signal_dict,
            family_map,
        )

    # Statistical structure
    for w in [20, 40, 60, 120]:
        mom = np.sign(close.pct_change(20))
        ac = features[f"ac1_{w}"]
        vr = features[f"var_ratio_{w}"]
        add_strategy(
            f"stat_ac_switch_{w}",
            pd.Series(np.where(ac > 0, mom, -mom), index=idx),
            rets,
            cfg.tcost,
            "stat_structure",
            pnl_dict,
            signal_dict,
            family_map,
        )
        add_strategy(
            f"stat_vr_{w}",
            pd.Series(np.where(vr > 1, 1.0, -1.0), index=idx),
            rets,
            cfg.tcost,
            "stat_structure",
            pnl_dict,
            signal_dict,
            family_map,
        )

    for w in [60, 120]:
        hurst = features[f"hurst_{w}"]
        entropy = features[f"entropy_{w}"]
        trend = np.sign(close.pct_change(20))
        entropy_med = entropy.rolling(252).median()
        add_strategy(
            f"stat_hurst_{w}",
            pd.Series(np.where(hurst > 0.55, trend, np.where(hurst < 0.45, -trend, 0.0)), index=idx),
            rets,
            cfg.tcost,
            "stat_structure",
            pnl_dict,
            signal_dict,
            family_map,
        )
        add_strategy(
            f"stat_entropy_{w}",
            pd.Series(np.where(entropy < entropy_med, trend, 0.0), index=idx),
            rets,
            cfg.tcost,
            "stat_structure",
            pnl_dict,
            signal_dict,
            family_map,
        )

    # Risk-managed alpha
    for w in [20, 60, 120]:
        signal = np.sign(close.pct_change(w))
        rv = features[f"rv_{w}"]
        for target in [0.10, 0.15, 0.20]:
            leverage = (target / (rv + EPS)).clip(lower=0.0, upper=2.0)
            add_strategy(
                f"risk_voltarget_{w}_{int(target*100)}",
                signal * leverage,
                rets,
                cfg.tcost,
                "risk_managed_alpha",
                pnl_dict,
                signal_dict,
                family_map,
            )

    dd = features["drawdown_120"]
    for cut in [-0.05, -0.10, -0.15]:
        trend = np.sign(close.pct_change(60))
        scale = np.where(dd < cut, 0.5, 1.0)
        add_strategy(
            f"risk_ddscale_{abs(int(cut*100))}",
            pd.Series(trend * scale, index=idx),
            rets,
            cfg.tcost,
            "risk_managed_alpha",
            pnl_dict,
            signal_dict,
            family_map,
        )

    for w in [20, 60, 120]:
        edge = rets.rolling(w).mean() / (rets.rolling(w).var() + EPS)
        add_strategy(
            f"risk_kellyproxy_{w}",
            edge.clip(-1.0, 1.0) * 5.0,
            rets,
            cfg.tcost,
            "risk_managed_alpha",
            pnl_dict,
            signal_dict,
            family_map,
        )

    # Hybrid and simple ML-like composites
    trend = np.sign(close.pct_change(60))
    z20 = features["z_price_20"]
    low_trend = features["trend_strength_60"] < features["trend_strength_60"].rolling(252).quantile(0.4)
    low_vol = features["vol_pct_20"] < 0.5
    high_vol = features["vol_pct_20"] > 0.7
    add_strategy(
        "hybrid_trend_lowvol",
        pd.Series(np.where(low_vol, trend, 0.0), index=idx),
        rets,
        cfg.tcost,
        "hybrid",
        pnl_dict,
        signal_dict,
        family_map,
    )
    add_strategy(
        "hybrid_mr_lowtrend",
        pd.Series(np.where(low_trend, np.where(z20 > 1.0, -1.0, np.where(z20 < -1.0, 1.0, 0.0)), 0.0), index=idx),
        rets,
        cfg.tcost,
        "hybrid",
        pnl_dict,
        signal_dict,
        family_map,
    )
    add_strategy(
        "hybrid_trend_highvol_off",
        pd.Series(np.where(high_vol, 0.0, trend), index=idx),
        rets,
        cfg.tcost,
        "hybrid",
        pnl_dict,
        signal_dict,
        family_map,
    )

    for w in [20, 40, 60]:
        model_score = (
            0.40 * features[f"momentum_vol_adj_{w}"]
            + 0.25 * features[f"z_price_{w}"] * -1
            + 0.20 * features[f"ac1_{w}"]
            + 0.15 * features[f"vol_pct_{w}"] * -1
        )
        add_strategy(
            f"ml_linear_score_{w}",
            np.sign(model_score),
            rets,
            cfg.tcost,
            "ml_like",
            pnl_dict,
            signal_dict,
            family_map,
        )
        for thr in [0.0, 0.25, 0.5]:
            raw = pd.Series(np.where(model_score > thr, 1.0, np.where(model_score < -thr, -1.0, 0.0)), index=idx)
            add_strategy(
                f"ml_linear_thr_{w}_{thr:.2f}",
                raw,
                rets,
                cfg.tcost,
                "ml_like",
                pnl_dict,
                signal_dict,
                family_map,
            )

    pnl_df = pd.DataFrame(pnl_dict).replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    signal_df = pd.DataFrame(signal_dict).reindex(columns=pnl_df.columns)
    return pnl_df, signal_df, family_map


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns.fillna(0.0)).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    return float(drawdown.min())


def avg_holding_period(signal: pd.Series) -> float:
    s = signal.fillna(0.0)
    group_id = (s != s.shift(1)).cumsum()
    lengths = s.groupby(group_id).size()
    states = s.groupby(group_id).first()
    active = lengths[states != 0]
    if active.empty:
        return np.nan
    return float(active.mean())


def compute_metrics(returns: pd.Series, signal: pd.Series | None = None) -> Dict[str, float]:
    r = returns.dropna()
    if r.empty:
        return {
            "sharpe": np.nan,
            "sortino": np.nan,
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan,
            "hit_rate": np.nan,
            "skew": np.nan,
            "kurtosis": np.nan,
            "turnover": np.nan,
            "avg_holding_period": np.nan,
            "rolling_sharpe_stability": np.nan,
        }

    mean = r.mean()
    std = r.std()
    downside_std = r[r < 0].std()
    sharpe = float(np.sqrt(TRADING_DAYS) * mean / (std + EPS))
    sortino = float(np.sqrt(TRADING_DAYS) * mean / (downside_std + EPS))
    ann_vol = float(std * np.sqrt(TRADING_DAYS))
    ann_return = float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1)
    mdd = max_drawdown(r)
    calmar = float(ann_return / (abs(mdd) + EPS))
    rolling_sharpe = np.sqrt(TRADING_DAYS) * r.rolling(126).mean() / (r.rolling(126).std() + EPS)
    rolling_sharpe_stability = float(rolling_sharpe.dropna().std()) if not rolling_sharpe.dropna().empty else np.nan

    metrics = {
        "sharpe": sharpe,
        "sortino": sortino,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "max_drawdown": mdd,
        "calmar": calmar,
        "hit_rate": float((r > 0).mean()),
        "skew": float(r.skew()),
        "kurtosis": float(r.kurtosis()),
        "turnover": np.nan,
        "avg_holding_period": np.nan,
        "rolling_sharpe_stability": rolling_sharpe_stability,
    }
    if signal is not None:
        metrics["turnover"] = float(signal.diff().abs().mean())
        metrics["avg_holding_period"] = avg_holding_period(signal)
    return metrics


def evaluate_strategy_library(
    pnl_df: pd.DataFrame, signal_df: pd.DataFrame, family_map: Dict[str, str]
) -> pd.DataFrame:
    rows = []
    for col in pnl_df.columns:
        metrics = compute_metrics(pnl_df[col], signal_df[col])
        metrics["strategy"] = col
        metrics["family"] = family_map.get(col, "unknown")
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)


def run_walkforward_grid(
    pnl_df: pd.DataFrame, cfg: ResearchConfig
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    rows = []
    best_payload: Dict[str, object] = {}
    best_sharpe = -np.inf

    for lb in cfg.lookback_grid:
        for rb in cfg.rebalance_grid:
            backtester = WalkforwardBacktester(
                pnl_df=pnl_df,
                lookback_period=f"{lb}M",
                rebalance_freq=f"{rb}M",
            )
            portfolio = backtester.run_backtest().dropna()
            metrics = compute_metrics(portfolio)
            metrics["lookback_m"] = lb
            metrics["rebalance_m"] = rb
            metrics["n_rebalances"] = len(backtester.rebalances)
            rows.append(metrics)
            if metrics["sharpe"] > best_sharpe:
                best_sharpe = metrics["sharpe"]
                best_payload = {
                    "lookback_m": lb,
                    "rebalance_m": rb,
                    "portfolio": portfolio,
                    "backtester": backtester,
                    "metrics": metrics,
                }

    grid = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    return grid, best_payload


def detect_regimes(features: pd.DataFrame) -> pd.DataFrame:
    regimes = pd.DataFrame(index=features.index)
    vol_rank = features["vol_pct_20"]
    trend_rank = features["trend_strength_60"].rank(pct=True)

    regimes["volatility_regime"] = pd.cut(
        vol_rank, bins=[-np.inf, 0.33, 0.66, np.inf], labels=["low_vol", "mid_vol", "high_vol"]
    )
    regimes["trend_regime"] = pd.cut(
        trend_rank, bins=[-np.inf, 0.33, 0.66, np.inf], labels=["weak_trend", "medium_trend", "strong_trend"]
    )
    regimes["autocorr_regime"] = np.where(
        features["ac1_60"] > 0.05, "positive_ac", np.where(features["ac1_60"] < -0.05, "negative_ac", "neutral_ac")
    )
    regimes["hurst_regime"] = np.where(
        features["hurst_120"] > 0.55,
        "persistent",
        np.where(features["hurst_120"] < 0.45, "mean_reverting", "random_walk"),
    )
    return regimes


def subperiod_breakdown(portfolio: pd.Series) -> pd.DataFrame:
    rows = []
    for year, grp in portfolio.dropna().groupby(portfolio.dropna().index.year):
        m = compute_metrics(grp)
        m["year"] = int(year)
        rows.append(m)
    return pd.DataFrame(rows).sort_values("year")


def block_bootstrap_sharpe(returns: pd.Series, n_sims: int = 300, block_size: int = 20) -> np.ndarray:
    arr = returns.dropna().to_numpy()
    n = len(arr)
    if n == 0:
        return np.array([])
    blocks = int(np.ceil(n / block_size))
    sharpes = []
    for _ in range(n_sims):
        idx = []
        for _ in range(blocks):
            start = np.random.randint(0, max(n - block_size, 1))
            idx.extend(range(start, min(start + block_size, n)))
        sim = arr[np.array(idx[:n])]
        sharpe = np.sqrt(TRADING_DAYS) * np.mean(sim) / (np.std(sim) + EPS)
        sharpes.append(sharpe)
    return np.array(sharpes)


def random_entry_baseline(rets: pd.Series, tcost: float, n_sims: int = 300, flip_prob: float = 0.08) -> np.ndarray:
    arr = rets.fillna(0.0).to_numpy()
    n = len(arr)
    sharpes = []
    for _ in range(n_sims):
        signal = np.ones(n)
        signal[0] = np.random.choice([-1.0, 1.0])
        flips = np.random.rand(n) < flip_prob
        for i in range(1, n):
            signal[i] = -signal[i - 1] if flips[i] else signal[i - 1]
        turnover = np.abs(np.diff(signal, prepend=signal[0]))
        pnl = signal * arr - tcost * turnover
        sharpe = np.sqrt(TRADING_DAYS) * np.mean(pnl) / (np.std(pnl) + EPS)
        sharpes.append(sharpe)
    return np.array(sharpes)


def noise_injection_test(
    signal: pd.Series,
    rets: pd.Series,
    tcost: float,
    noise_levels: Iterable[float] = (0.05, 0.1, 0.2),
    n_sims: int = 100,
) -> pd.DataFrame:
    rows = []
    base = signal.fillna(0.0).to_numpy().copy()
    ret_arr = rets.fillna(0.0).to_numpy()
    n = len(base)
    for nl in noise_levels:
        sharpes = []
        for _ in range(n_sims):
            noisy = base.copy()
            mask = np.random.rand(n) < nl
            noisy[mask] *= -1
            turnover = np.abs(np.diff(noisy, prepend=noisy[0]))
            pnl = noisy * ret_arr - tcost * turnover
            sharpe = np.sqrt(TRADING_DAYS) * np.mean(pnl) / (np.std(pnl) + EPS)
            sharpes.append(sharpe)
        rows.append(
            {
                "noise_level": nl,
                "sharpe_mean": float(np.mean(sharpes)),
                "sharpe_p05": float(np.percentile(sharpes, 5)),
                "sharpe_p50": float(np.percentile(sharpes, 50)),
                "sharpe_p95": float(np.percentile(sharpes, 95)),
            }
        )
    return pd.DataFrame(rows)


def build_ensemble(
    pnl_df: pd.DataFrame, strategy_metrics: pd.DataFrame, method: str = "inverse_vol", top_n: int = 8
) -> pd.Series:
    names = strategy_metrics.head(top_n)["strategy"].tolist()
    subset = pnl_df[names].copy()
    if method == "inverse_vol":
        vol = subset.rolling(63).std()
        weights = 1 / (vol + EPS)
        weights = weights.div(weights.sum(axis=1), axis=0)
    else:
        sharpe = subset.rolling(126).mean() / (subset.rolling(126).std() + EPS)
        sharpe = sharpe.clip(lower=0.0)
        weights = sharpe.div(sharpe.sum(axis=1), axis=0)
    return (weights.shift(1).fillna(0.0) * subset).sum(axis=1)


def summarize_rebalances(backtester: WalkforwardBacktester) -> pd.DataFrame:
    rows = []
    for item in backtester.rebalances:
        best = item["best_strategy"]
        scores = item["strategy_scores"]
        rows.append(
            {
                "start_date": item["start_date"],
                "end_date": item["end_date"],
                "lookback_start": item["lookback_start"],
                "lookback_end": item["lookback_end"],
                "best_strategy": best,
                "best_score": float(scores[best]),
            }
        )
    return pd.DataFrame(rows)


def run_research(cfg: ResearchConfig) -> Dict[str, object]:
    np.random.seed(cfg.random_seed)
    close, rets = load_price_data(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        features = generate_feature_library(close)
    diagnostics = build_feature_diagnostics(features, rets)
    selected_features = select_stable_features(features, diagnostics, cfg)

    pnl_df, signal_df, family_map = build_strategy_library(close, rets, features, cfg)
    if len(pnl_df.columns) < 100:
        raise ValueError(f"Only {len(pnl_df.columns)} strategies generated; expected at least 100.")

    strategy_metrics = evaluate_strategy_library(pnl_df, signal_df, family_map)
    wf_grid, best_wf = run_walkforward_grid(pnl_df, cfg)
    regimes = detect_regimes(features)
    subperiod = subperiod_breakdown(best_wf["portfolio"])

    top_strategy = strategy_metrics.iloc[0]["strategy"]
    top_signal = signal_df[top_strategy]
    top_metrics = compute_metrics(pnl_df[top_strategy], top_signal)

    ensemble_iv = build_ensemble(pnl_df, strategy_metrics, method="inverse_vol", top_n=8)
    ensemble_sh = build_ensemble(pnl_df, strategy_metrics, method="rolling_sharpe", top_n=8)

    ensemble_metrics = pd.DataFrame(
        [
            {"ensemble": "inverse_vol", **compute_metrics(ensemble_iv)},
            {"ensemble": "rolling_sharpe", **compute_metrics(ensemble_sh)},
        ]
    )

    mc_sharpes = block_bootstrap_sharpe(best_wf["portfolio"])
    random_baseline = random_entry_baseline(rets, cfg.tcost)
    noise_test = noise_injection_test(top_signal, rets, cfg.tcost)

    robustness = pd.DataFrame(
        [
            {
                "test": "walkforward_bootstrap",
                "sharpe_mean": float(np.mean(mc_sharpes)),
                "sharpe_p05": float(np.percentile(mc_sharpes, 5)),
                "sharpe_p50": float(np.percentile(mc_sharpes, 50)),
                "sharpe_p95": float(np.percentile(mc_sharpes, 95)),
            },
            {
                "test": "random_entry_baseline",
                "sharpe_mean": float(np.mean(random_baseline)),
                "sharpe_p05": float(np.percentile(random_baseline, 5)),
                "sharpe_p50": float(np.percentile(random_baseline, 50)),
                "sharpe_p95": float(np.percentile(random_baseline, 95)),
            },
        ]
    )

    best_rebalances = summarize_rebalances(best_wf["backtester"])

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_dir / "features.csv")
    diagnostics.to_csv(output_dir / "feature_diagnostics.csv", index=False)
    selected_features.to_csv(output_dir / "selected_features.csv")
    strategy_metrics.to_csv(output_dir / "strategy_metrics.csv", index=False)
    wf_grid.to_csv(output_dir / "walkforward_grid.csv", index=False)
    best_wf["portfolio"].to_frame("walkforward_portfolio").to_csv(output_dir / "walkforward_portfolio.csv")
    best_rebalances.to_csv(output_dir / "rebalance_details.csv", index=False)
    regimes.to_csv(output_dir / "regimes.csv")
    subperiod.to_csv(output_dir / "subperiod_breakdown.csv", index=False)
    robustness.to_csv(output_dir / "robustness_summary.csv", index=False)
    noise_test.to_csv(output_dir / "noise_injection_test.csv", index=False)
    ensemble_metrics.to_csv(output_dir / "ensemble_metrics.csv", index=False)

    summary = {
        "n_features": int(features.shape[1]),
        "n_selected_features": int(selected_features.shape[1]),
        "n_strategies": int(pnl_df.shape[1]),
        "top_strategy": str(top_strategy),
        "top_strategy_metrics": top_metrics,
        "best_walkforward_config": {
            "lookback_m": best_wf["lookback_m"],
            "rebalance_m": best_wf["rebalance_m"],
            **best_wf["metrics"],
        },
        "output_dir": str(output_dir.resolve()),
    }
    return summary


if __name__ == "__main__":
    config = ResearchConfig()
    report = run_research(config)
    print("Research run complete.")
    print(f"Features generated: {report['n_features']}")
    print(f"Selected stable features: {report['n_selected_features']}")
    print(f"Strategies generated: {report['n_strategies']}")
    print(f"Top in-sample strategy: {report['top_strategy']}")
    print(f"Best walkforward Sharpe: {report['best_walkforward_config']['sharpe']:.3f}")
    print(
        f"Best walkforward config: lookback={report['best_walkforward_config']['lookback_m']}M, "
        f"rebalance={report['best_walkforward_config']['rebalance_m']}M"
    )
    print(f"Output folder: {report['output_dir']}")
