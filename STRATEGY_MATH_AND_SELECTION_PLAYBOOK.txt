# Strategy Math and Selection Playbook

Date: 2026-02-19
Repository: `cand-project-2`
Audience: Client, PM, and quant reviewer

---

## 1) What this document covers

This document explains:
- The math behind the strategy engine.
- How signals become returns (including transaction cost treatment).
- How the system decides what is "best" at each stage.
- How final submission strategies are selected.
- What the top strategies mean in plain English and technical terms.

This is intentionally dual-language:
- Plain-English explanation for non-technical readers.
- Technical explanation for quant review.

---

## 2) Plain-English summary (fast read)

The system works in layers:

1. Build many candidate strategies from price behavior, calendar effects, volatility regimes, and ML-light signals.
2. Convert each strategy signal into daily PnL after transaction costs.
3. Score each strategy using risk-adjusted metrics (Sharpe, drawdown, etc.).
4. Stress-test strategies with walk-forward, parameter stability, IS/OOS consistency, and regime behavior.
5. For final submission, combine selected strategy-signals into model portfolios (Linear, GB proxy, Ensemble), then check if they beat Brent out-of-sample.
6. Generate fixed Top-5 proposal candidates for client presentation.

Key idea:
- The system does not rely on one metric only. It combines return quality, stability, trading realism, and benchmark-relative performance.

---

## 3) Data and return construction math

## 3.1 Input series
- Price series: Brent total return proxy from `brent_index.xlsx`.
- Daily return:
  - `r_t = P_t / P_{t-1} - 1`

## 3.2 Signal to PnL with cost

Plain English:
- Use yesterday's signal today (no lookahead).
- Pay cost when position changes.

Technical:
- Signal lagging:
  - `s_t = clip(raw_signal_{t-1}, -2, 2)`
- Turnover:
  - `turnover_t = |s_t - s_{t-1}|`
- PnL:
  - `pnl_t = s_t * r_t - c * turnover_t`
- Transaction cost in code:
  - `c = 0.00015` (0.015% per trade unit)

This is the core realism constraint across strategies.

---

## 4) Core performance metrics math

From `compute_metrics`:

- Annualized Sharpe:
  - `Sharpe = sqrt(252) * mean(r) / std(r)`
- Annualized Sortino:
  - `Sortino = sqrt(252) * mean(r) / std(r | r<0)`
- Annualized volatility:
  - `AnnVol = std(r) * sqrt(252)`
- Annualized return (geometric):
  - `AnnReturn = (prod(1+r))^(252/N) - 1`
- Max drawdown:
  - `MDD = min(Wealth / CumMax(Wealth) - 1)`
- Calmar:
  - `Calmar = AnnReturn / |MDD|`

Additional diagnostics:
- Hit rate, skew, kurtosis.
- Turnover and average holding period (if signal available).
- Rolling Sharpe stability proxy.

---

## 5) How candidate strategies are built

The factory creates a wide universe across families.

## 5.1 Family examples (plain English + technical)

1. `trend_following`
- Plain English: follow direction if market keeps moving.
- Technical: sign of multi-horizon momentum, slope, Donchian breakout, MA crosses.

2. `mean_reversion`
- Plain English: fade short-term overreaction.
- Technical: z-score/RSI threshold contrarian signals, short-term reversal signs.

3. `volatility_regime`
- Plain English: use different behavior in calm vs stressed markets.
- Technical: condition trend/MR logic on volatility percentile states.

4. `time_calendar` and `event_flow`
- Plain English: exploit day/month/turn-of-period behavior.
- Technical: DOW/month/turn-month/quarter-start-end indicator-driven signals.

5. `stat_structure`
- Plain English: use distribution/autocorrelation structure.
- Technical: variance-ratio, autocorr switches, Hurst/entropy regime switches.

6. `risk_managed_alpha` and `risk_overlay`
- Plain English: keep edge but scale risk.
- Technical: vol-target sizing, drawdown cutback, risk budget multipliers.

7. `ml_like` and `ml_light`
- Plain English: interpretable pseudo-ML or light ML voting/classification.
- Technical: linear feature scoring, logistic/ridge proxy, random-forest voting proxy.

8. `time_series` and `microstructure_proxy`
- Plain English: use multi-horizon trend confirmation and short-lag micro patterns.
- Technical: dual-momentum agreement, EWM edge, lag reversal/momentum, post-shock effects.

---

## 6) How "best" is found in each layer

## 6.1 Discovery layer (broad ranking)

Plain English:
- Rank all strategies by performance metrics (especially Sharpe), while also viewing trade realism and risk.

Technical:
- `evaluate_strategy_library(...)` computes metric vector per strategy.
- Tables can be sorted by Sharpe, activity-adjusted Sharpe, return, drawdown, trade frequency, etc.

## 6.2 Walk-forward layer (adaptivity test)

Plain English:
- Repeatedly train on the past, pick best strategy, test in next period.

Technical:
- Grid over lookback and rebalance frequencies.
- For each config, stitched OOS portfolio is built and scored.
- Best walk-forward config chosen by highest Sharpe in grid.

## 6.3 Survival layer (robustness score)

Plain English:
- A strategy must not only perform; it must survive stability and quality checks.

Technical scoring (0 to 100):
- Intermediate components:
  - walk-forward quality
  - parameter stability
  - trade consistency
  - exposure stability
  - regime diversification
- Weighted raw score:
  - `raw = 0.28*WF + 0.22*Param + 0.18*Trade + 0.14*Exposure + 0.18*Regime`
- Penalties:
  - `0.06 * red_flag_count`
  - +0.12 for HIGH overfit risk, +0.05 for MEDIUM
- Final:
  - `final_robustness_score = 100 * clip(raw - penalty, 0, 1)`

Hard rejections include low trades, low exposure, flat periods, parameter instability, large IS/OOS gap, and optional missing economic logic.

## 6.4 Final submission feature ranking

Plain English:
- From eligible robust strategies, rank features using OOS-oriented quality + consistency + activity.

Technical:
- `sample_sharpe = out_sample_sharpe (fallback sharpe)`
- `walkforward_ann_return = out_sample_cagr (fallback ann_return)`
- `selection_score = 0.50*z(sample_sharpe) + 0.35*consistency + 0.15*z(trades_per_year)`
- Default basket:
  - top 3 by walkforward annualized return from different families when possible.

---

## 7) Final model math (submission layer)

Given selected feature-signals `X` and target next-day Brent return `y`:
- Target: `y_t = r_{t+1}`
- Train/test split by `split_ratio` (chronological).

## 7.1 Linear Regression (ridge)
- Standardize features using train mean/std.
- Coefficients:
  - `beta = (X'X + alpha*I)^(-1) X'y`
- Score:
  - `score_t = X_t * beta`

## 7.2 Gradient Boosting proxy
- Stagewise additive fit to residuals.
- At each round, choose feature with strongest residual explanatory step.
- Add weighted step with `learning_rate`.

## 7.3 Signal transform
- Convert score to bounded position:
  - `signal_t = tanh(score_t / std_train(score))`

## 7.4 Quarterly OOS rebalance

Plain English:
- During test period, models are re-fit each quarter using only data available up to quarter start.

Technical:
- For each OOS quarter segment:
  - fit ridge + GB on data up to segment start
  - predict segment signals
- Ensemble is average of quarterly ridge and quarterly GB signals.

## 7.5 Final model ranking and Brent checks
- OOS/full excess return vs Brent.
- OOS/full excess Sharpe vs Brent.
- Beat flag requires both OOS return and OOS Sharpe to exceed Brent.

---

## 8) How fixed Top-5 proposals are found

This is the current client-facing approach.

## 8.1 Candidate feature-combo generation
- Seed top features by walk-forward annualized return + quality scores.
- Build multiple combo sizes (3 to 9 features).
- Encourage family diversity.
- Add deterministic and random-combination candidates.
- Heuristic combo ranking:
  - average WF annual return
  - average sample Sharpe
  - max WF annual return
  - family diversity

## 8.2 Hyperparameter scan
For each candidate combo, evaluate fixed grid:
- split ratio in `{0.70, 0.75}`
- ridge alpha in `{3.0, 6.0}`
- GB rounds in `{80, 120}`
- GB learning rate `0.08`

## 8.3 Candidate model set
Only these are considered for proposal winner per combo:
- `Linear Regression (Quarterly Rebalanced OOS)`
- `Gradient Boosting Proxy (Quarterly Rebalanced OOS)`
- `Ensemble (Quarterly Rebalanced OOS)`

## 8.4 Proposal ranking key
Sort by:
1. `beats_brent_oos`
2. `oos_excess_ann_return_vs_brent`
3. `out_sample_ann_return`
4. `out_sample_sharpe`
5. `full_sample_ann_return`

Then keep top 5 (with dedupe rules).

---

## 9) Current best strategies snapshot (as of 2026-02-19 run)

## 9.1 Base universe top examples (by Sharpe)
- `vol_lowtrend_20_10_0.4` (volatility regime)
- `time_dow_3_1` (calendar)
- `vol_lowtrend_20_10_0.5` (volatility regime)
- `ts_drawdown_vol_gate_20` (time series)

Interpretation (plain English):
- Current winners combine low-vol trend following, calendar edge, and risk-gated trend behavior.

## 9.2 Fixed Top-5 submission proposals (current)
Dominant feature set appears repeatedly:
- `mr_st_rev_10`
- `ml_light_random_forest_vote`
- `micro_lag_revert_2`

Top model family currently selected:
- `Linear Regression (Quarterly Rebalanced OOS)`

Interpretation:
- A blended feature triad (short-term reversion + ML-light vote + micro lag reversal) is currently strongest under the present OOS benchmark criteria.

---

## 10) Easy explanation of what these top features mean

1. `mr_st_rev_10`
- Easy: if market moved one way recently (10-day), fade it expecting snapback.
- Technical: `-sign(pct_change(10))` style short-horizon contrarian component.

2. `ml_light_random_forest_vote`
- Easy: many simple rule stumps vote up/down, then we follow majority confidence.
- Technical: randomized threshold voting over standardized feature subset, sign thresholding.

3. `micro_lag_revert_2`
- Easy: very short-lag move tends to mean-revert after 2-day lag context.
- Technical: contrarian sign of lagged return microstructure proxy.

Why this mix can work:
- It combines different alpha mechanisms with low design complexity and good OOS benchmark fit.

---

## 11) What to tell a client in one minute

"We generate a large cross-family strategy universe, convert all signals to cost-adjusted returns, then score them with risk-adjusted and robustness criteria. For final submission, we combine selected robust signals into transparent models and require out-of-sample benchmark checks vs Brent. We also keep a fixed Top-5 proposal set with explicit features and hyperparameters so recommendations are stable and auditable."

---

## 12) Caveats

1. Robustness hard filters are currently very strict; in some runs all candidates can be flagged rejected.
2. Top-5 proposal diversity can be limited if one feature bundle dominates.
3. All numbers are data-and-period dependent; this is a process, not a guaranteed future outcome.

---

## 13) Where these formulas live in code

- Signal/PnL + base metrics: `research_engine.py`
- Survival robustness score and penalties: `robustness_engine/engine.py`
- Rolling walk-forward score formula: `validation_framework/walk_forward.py`
- Final submission ranking/model/proposal logic: `dashboard.py`

This document is the methodology companion to:
- `PRODUCT_360_ANALYSIS.md`
- `GIT_CHANGE_STORY.md`
- `PROJECT_ORIGIN_AND_PRE_PUSH_TRACE.md`
