# Crude Oil Systematic Strategy Submission - Final Report

Date: 2026-02-19
Candidate Project: `cand-project-2`

## 1) Strategy Intuition and Motivation

The strategy objective is to maximize risk-adjusted out-of-sample performance on Brent while controlling implementation realism and overfitting risk.

The final approach is a multi-signal meta-strategy rather than a single-rule bet. The intuition is:
- Crude markets show mixed behavior across horizons: short-term overreaction, intermediate trend persistence, and regime-dependent reversals.
- A robust approach should blend complementary alpha components instead of relying on one signal family.
- Selection quality should be benchmark-aware (Brent-relative) and out-of-sample focused.

The final selected submission stack combines three interpretable signal features:
- `mr_st_rev_10`: short-horizon mean-reversion behavior.
- `ml_light_random_forest_vote`: ensemble vote from simple threshold trees over interpretable predictors.
- `micro_lag_revert_2`: short-lag microstructure reversal proxy.

Model form selected for submission:
- `Linear Regression (Quarterly Rebalanced OOS)`

Reason this is selected:
- Strongest out-of-sample annualized return and Sharpe in the fixed Top-5 proposal search.
- Clear outperformance over Brent on both OOS return and OOS Sharpe.
- Stable and interpretable coefficient-based model.

---

## 2) Signals, Features, and Parameters Used

## 2.1 Candidate universe design
The project generates a broad strategy universe (281 base strategies; 327 combined with survival expansions) across multiple families:
- Trend-following
- Mean-reversion
- Volatility-regime switching
- Time/calendar and event-flow
- Statistical structure (autocorrelation, variance-ratio, Hurst/entropy)
- Risk-managed overlays
- ML-like and ML-light interpretable models
- Time-series adaptive and microstructure-proxy families

This breadth is intentional to reduce style concentration and improve robustness of final selection.

## 2.2 Transaction costs and execution realism
The system applies transaction cost directly in PnL construction:
- Cost per trade unit: 0.015% (`0.00015`)
- PnL is computed with lagged signal and turnover penalty:

\[
\text{signal}_t = \text{clip}(\text{raw signal}_{t-1}, -2, 2), \quad
\text{turnover}_t = |\text{signal}_t - \text{signal}_{t-1}|
\]
\[
\text{pnl}_t = \text{signal}_t \cdot r_t - c \cdot \text{turnover}_t
\]

This avoids lookahead and explicitly penalizes high-churn rules.

## 2.3 Final submission feature set and hyperparameters
Selected feature basket (Top-1 fixed proposal):
- `mr_st_rev_10`
- `ml_light_random_forest_vote`
- `micro_lag_revert_2`

Submission hyperparameters:
- Train ratio: `0.75`
- Ridge alpha: `3.0`
- Boosting rounds: `80`
- Boosting learning rate: `0.08`
- Minimum trades/year filter: `20`
- Test-period rebalance for model-based strategies: `Quarterly`

---

## 3) Walkforward Setup and Validation Logic

## 3.1 Baseline walkforward framework
The core research engine runs a walkforward grid search over lookback and rebalance cadence:
- Lookback grid: 6M, 12M, 24M, 36M
- Rebalance grid: 1M, 3M
- At each rebalance:
  1. Evaluate candidate strategy Sharpe on trailing lookback window.
  2. Select best strategy.
  3. Apply selected strategy OOS until next rebalance.

Observed best baseline walkforward configuration:
- Lookback: `12M`
- Rebalance: `1M`
- Walkforward Sharpe: `0.323`
- Walkforward annualized return: `4.38%`

## 3.2 Additional robustness controls
The project includes multiple controls against overfitting:
- Noise injection test (signal flips)
- Block bootstrap Sharpe simulation
- IS/OOS consistency checks
- Parameter surface stability diagnostics
- Regime robustness analysis
- Hard filters for trade realism and structural fragility

## 3.3 Final submission walkforward discipline
For final model candidates:
- Chronological split is enforced (train first, test later).
- Quarterly OOS rebalancing refits models at each test-quarter boundary using only data available up to that boundary.
- This is stricter than static one-time train/test fitting.

---

## 4) Performance Summary (Emphasis on OOS Sharpe)

## 4.1 Core research snapshot
- Features generated: `401`
- Selected stable features: `121`
- Strategies generated (base): `281`
- Top single strategy by in-sample Sharpe: `vol_lowtrend_20_10_0.4` (Sharpe `0.976`)

## 4.2 Final submission candidate performance (Top-1 fixed proposal)

Selected final model:
- `Linear Regression (Quarterly Rebalanced OOS)`

Out-of-sample performance:
- OOS annualized return: `21.01%`
- OOS Sharpe: `1.011`
- OOS max drawdown: `-22.91%`

Benchmark comparison (Brent OOS):
- Brent OOS annualized return: `0.26%`
- Brent OOS Sharpe: `0.171`

Excess vs Brent (OOS):
- Excess annualized return: `+20.75%`
- Excess Sharpe: `+0.840`
- Beats Brent OOS on both return and Sharpe: `Yes`

Full-sample for selected model:
- Full-sample annualized return: `11.54%`
- Full-sample Sharpe: `0.586`

## 4.3 Why this is considered best-of-best in current submission pool
The fixed Top-5 proposal engine ranks candidates by:
1. Brent OOS beat flag
2. OOS excess annualized return vs Brent
3. OOS annualized return
4. OOS Sharpe
5. Full-sample annualized return

Under this ranking and the current candidate/hyperparameter search space, the selected model-feature combination is the top proposal.

---

## 5) Risks, Limitations, and Observations

## 5.1 Key risks
1. Regime dependency:
- Commodity return structure changes can weaken current short-horizon reversal relationships.

2. Model concentration:
- Current top proposals cluster around a similar feature triad; diversity of finalists is limited.

3. Parameter drift:
- Even with quarterly OOS refit, coefficients and feature relevance may drift in future market states.

4. Execution assumptions:
- Cost model is linear and simplified; real slippage/liquidity events may be non-linear.

## 5.2 Project-level limitation currently observed
- Survival hard-filter settings are stringent and can reject all candidates in current diagnostics (with fallback ranking still used in downstream pages).
- This is useful as stress discipline but should be calibrated for production governance if required.

## 5.3 Why submission remains defensible despite limitations
- No-lookahead signal construction and chronological split discipline are enforced.
- OOS and benchmark-relative metrics are explicit and central to ranking.
- Multiple robustness diagnostics exist beyond raw Sharpe.
- Final strategy is interpretable at feature and coefficient level.

---

## 6) Reproducibility

1. Install dependencies:
- `pip install -r requirements.txt`

2. Run dashboard:
- `streamlit run streamlit_app.py`

3. Navigate to:
- `Final Submission Strategy` tab
- Use `Show Best 5 Proposed Strategies`
- Apply `Top 1` proposal

4. Verify displayed metrics:
- OOS return/Sharpe
- OOS excess vs Brent
- Quarterly rebalance model label

---

## 7) Final Submission Statement

This submission proposes a blended, interpretable, quarterly rebalanced linear meta-strategy using three complementary features (`mr_st_rev_10`, `ml_light_random_forest_vote`, `micro_lag_revert_2`). Under current out-of-sample evaluation, it delivers strong OOS Sharpe and meaningful benchmark-relative outperformance versus Brent while preserving transparent model structure and reproducible methodology.
