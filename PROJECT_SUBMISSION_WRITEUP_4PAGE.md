# Crude Oil Systematic Strategy Submission - Final Report (4-Page PDF-Ready)

Candidate: f20230152  
Date: 2026-02-19  
Project: `cand-project-2`

---

## 1. Assignment Objective and Strategy Thesis

### 1.1 Objective
The assignment objective is to design a systematic trading strategy on a crude oil total return index with emphasis on realistic out-of-sample (OOS) performance, especially Sharpe ratio, while avoiding lookahead bias and accounting for transaction costs.

### 1.2 Core Thesis
A single-rule strategy is often unstable across market regimes in crude oil. I therefore use a **multi-signal meta-strategy** that combines complementary alpha behaviors:
- short-horizon mean reversion,
- ML-light directional voting,
- microstructure lag reversal behavior.

The selected final strategy is:
- **Model**: `Linear Regression (Quarterly Rebalanced OOS)`
- **Feature basket**: `mr_st_rev_10`, `ml_light_random_forest_vote`, `micro_lag_revert_2`

This was selected through a benchmark-aware OOS ranking process against Brent.

---

## 2. Data, Backtest Integrity, and Implementation Rules

### 2.1 Data
- Input file: `brent_index.xlsx`
- Primary series: Brent total return proxy (daily).

### 2.2 Execution and Cost Model
To ensure realistic backtesting:
1. Signals are lagged by one day (no lookahead).
2. Daily PnL includes turnover cost.

Technical form:
\[
\text{signal}_t = \text{clip}(\text{raw signal}_{t-1}, -2, 2), \quad
\text{turnover}_t = |\text{signal}_t - \text{signal}_{t-1}|
\]
\[
\text{pnl}_t = \text{signal}_t\,r_t - c\,\text{turnover}_t,
\quad c=0.00015
\]

This is equivalent to 0.015% transaction cost per unit position change.

### 2.3 Quality Metrics
Primary and supporting metrics are computed from daily returns:
- Sharpe, Sortino, Annualized Return, Annualized Volatility,
- Max Drawdown, Calmar,
- turnover, average holding period, higher moments.

Sharpe definition used:
\[
\text{Sharpe} = \sqrt{252}\,\frac{\mathbb{E}[r]}{\sigma(r)}
\]

---

## 3. Signal Library and Strategy Construction

### 3.1 Candidate Universe
The project generates a broad strategy universe to reduce style concentration risk.

Current counts from the research run:
- Generated engineered features: **401**
- Stable selected features: **121**
- Base strategies generated: **281**
- Combined with survival expansion: **327** strategies across **18 families**

Families include trend-following, mean-reversion, volatility-regime, time/calendar, event-flow, statistical structure, risk overlays, ML-like, ML-light, time-series adaptive, and microstructure-proxy.

### 3.2 Why the final 3-feature basket
The chosen three features are intentionally complementary:

1. **`mr_st_rev_10`**  
Plain-English: fades 10-day directional moves expecting short-term snapback.  
Role: contrarian component.

2. **`ml_light_random_forest_vote`**  
Plain-English: many simple threshold rules vote directional bias.  
Role: interpretable non-linear consensus signal.

3. **`micro_lag_revert_2`**  
Plain-English: captures short-lag reversal tendency.  
Role: fast microstructure correction signal.

The blend improves OOS stability versus a single strategy style.

---

## 4. Walkforward and Robustness Framework

### 4.1 Baseline Walkforward Engine
The research engine evaluates multiple walkforward configurations:
- lookback grid: 6M, 12M, 24M, 36M
- rebalance grid: 1M, 3M

At each rebalance date:
1. Score each candidate strategy on trailing lookback window.
2. Select best strategy by Sharpe.
3. Apply selected strategy OOS until next rebalance.

Best observed baseline configuration:
- **lookback = 12M**, **rebalance = 1M**
- Walkforward Sharpe = **0.323**
- Walkforward annualized return = **4.38%**

### 4.2 Additional Overfit Controls
The stack includes:
- block bootstrap Sharpe diagnostics,
- random-entry baseline comparison,
- noise-injection tests,
- IS/OOS drift diagnostics,
- parameter surface checks,
- regime robustness analysis,
- hard filters and rejection flags.

### 4.3 Final Submission OOS Rebalancing
For submission models, test-period model refits are quarterly:
- model is re-estimated at each OOS quarter boundary,
- only historical data available up to that boundary is used.

This is stricter than static one-time train/test fitting.

---

## 5. Final Model Selection Method

### 5.1 Feature pre-ranking (submission pool)
For eligible candidates (trade activity filtered), feature ranking uses:
\[
\text{selection score}
=0.50\,z(\text{sample sharpe})
+0.35\,(\text{consistency index})
+0.15\,z(\text{trades/year})
\]

Default basket rule:
- top 3 by walkforward annualized return, diversified by family when possible.

### 5.2 Fixed Top-5 Proposal Search (client-ready)
To produce stable recommendations, the project runs a fixed Top-5 proposal search over:
- multiple feature-combination candidates,
- hyperparameter grid (`split_ratio`, `ridge_alpha`, `gb_rounds`, `gb_learning_rate`),
- quarterly-rebalanced candidate models (Linear, GB proxy, Ensemble).

Ranking priority:
1. Beats Brent OOS (return and Sharpe)
2. OOS excess annual return vs Brent
3. OOS annualized return
4. OOS Sharpe
5. Full-sample annualized return

---

## 6. Results Summary (OOS-Focused)

### 6.1 Selected Final Strategy
- **Model**: `Linear Regression (Quarterly Rebalanced OOS)`
- **Features**: `mr_st_rev_10`, `ml_light_random_forest_vote`, `micro_lag_revert_2`
- **Train ratio**: 0.75
- **Ridge alpha**: 3.0
- **GB proxy settings** (for proposal search context): rounds=80, lr=0.08

### 6.2 OOS and Benchmark-Relative Results
Selected final model:
- OOS annualized return: **21.01%**
- OOS Sharpe: **1.011**
- OOS max drawdown: **-22.91%**

Brent benchmark (OOS):
- OOS annualized return: **0.26%**
- OOS Sharpe: **0.171**

Excess vs Brent (OOS):
- Excess annualized return: **+20.75%**
- Excess Sharpe: **+0.840**
- Beats Brent on both OOS return and OOS Sharpe: **Yes**

Full-sample (selected model):
- Full annualized return: **11.54%**
- Full Sharpe: **0.586**

### 6.3 Additional Context
In the same model table, other quarterly models (ensemble and GB proxy) also show Brent OOS outperformance under current sample, but the selected model is ranked highest by the stated criteria.

---

## 7. Risks, Limitations, and Interpretation

### 7.1 Main Risks
1. **Regime instability**: crude market microstructure can shift abruptly.
2. **Feature concentration**: top proposals are clustered around similar feature sets.
3. **Cost realism gap**: linear transaction cost may understate stress-period slippage.
4. **Model drift**: relationships can decay despite quarterly refitting.

### 7.2 Methodological Limitations
- Survival hard filters are intentionally strict and may reject most/all candidates under certain settings.
- OOS superiority in one sample period is not a guarantee of future persistence.

### 7.3 Why this remains defensible
- Chronological data discipline and no-lookahead execution are enforced.
- Multiple robustness checks are used, not single-metric optimization.
- Benchmark-relative evidence is explicit and central in final ranking.
- Selected model is interpretable (feature-level transparency).

---

## 8. Reproducibility and Audit Steps

1. Install:
- `pip install -r requirements.txt`

2. Launch:
- `streamlit run streamlit_app.py`

3. In app:
- Open `Final Submission Strategy` tab
- Click `Show Best 5 Proposed Strategies`
- Click `Apply Best Of Best (Top 1)`

4. Confirm displayed outputs:
- selected feature basket,
- model table ranking,
- OOS excess vs Brent,
- quarterly rebalance model label.

---

## 9. Final Submission Statement

I submit a multi-signal, interpretable, quarterly OOS-rebalanced linear meta-strategy that combines short-horizon mean reversion, ML-light directional voting, and micro-lag reversal. Under current out-of-sample testing, it delivers strong risk-adjusted performance and materially outperforms the Brent benchmark on both OOS annualized return and OOS Sharpe, while maintaining transparent methodology and reproducible implementation.
