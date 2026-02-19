# Crude Oil Strategy Dashboard - 360 Degree Product and Code Analysis

Date: 2026-02-19
Repository: `cand-project-2`
Main app entrypoint: `streamlit_app.py`
Primary engine modules: `dashboard.py`, `research_engine.py`, `robustness_engine/engine.py`

---

## Part A: Refined Prompt (Improved Version)

Use this refined prompt when you want an exhaustive, investor/client-grade review of this product:

"Create an end-to-end, code-grounded 360 degree analysis of the Crude Oil Strategy Dashboard. Cover product goals, user workflows, architecture, data pipeline, feature engineering, strategy generation, validation logic, robustness framework, and final submission workflow. Map every major module and function to user-facing behavior. Explain all dashboard tabs, controls, and outputs; show how strategy families are generated and evaluated; detail walk-forward and overfit controls; and explain how Brent benchmark comparison is integrated. Include operational guidance (run/deploy), known limitations, current risks, and prioritized improvement roadmap. The analysis should be practical: what the product is, what it does well, where it is fragile, and how to use it correctly to produce defensible strategy recommendations for clients."

Why this refined prompt is better:
- It forces code-grounded analysis, not surface-level product marketing.
- It ties code internals to user behavior and business outcomes.
- It explicitly asks for risk/limitations and actionable roadmap.
- It is suitable for both technical review and client-facing preparation.

---

## Part B: Plan (How This 360 Analysis Was Built)

1. Inventory the repo structure and entrypoints.
2. Map core modules and all major functions.
3. Trace dataflow from input data to output artifacts.
4. Decode strategy-generation families and model workflows.
5. Map dashboard UX flows (Discovery, Survival, Final Submission).
6. Validate outputs and current runtime behavior with available CSV artifacts.
7. Summarize strengths, gaps, failure modes, and roadmap.
8. Write this document as implementation-grounded reference.

---

## Part C: Full 360 Product + Code Analysis

## 1) What this product is

This is a Streamlit-based quantitative research and strategy-submission platform for Brent crude systematic trading. It combines:
- A broad rule-based strategy factory.
- Walk-forward model selection and robustness diagnostics.
- A survival/rejection framework with overfit flags.
- A final submission builder that turns selected strategy signals into model-based portfolios and compares them versus Brent.

In plain terms:
- It is not just a backtest notebook.
- It is a research system with diagnostics, auditability, and client-ready selection workflows.

## 2) Product objective and intended outcome

Primary objective:
- Build strategy candidates and select robust ones with realistic validation logic.

Secondary objective:
- Package a defensible final strategy proposal with out-of-sample diagnostics and benchmark-relative evidence.

The system now explicitly includes:
- Quarterly OOS rebalancing for final model variants.
- Brent outperformance checks across selected strategies and submission models.
- Fixed Top-5 proposed submission candidates (client-facing mode).

## 3) Repo architecture and code map

## 3.1 Entrypoints
- `streamlit_app.py`
  - Thin launcher: imports and calls `main()` from `dashboard.py`.
- `dashboard.py`
  - App UI, control logic, metrics visualizations, Final Submission workflow.

## 3.2 Core research engines
- `research_engine.py`
  - Data load, feature engineering, strategy generation, metric computation, walkforward grid, robustness tests, and output CSV persistence.
- `robustness_engine/engine.py`
  - Survival framework: expanded strategy families, hard filters, parameter/regime diagnostics, rolling walk-forward survival, feature blending, and output persistence under `outputs/survival/`.

## 3.3 Supporting analytics
- `strategy_diagnostics/diagnostics.py`
  - Trade exposure metrics, parameter-surface diagnostics, IS/OOS drift diagnostics, logic registry, hard filters.
- `validation_framework/walk_forward.py`
  - Rolling train/test walk-forward process and score summary.
- `regime_analysis/regimes.py`
  - Regime table, regime-level strategy breakdown, regime robustness score, rolling stability time series.
- `feature_blending_lab/blending.py`
  - OLS/Ridge/Lasso/Weighted-voting blends over robust features.
- `backtest.py`
  - Legacy/adaptive monthly walk-forward backtester used by baseline research engine grid.

## 4) Data and output artifacts

## 4.1 Input
- `brent_index.xlsx`
  - Primary price source (date + Brent total return proxy series column).

## 4.2 Output folders
- `outputs/` (baseline discovery outputs)
- `outputs/survival/` (survival diagnostics outputs)

Key CSV artifacts include:
- `strategy_metrics.csv`
- `walkforward_grid.csv`
- `rebalance_details.csv`
- `walkforward_portfolio.csv`
- `robustness_summary.csv`
- `noise_injection_test.csv`
- `ensemble_metrics.csv`
- Survival files like `robustness_diagnostics.csv`, `strategy_graveyard.csv`, `parameter_surface.csv`, `regime_breakdown.csv`, etc.

## 4.3 Current observed output snapshot

From current local outputs/code execution:
- Base universe: 281 strategies, 11 families.
- Combined (base + survival expansion): 327 strategies, 18 families.
- Top base-strategy Sharpe includes: `vol_lowtrend_20_10_0.4` (~0.976 Sharpe).
- Best baseline walkforward grid row observed: lookback 12M, rebalance 1M, Sharpe ~0.323.
- Ensemble metrics are materially stronger than single walkforward stitch in provided outputs (inverse-vol Sharpe > 1.0).

Important observed behavior:
- Survival hard filters currently reject all strategies in current diagnostics, mostly with rejection reason `Flat performance`.
- The app still remains usable because downstream logic has fallback behavior (takes top-ranked rows if pass-set is empty).

This is an important point for governance and interpretation.

## 5) Feature engineering system (`research_engine.py`)

The feature library is broad and mixed-domain. It includes:

## 5.1 Calendar features
- Day-of-week, week-of-month/year, month, quarter.
- Month start/end flags, turn-of-month proxy, first trading day flags.
- Seasonal proxies (summer driving season, pre-OPEC month proxy).
- Trig encodings for cyclical calendar positions.
- One-hot flags for DOW/month/quarter.

## 5.2 Price and return features
- 1D and lagged returns/log-returns.
- Multi-horizon return windows (2 to 252 days).
- Compounded returns, rolling vol and annualized realized vol.
- Vol-of-vol, SMA/EMA ratios, rolling z-scores.
- Distance to rolling highs/lows, breakout position, rolling drawdowns.

## 5.3 Statistical structure features
- Momentum adjusted by vol.
- Rolling Sharpe, skew, kurtosis.
- Autocorrelation proxies (lag 1 and lag 5).
- Rolling slope of log-price.
- Variance ratio proxies.
- Volatility percentile.
- RSI family, MACD components.
- Hurst and entropy proxies.
- Trend strength and regime flags.

Interpretation:
- This is a rich alpha-factor substrate.
- It combines signal types that can support trend, MR, regime-switching, and lightweight ML rules.

## 6) Strategy generation factory (`build_strategy_library`)

Strategy generation is explicitly additive and family-structured.

Families present in base engine include:
- `trend_following`
- `mean_reversion`
- `volatility_regime`
- `time_calendar`
- `stat_structure`
- `risk_managed_alpha`
- `hybrid`
- `ml_like`
- `time_series`
- `event_flow`
- `microstructure_proxy`

Representative construction logic:
- Trend: momentum signs, slope signs, Donchian, MA crosses.
- Mean reversion: z-score thresholds, RSI threshold reversals, short-term reversal.
- Volatility regime: low-vol trend continuation and high-vol reversal switching.
- Calendar: DOW, month bias, turn-of-month, seasonality.
- Statistical: autocorr switches, variance ratio regimes, Hurst/entropy switching.
- Risk managed: vol-target leverage, drawdown scaling, Kelly proxy.
- ML-like: linear score blends and thresholded pseudo-models.
- Time-series/event/microstructure expansions: dual momentum agreements, EWM edge, turn-month and quarter turns, shock reversals, calm-bounce logic.

Net effect:
- Broad candidate diversity, good for avoiding single-style overfitting.

## 7) Metric and validation framework

## 7.1 Core metrics
`compute_metrics` evaluates:
- Sharpe, Sortino, annual return, annual vol, max drawdown, Calmar.
- Hit rate, skew, kurtosis.
- Turnover and average holding period (when signal provided).
- Rolling Sharpe stability proxy.

## 7.2 Baseline walkforward grid
Using `WalkforwardBacktester` (monthly lookback/rebalance grids):
- Searches `lookback_grid` x `rebalance_grid`.
- Picks best strategy by score in each rebalance segment.
- Stitches OOS returns for walk-forward portfolio.

## 7.3 Robustness checks
- Block bootstrap Sharpe distribution.
- Random-entry baseline comparison.
- Noise injection by flipping signal states.

Interpretation:
- The platform checks not only raw KPI but also perturbation stability.

## 8) Survival framework (`run_survival_framework`)

This is the second validation layer.

What it adds:
- Additional strategy families (time-based, vol-based, regime detection, statistical, ml_light, time_series_adaptive, event_flow, risk_overlay).
- Trade and exposure diagnostics.
- Parameter surface diagnostics.
- IS/OOS consistency deltas.
- Regime robustness scoring.
- Hard-filter rejection logic with reasons.
- Rolling walk-forward summary and stitched OOS curve.
- Feature blending lab outputs.

Final robustness scoring (`final_robustness_score`) combines:
- Walk-forward quality.
- Parameter stability.
- Trade consistency.
- Exposure stability.
- Regime diversification.
- Penalty by red flags and overfit-risk tags.

Hard filter mechanics (`apply_hard_filters`) can reject on:
- Low trade count/frequency.
- Low exposure/high inactivity.
- Flat/stagnation behavior.
- Parameter instability/sharp peaks.
- IS/OOS gap.
- Missing logic docs (strict profile only).

Critical governance point:
- In current state, all strategies are rejected by these filters, implying thresholds and/or stagnation detection may be too strict for current signal set.

## 9) Dashboard UX: mode-by-mode

## 9.1 Sidebar controls
- Engine mode: `Discovery Mode` or `Survival Mode`.
- Tutor layer toggle.
- Discovery option to include survival-generated strategy universe.
- Sorting/filtering controls in discovery mode.
- Survival filter profile in survival mode.

## 9.2 Discovery Mode tabs
1. `Universe Overview`
   - KPI summaries, ranking table, risk-return scatter, family aggregates, hero picks.
2. `Strategy Explorer`
   - Single-strategy deep dive: KPI tiles, activity snapshot, cumulative chart vs Brent, drawdown/rolling charts, regime decomposition, monthly heatmap, yearly table.
3. `Cross-Compare`
   - Multi-strategy cumulative comparison, KPI table, correlation matrix, Brent baseline KPI view.
4. `Walkforward & Robustness`
   - Walkforward grid, best config stats, decision logs, robustness/noise diagnostics, ensemble outcomes.
5. `Final Submission Strategy`
   - Feature basket control, model fitting, model comparison, benchmark checks, final scorecard and charts.

## 9.3 Survival Mode tabs
1. `Robustness Diagnostics`
2. `Walk-Forward Survival`
3. `Parameter Surfaces`
4. `Feature Blending Lab`
5. `Strategy Graveyard`

This mode is designed to be adversarial against overfit and weak documentation.

## 10) Final Submission system (deep dive)

The Final Submission page is now a full model-construction and audit surface.

Core flow:
1. Build eligible strategy feature pool from survival diagnostics.
2. Default feature basket: top-3 by walkforward annualized return, family-diversified when possible.
3. Allow manual feature add/remove (dynamic recalculation).
4. Fit model stack on selected feature signals to predict next-day Brent return.

Model set includes:
- Linear Regression
- Linear Regression (Quarterly Rebalanced OOS)
- Gradient Boosting Proxy
- Gradient Boosting Proxy (Quarterly Rebalanced OOS)
- Ensemble (50/50)
- Ensemble (Quarterly Rebalanced OOS)
- Equal-Weight Feature Blend
- Best Pool Strategy (auto-picked from eligible universe)
- Buy & Hold Brent benchmark

Benchmark-aware fields:
- OOS/full excess ann return vs Brent.
- OOS/full excess Sharpe vs Brent.
- Boolean beat flags.

Quarterly OOS rebalance:
- Re-fits ridge and boosting proxy at each OOS quarter boundary.
- Uses only data up to boundary date (chronological discipline).

## 10.1 Fixed Best-5 proposal workflow (current)

Current code includes a fixed proposal layer for client-facing submission:
- Precomputes top 5 proposal rows via broad feature-combo + alpha-grid search.
- Surfaces them in page and sidebar.
- Provides:
  - `Show Best 5 Proposed Strategies` button.
  - Dropdown with fixed proposals.
  - `Apply Selected Proposed Strategy`.
  - `Apply Best Of Best (Top 1)`.

Proposal generation logic:
- Candidate combos from top seeded features and family-diverse heuristics.
- Feature counts from 3 to 9.
- Hyperparameter grid over split ratio, ridge alpha, GB rounds, GB learning rate.
- Candidate models restricted to quarterly LR/GB/Ensemble variants.
- Ranked by Brent beat + OOS excess return + OOS Sharpe style ordering.

Current observed Top-5 (example run):
- Dominated by `Linear Regression (Quarterly Rebalanced OOS)` with 3-4 features.
- Top feature set repeatedly includes:
  - `mr_st_rev_10`
  - `ml_light_random_forest_vote`
  - `micro_lag_revert_2`

Interpretation:
- Current search space is finding a strong triad plus slight hyperparameter variants.
- If diversity in final top-5 is desired, ranking constraints should enforce uniqueness on feature-set and model family beyond current dedupe keys.

## 11) How to use this product correctly (practical SOP)

## 11.1 Research phase
1. Start in Discovery mode with survival universe included.
2. Filter by family and sort by `activity_adjusted_sharpe` and risk metrics.
3. Use Strategy Explorer to validate path behavior and regime dependence.
4. Use Cross-Compare to check diversification (correlation matrix).

## 11.2 Validation phase
1. Open Walkforward & Robustness tab.
2. Inspect best walkforward config and rebalancing log.
3. Check noise injection sensitivity and bootstrap context.
4. Compare ensemble outcomes vs single-strategy outcomes.

## 11.3 Survival/adversarial phase
1. Switch to Survival mode.
2. Review diagnostics and red flags.
3. Use graveyard reasons as rejection audit trail.
4. Use feature blending lab when robust candidate set exists.

## 11.4 Submission phase
1. Open Final Submission tab.
2. Either manually build basket or apply fixed top proposal.
3. Verify model table fields versus Brent OOS.
4. Choose final model and export narrative from KPI + IS/OOS + yearly breakdown.

## 12) Strengths of the current product

1. Broad strategy/factor coverage across multiple domains.
2. Clear transaction-cost integration in PnL construction.
3. Multiple validation layers (walkforward, noise, bootstrap, regime).
4. Strong dashboard observability for both quant and non-quant stakeholders.
5. Explicit benchmark-relative metrics in final submission workflow.
6. Audit-friendly survival diagnostics with rejection reasons.
7. Additive architecture: new families and modes integrated without removing prior functionality.

## 13) Risks and weaknesses

1. Survival hard filters currently reject 100% of strategies in present diagnostics.
   - This can confuse users and reduce trust unless clearly explained.
2. Potential inconsistency between baseline output files and runtime regeneration behavior.
   - `load_output_tables()` forces `run_research()`, which can be expensive and introduces startup latency.
3. Heavy compute on page load due cached builders + large feature matrix generation.
4. Fixed proposal top-5 currently shows closely related variants; limited diversity in candidate set.
5. No formal test suite in repo (unit/integration) for regression prevention.
6. Streamlit state complexity in Final Submission can be fragile if further callbacks mutate keyed widgets after creation.

## 14) Engineering quality review

What is strong:
- Modularized engines by concern.
- Functional decomposition is mostly clear.
- Consistent use of pandas/numpy idioms.
- Reasonable metric standardization and naming.

What is fragile:
- `dashboard.py` is very large and contains substantial business logic.
- UI and model-generation logic are tightly coupled in same file.
- Could benefit from service-layer extraction to reduce state/callback complexity.

## 15) Performance and scalability considerations

Current bottlenecks:
- Feature generation over large windows and large column set.
- Strategy factory producing hundreds of strategies.
- Repeated survival + final proposal computations.

Recommendations:
- Precompute/cached artifacts with version stamps.
- Separate heavy compute from render cycle.
- Add manual “recompute” toggles for expensive blocks.
- Persist proposal candidates to artifact file with refresh button.

## 16) Suggested prioritized roadmap

## P0 (stability + correctness)
1. Add automated tests for:
   - strategy generation counts and family presence,
   - final submission payload schema,
   - Streamlit state-safe apply logic.
2. Diagnose and tune survival `Flat performance` rejection thresholds.
3. Add explicit on-screen warning when robust pass-set is empty and fallback is used.

## P1 (usability + trust)
1. Add “data/build timestamp” indicator for outputs.
2. Add “why this proposal” explainability card for Top-5 proposals.
3. Add one-click export bundle for client memo (tables + key charts).

## P2 (research depth)
1. Add additional benchmark set (e.g., static long-only variants).
2. Add sensitivity sweeps for transaction cost assumptions.
3. Add regime-specific submission candidates (bull/bear/high-vol segmentation).

## 17) Client-facing narrative template (short)

Use this narrative in meetings:
- "We generate a broad, multi-family strategy universe on Brent, apply transaction costs, and rank by robust risk-adjusted metrics."
- "We validate with walkforward, noise-injection, and regime decomposition to reduce overfitting risk."
- "Final submission models are benchmarked against Brent with explicit OOS excess return and Sharpe checks."
- "We maintain a fixed top-5 proposal set for client decisioning, with transparent features, parameters, and performance diagnostics."

## 18) How to run and deploy

Local:
1. `pip install -r requirements.txt`
2. `streamlit run streamlit_app.py`

Deploy (Streamlit Community Cloud):
1. Push to GitHub.
2. Set main file `streamlit_app.py`.
3. Reboot app after major changes.

If UI behavior appears stale:
- Clear Streamlit cache + reboot app.
- Confirm branch is `main` and latest commit is deployed.

## 19) Final assessment

This product is a strong research and submission cockpit for systematic crude strategy development. It has meaningful depth in strategy generation and diagnostics, and now includes a practical client-facing fixed proposal workflow. The biggest current gap is survival filter calibration (all strategies rejected) and general hardening via tests/state management. Once those are addressed, this can serve as a robust candidate-assignment platform and a credible client demo system.

---

## Appendix A: Quick module reference

- `streamlit_app.py`: app launcher.
- `dashboard.py`: UI, controls, analytics views, final submission logic.
- `research_engine.py`: base research run + output generation.
- `robustness_engine/engine.py`: survival framework and robustness scoring.
- `strategy_diagnostics/diagnostics.py`: hard filters + parameter/IS-OOS diagnostics.
- `validation_framework/walk_forward.py`: rolling train/test walkforward.
- `regime_analysis/regimes.py`: regime assignment and robustness.
- `feature_blending_lab/blending.py`: blended meta-signal models.
- `backtest.py`: monthly adaptive backtest class.

## Appendix B: Current fixed proposal behavior note

The current fixed top-5 search can produce near-duplicate rows if a feature set remains dominant and only hyperparameters differ slightly. This is mathematically consistent with current ranking logic, but for presentation diversity you may enforce minimum proposal distance constraints (feature-set overlap cap + model diversity cap).
