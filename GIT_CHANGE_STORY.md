# Git Change Story - From Initial Build to Current Product

Date: 2026-02-19
Repository: `cand-project-2`
Range analyzed: first commit to current `main` head

---

## Executive Story

This repository evolved in one day through a clear sequence:
1. Build the full quant research product baseline.
2. Make it deployable on Streamlit Cloud.
3. Add a submission-oriented workflow and deeper survival diagnostics.
4. Expand strategy domains and dynamic feature/model controls.
5. Add benchmark-aware (Brent) OOS governance and quarterly rebalancing.
6. Add Top-5 proposal UX, then shift from dynamic presets to fixed client-facing proposals.
7. Stabilize Streamlit session state behavior through two follow-up fixes.

In short: the project moved from "research dashboard" to "client-ready strategy submission cockpit" with governance and benchmark-relative controls.

---

## Commit-by-Commit Timeline

## 1) `1bdfc39` - Initial commit: crude oil strategy dashboard

What changed:
- Added the full project skeleton and core engines:
  - `dashboard.py`
  - `research_engine.py`
  - `robustness_engine/engine.py`
  - diagnostics/walk-forward/regime/blending submodules
- Added data and precomputed outputs (`outputs/`, `outputs/survival/`).
- Added notebook and assignment document.

Why it matters:
- This created the complete baseline product in one commit: strategy generation, evaluation, walkforward, robustness, and UI.

Product impact:
- First usable version of the crude oil strategy research dashboard.

---

## 2) `c18ac13` - Add Streamlit Cloud entrypoint

What changed:
- Added `streamlit_app.py` and updated `README.md` run/deploy guidance.

Why it matters:
- Standardized cloud entrypoint for Streamlit deployments.

Product impact:
- App became straightforward to deploy on Streamlit Community Cloud.

---

## 3) `9b58b37` - Add one-click Streamlit deploy badge

What changed:
- Added deploy badge in `README.md`.

Why it matters:
- Lowered deployment friction for reviewers/users.

Product impact:
- Better developer/user onboarding.

---

## 4) `6ad4daf` - Add submission strategy page and survival/profile explorer upgrades

What changed:
- Major `dashboard.py` enhancements around submission flow and survival-mode UX.
- Minor updates in `robustness_engine/engine.py` and `strategy_diagnostics/diagnostics.py`.

Why it matters:
- Introduced a more practical "submission" orientation instead of only exploratory analytics.

Product impact:
- Dashboard became much more decision-oriented for candidate strategy selection.

---

## 5) `ff8c59b` - Fix datetime split marker rendering on final submission chart

What changed:
- Corrected datetime rendering for train/test split marker in final submission chart.

Why it matters:
- Fixed chart interpretability and avoided misleading visual cues.

Product impact:
- Improved trust/readability in final evaluation visuals.

---

## 6) `fa11c39` - Add dynamic final feature basket and expand strategy domains

What changed:
- Added dynamic feature add/remove behavior in final submission.
- Added default family-diverse top-3 feature pick by walkforward annualized return.
- Expanded strategy factories with new families/domains (time-series/event/microstructure/risk overlays).
- Updated outputs and synced regenerated tables.

Why it matters:
- Increased model design flexibility and universe diversity.
- Made final submission reactive to feature basket changes.

Product impact:
- Larger, richer strategy search space and more interactive final strategy construction.

---

## 7) `4d8370e` - Add quarterly OOS rebalancing and Brent outperformance checks

What changed:
- Added quarterly out-of-sample re-fit/rebalance variants for LR/GB/Ensemble.
- Added explicit Brent-relative excess metrics and beat flags in model/feature tables.
- Added auto-pick best pool strategy with benchmark-aware logic.

Why it matters:
- Introduced stronger OOS governance and benchmark accountability.

Product impact:
- Submission workflow moved closer to real PM/client expectations: "Are we truly beating the index OOS?"

---

## 8) `e45fff8` - Add top-5 OOS feature combo presets to final submission sidebar

What changed:
- Added top-5 combo preset engine based on OOS model outcomes.
- Added sidebar controls to pick preset baskets.

Why it matters:
- Enabled quick navigation across high-performing candidate feature sets.

Product impact:
- Faster what-if evaluation in final submission stage.

---

## 9) `6e63868` - Add fixed top-5 submission proposals and apply controls

What changed:
- Shifted from purely dynamic combo presets to fixed proposal workflow.
- Added top button and sidebar controls for fixed "Best 5" client-facing proposals.
- Added proposal application behavior to load feature basket + hyperparameters + model preference.

Why it matters:
- Addresses client communication needs: stable recommended options, not constantly changing ranks.

Product impact:
- Product gained a direct "presentation mode" for final recommendations.

---

## 10) `c0265ad` - Fix Streamlit widget-state error when applying fixed proposals

What changed:
- Added pending-apply rerun-safe mechanism to avoid illegal widget key mutation timing.

Why it matters:
- Resolved runtime Streamlit `SessionState` exceptions in proposal application path.

Product impact:
- Proposal application became more reliable in deployed app contexts.

---

## 11) `a7a454f` - Harden fixed proposal apply flow to avoid widget state mutations

What changed:
- Further simplified/hardened apply logic by removing rerun-based mutation complexity.
- Applied proposal settings via in-run payload override pattern.

Why it matters:
- Eliminated remaining class of widget-state mutation failures.

Product impact:
- Final submission proposal controls are now significantly more stable.

---

## Technical Evolution by Theme

## A) Product UX evolution
- Started as analysis-heavy dashboard.
- Added survival diagnostics and final submission workflow.
- Added client-facing fixed Top-5 proposal UX with apply actions.

## B) Strategy research evolution
- Expanded from baseline families to broader domains:
  - time series adaptive
  - event flow
  - microstructure proxies
  - risk overlays

## C) Governance evolution
- Added quarterly OOS rebalancing in final model path.
- Added Brent-relative outperformance checks and beat flags.
- Added proposal ranking tied to OOS and benchmark excess.

## D) Reliability evolution
- Two consecutive hardening commits addressed Streamlit session-state constraints in callback/apply flow.

---

## Net Result at Current Head

The current product is no longer just a strategy explorer. It is now:
- A multi-family systematic strategy research platform,
- with survival diagnostics and rejection reporting,
- plus a benchmark-aware final submission engine,
- and a fixed Top-5 proposal flow designed for client presentation.

The dominant trajectory of the commit history is clear: **from research breadth -> submission rigor -> client-ready recommendation workflow -> runtime hardening**.
