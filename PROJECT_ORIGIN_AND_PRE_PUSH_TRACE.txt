# Project Origin and Pre-Push Trace

Date: 2026-02-19
Repository: `cand-project-2`
Purpose: Provide another way to understand how this project started and what changed before/through pushes.

---

## Short Answer

Yes, there is another way to understand everything before/around pushes:
- Use **Git history + reflog + initial commit snapshot** for code truth.
- Use the **assignment document** (`project description.docx`) for original objective.
- Use available **chat intent in this thread** for requirement context.

This file is that combined trace.

---

## Sources Used (and Limits)

Sources:
1. `git log --reverse` (full chronological commit history)
2. `git show 1bdfc39 --name-status` (initial project state)
3. `git reflog` (local action timeline)
4. `project description.docx` (project brief)

Limit:
- I cannot access any separate chat history outside this current conversation thread.
- For chats outside this thread, Git is the reliable historical source.

---

## 1) How the Project Started

## 1.1 Assignment-driven start
The assignment brief in `project description.docx` defines:
- Objective: systematic crude strategy on commodity total return index.
- Priority: realistic OOS Sharpe and rigorous process over raw headline returns.
- Constraints: no lookahead, include transaction costs, walkforward methodology.

## 1.2 Initial code baseline
Initial commit: `1bdfc39` (`2026-02-19 17:28:21 +0530`)

It already contained a full product baseline, not a tiny scaffold:
- Core engines (`research_engine.py`, `robustness_engine/engine.py`)
- Dashboard (`dashboard.py`)
- Walkforward framework (`backtest.py`, `validation_framework/walk_forward.py`)
- Diagnostics/regime/blending modules
- Input data (`brent_index.xlsx`)
- Precomputed outputs (`outputs/`, `outputs/survival/`)

So the project did not start as an empty app; it started as a working research system.

---

## 2) Full Commit Timeline (Chronological)

1. `1bdfc39` - Initial commit: crude oil strategy dashboard
2. `c18ac13` - Add Streamlit Cloud entrypoint
3. `9b58b37` - Add one-click Streamlit deploy badge
4. `6ad4daf` - Add submission strategy page and survival/profile explorer upgrades
5. `ff8c59b` - Fix datetime split marker rendering on final submission chart
6. `fa11c39` - Add dynamic final feature basket and expand strategy domains
7. `4d8370e` - Add quarterly OOS rebalancing and Brent outperformance checks
8. `e45fff8` - Add top-5 OOS feature combo presets to final submission sidebar
9. `6e63868` - Add fixed top-5 submission proposals and apply controls
10. `c0265ad` - Fix Streamlit widget-state error when applying fixed proposals
11. `a7a454f` - Harden fixed proposal apply flow to avoid widget state mutations
12. `ca2cedb` - Add full product 360 analysis and git history change story

---

## 3) “Before Push” Understanding (Practical View)

If by “before push changes” you mean “what the project looked like at each stage”, the safest checkpoints are commit SHAs.

Recommended checkpoints:
- **Origin state**: `1bdfc39`
- **Cloud-deploy ready baseline**: `c18ac13`, `9b58b37`
- **Submission-feature expansion phase**: `6ad4daf` to `4d8370e`
- **Top-5 recommendation UX phase**: `e45fff8` to `a7a454f`
- **Documentation/provenance phase**: `ca2cedb`

This is stronger than relying on memory, because each state is reproducible.

---

## 4) Intent Timeline from This Conversation (Available Context)

Within this thread, requirements evolved like this:
1. Dynamic feature picking in final submission with full recalculation.
2. Expand strategy domains while preserving existing ones.
3. Add quarterly test-period rebalance and Brent beat checks.
4. Add top candidate combinations for client-facing reporting.
5. Change from dynamic shortlist to fixed proposal set for client submission.
6. Fix Streamlit runtime state errors in proposal-apply flow.
7. Produce long-form product analysis docs and commit-history story.

This intent sequence matches the commit sequence from `fa11c39` onward.

---

## 5) How You Can Inspect Any Past State Yourself

Commands:
- Show timeline:
  - `git log --reverse --oneline`
- Show first state:
  - `git show --name-status 1bdfc39`
- Compare two stages:
  - `git diff 1bdfc39..ca2cedb -- dashboard.py`
- Inspect one stage quickly:
  - `git checkout <sha>`
  - `streamlit run streamlit_app.py`

Note: after detached-head inspection, return with:
- `git checkout main`

---

## 6) Bottom Line

Yes, there is a robust way to understand all pre-push changes and project origin:
- Git commit graph + initial snapshot + assignment brief.
- This method is reproducible and more reliable than memory/chat fragments.

This document is now part of the repo as a provenance layer.
