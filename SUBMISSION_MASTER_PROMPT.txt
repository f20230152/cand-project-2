# Best Submission Prompt (Project-Aligned)

Use this prompt to generate a high-quality final submission document for this project:

---

You are preparing a final candidate submission for a systematic crude oil strategy assignment.

Context and source files:
- Assignment brief: `project description.docx`
- Product architecture and full analysis: `PRODUCT_360_ANALYSIS.md`
- Mathematical details and selection methodology: `STRATEGY_MATH_AND_SELECTION_PLAYBOOK.md`
- Project history and evolution: `GIT_CHANGE_STORY.md`
- Project provenance and origin trace: `PROJECT_ORIGIN_AND_PRE_PUSH_TRACE.md`
- Current codebase: `dashboard.py`, `research_engine.py`, `robustness_engine/engine.py`, and supporting modules
- Current output artifacts: `outputs/*`, `outputs/survival/*`

Goal:
Write a submission-ready report that is clear, rigorous, and concise, optimized for evaluator expectations in the assignment.

Requirements:
1. Follow the assignment structure exactly:
- Strategy intuition and motivation
- Signals/features and parameters
- Walkforward setup (lookback + rebalance + no-lookahead handling)
- Performance summary with emphasis on out-of-sample Sharpe
- Risks, limitations, and observations

2. Use both plain-English and technical language:
- Explain each major concept in simple terms
- Also include technical formulation where useful (PnL with transaction costs, Sharpe, walkforward logic)

3. Use current project evidence (not generic finance text):
- Mention the implemented strategy families and why they exist
- Describe the final selected submission strategy and why it was chosen
- Include benchmark-relative evidence versus Brent (excess return and Sharpe)

4. Stay credible and evaluator-friendly:
- Do not over-claim predictive power
- Explicitly discuss overfit controls and remaining risks
- Include reproducibility steps

5. Style constraints:
- Professional, concise, and audit-friendly
- Use headings and compact tables where useful
- Target ~3-4 pages worth of content

Expected output title:
"Crude Oil Systematic Strategy Submission - Final Report"

---

Optional extension:
After generating the final report, produce a 1-page executive summary for non-technical stakeholders.
