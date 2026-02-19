# Crude Oil Strategy Research Dashboard

Interactive Streamlit dashboard for evaluating quantitative strategy robustness and survival diagnostics on crude oil data.

## What this includes

- Strategy ranking and explainers
- Risk, return, and regime-aware diagnostics
- Walk-forward validation summaries
- Robustness and survival framework views

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Data and outputs

This repo includes:

- Input data file: `brent_index.xlsx`
- Precomputed output tables under `outputs/`

The app can use these directly for fast startup.

## Deploy (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud and create a new app from the repo.
3. Set:
   - Main file path: `streamlit_app.py`
   - Python dependencies: `requirements.txt`
4. Deploy and share the generated app URL.
