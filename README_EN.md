# Risk Modeling Assistant

> No-code credit scorecard modeling platform — from raw data to an explainable risk scorecard, built for analysts without writing a single line of code.

A visual modeling tool for risk / actuarial / strategy professionals. It covers the full pipeline — **data ingestion → automated EDA → WOE/IV binning → scorecard training → model evaluation → explainable reporting → multi-format export** — with zero coding required.

## Features (all implemented)

| Module | Capability |
|--------|-----------|
| Data Ingestion | Upload CSV/Excel, auto-detect field types, identify target column, report missing-rate stats |
| Auto EDA | Target distribution, univariate statistics, correlation matrix, missing-value overview |
| WOE/IV Binning | Optimal tree-based binning, IV computation & feature selection (manual bin-edge tuning supported) |
| Scorecard Modeling | Logistic-regression training with PDO / base score / points-double parameters, outputs a standard scorecard table |
| Model Evaluation | KS, AUC/Gini, confusion matrix, Lift curve, score distribution, PSI stability |
| Explainability | Model-selection rationale, feature importance, per-sample score breakdown & contribution |
| Multi-format Export | One-click HTML report / Python scoring code / SQL scoring rules |
| Deployment & Monitoring | Deployment guide, PSI monitoring dashboard |

## Tech Stack

- **Frontend**: HTML + CSS + vanilla JS (static SPA, no build step)
- **Backend**: Python + FastAPI (25+ RESTful endpoints)
- **Engine**: pandas / numpy / scikit-learn (WOE/IV + logistic-regression scorecard)
- **Data**: Built-in German Credit demo dataset (works out of the box)

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# If pip fails due to network issues:
# pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. Start the backend

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8080
```

### 3. Open the frontend

Open `frontend/index.html` directly in a browser. It calls `http://127.0.0.1:8080` by default.

> Or use `start.bat` for a one-click launch on Windows.

### Pure-Frontend Mode (no backend / live demo)

The core algorithms (WOE/IV binning, logistic-regression scorecard, KS/AUC evaluation, PSI, interpretation, and export) have been fully ported to `frontend/engine.js`, so the app runs entirely in the browser with **no backend**:

- Default mode is pure-frontend: click "Load Sample Data" to run the full pipeline locally. Ideal for static hosting (e.g. GitHub Pages).
- The built-in German Credit dataset (`frontend/german_credit.csv`) ships with the frontend.
- To use your own backend, append `?api=https://your-backend` to the URL; it falls back to the local engine when no backend is configured.

```
# GitHub Pages live demo (pure frontend):
https://modaniel923-coder.github.io/risk-modeling-assistant/
```

> In pure-frontend mode, custom CSV upload is parsed locally in the browser — no data leaves the device.

### 4. Run tests

```bash
pytest tests/
# or run_tests.bat on Windows
```

## Project Structure

```
risk-modeling-assistant/
├── frontend/              # Static frontend (10 functional pages)
│   ├── index.html         # Main page
│   ├── styles.css         # Styles
│   └── app.js             # Interaction & API calls
├── backend/
│   ├── main.py            # Entry point
│   ├── api/routes.py      # FastAPI routes (25+ endpoints)
│   ├── engine/            # Core algorithm engine
│   │   ├── data_loader.py # Data loading + type inference
│   │   ├── eda.py         # EDA analysis
│   │   ├── woe_iv.py      # WOE/IV binning
│   │   ├── scorecard.py   # Scorecard training
│   │   ├── evaluator.py   # Model evaluation
│   │   ├── explainer.py   # Model explanation
│   │   └── exporter.py    # Multi-format export
│   └── data/              # Built-in sample data (german_credit.csv)
├── tests/                 # 54 test cases
├── requirements.txt
├── start.bat              # One-click launch
└── run_tests.bat          # One-click test
```

## API Overview

| Method | Path | Function |
|--------|------|----------|
| GET | /api/v1/health | Health check |
| POST | /api/v1/data/upload | Upload data file |
| POST | /api/v1/data/load-sample | Load built-in sample |
| GET | /api/v1/data/summary | Data summary |
| GET | /api/v1/data/preview | Data preview |
| POST | /api/v1/eda/run | Run EDA |
| POST | /api/v1/binning/run | Run WOE binning |
| GET | /api/v1/binning/iv-ranking | IV ranking |
| GET | /api/v1/binning/woe/{feature} | WOE table for a feature |
| POST | /api/v1/training/run | Train scorecard |
| GET | /api/v1/scorecard/table | Scorecard table |
| POST | /api/v1/evaluation/run | Model evaluation |
| POST | /api/v1/explain/sample | Single-sample explanation |
| POST | /api/v1/export/html | Export HTML |
| POST | /api/v1/export/python | Export Python |
| POST | /api/v1/export/sql | Export SQL |

Full endpoints are in `backend/api/routes.py`. Full Chinese documentation: [README.md](README.md).

## Author

modaniel923 — former actuary, currently in overseas credit-risk, transitioning toward AI Builder.
