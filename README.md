# 🏦 IDBI Credit Risk Intelligence Engine

[![CI](https://github.com/kushal040511/IDBI-main/actions/workflows/ci.yml/badge.svg)](https://github.com/kushal040511/IDBI-main/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-EB5E28)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](#-license)

> An **explainable, forward-looking credit-default prediction engine** and early-warning dashboard for MSME lending — *IDBI Track 04: MSME Credit / Predictive AI / Risk Management.*

The platform scores a borrower's **12-month Probability of Default (PD)**, quantifies **Expected Loss (PD × EAD × LGD)**, explains *why* with **SHAP**, and recommends concrete banker interventions — all behind a modern, single-page risk console with an **Agentic Risk Copilot**.

---

## ✨ Highlights

- 🎯 **Hybrid ML model** — XGBoost on **structured credit features + unstructured officer-notes NLP** (`note_stress_index`), Optuna-tuned, isotonic-calibrated, leak-free CV.
- 📈 **Held-out ROC-AUC 0.927** (structured-only 0.898 → **+0.025 NLP uplift**), Accuracy 90%, Precision 73%, Recall 65%, Brier 0.074.
- 🧠 **Full explainability** — per-borrower SHAP attribution translated into plain-English risk drivers.
- 🔮 **Forward-looking** — simulates a borrower's PD trajectory across a 12-month horizon.
- 💸 **Banker's-eye financials** — Expected Loss, exposure, LGD, recovery, and a priority **Action Escalation Ladder**.
- 🤖 **Agentic Risk Copilot** — interactive Q&A on any borrower (Claude → Gemini → offline template fallback).
- 🖥️ **Modern dashboard** — fully wired SPA: Portfolio Overview, Borrowers, Risk Trends, Early Warnings, SHAP Explainer, Expected Loss, Model Performance, and CSV Upload.
- ⚡ **Runs instantly** — the trained model ships in `models/`; no training step required.

---

## 🖼️ Dashboard

The console is a single-page app served by FastAPI. Every sidebar module is live and driven by the backend API.

| Module | What it does |
|---|---|
| **Portfolio Overview** | Metric cards (critical borrowers, expected loss, total book, ROC-AUC) + sortable borrower table + active borrower **dossier** |
| **Borrowers** | Full borrower table with filter chips (All / Critical / High / Watch / Safe) and live search |
| **Risk Trends** | Risk-band distribution + sector risk heatmap (avg PD per sector) |
| **Early Warnings** | One-click filter to the High + Critical alert queue |
| **SHAP Explainer** | Ranked, signed SHAP attribution for the selected borrower |
| **Model Performance** | Visual evaluation report — ROC/PR curves, confusion matrices, score distributions |
| **Expected Loss** | Portfolio loss rate + Expected Loss by sector (ranked by contribution) |
| **Upload Portfolio** | Drop a `credit_risk_dataset` CSV → instant scoring of the whole book |

> Model evaluation charts live in [`evaluation_report/`](evaluation_report/) and are served at `/reports/...`
> (open the **Model Performance** module or visit `/dashboard/evaluation_report.html`).

---

## ⚡ Quickstart

```bash
# 1. Clone
git clone https://github.com/kushal040511/IDBI-main.git
cd IDBI-main

# 2. Environment + dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-runtime.txt      # minimal, pinned — just enough to serve the app
# (or: pip install -r requirements.txt        # full set incl. training / NLP / LLM extras)
brew install libomp                           # macOS only — XGBoost OpenMP runtime

# 3. (Optional) enable the LLM Copilot — else it uses offline templates
cp .env.example .env                          # add ANTHROPIC_API_KEY or GEMINI_API_KEY

# 4. Run
uvicorn main:app --reload --port 8000
```

Open **http://127.0.0.1:8000/** — it auto-redirects to the dashboard.

The trained model ships in `models/`, so the app runs immediately. To **retrain** the shipped model
run `python retrain_calibrated.py` — it regenerates the synthetic dataset, retrains XGBoost with
monotonic constraints + isotonic calibration, and rewrites the `models/*.joblib` artifacts and
`models/performance_report.json` with fresh held-out metrics.

---

## 🧩 API Reference

| Purpose | Endpoint |
|---|---|
| Portfolio summary (totals, risk distribution, sector analytics) | `GET /portfolio` |
| Borrower list (ranked by PD) | `GET /borrowers?limit=50` |
| Borrower detail — PD, 12-month trajectory, Expected Loss, actions | `GET /borrowers/{id}` |
| SHAP explainability for a borrower | `GET /borrowers/{id}/explain` |
| Model metrics | `GET /model-performance` |
| Upload a portfolio CSV (credit_risk schema) | `POST /upload` |
| Simulate onboarding new loans | `POST /refresh?count=10` |
| Restore the demo portfolio | `POST /reset` |
| Agentic Risk Copilot (Claude / Gemini / template) | `POST /copilot` |

**Copilot engine selection:** `/copilot` auto-detects its backend — `ANTHROPIC_API_KEY` → Claude
(`claude-opus-4-8`), else `GEMINI_API_KEY` → Gemini, else a fully-offline rule-based template engine.
No key is required to run the demo.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.11+, FastAPI, Uvicorn |
| **Machine Learning** | XGBoost, scikit-learn, Optuna, pandas, NumPy |
| **Explainability** | SHAP (TreeExplainer) |
| **NLP** | sentence-transformers (officer-notes → `note_stress_index`) |
| **Frontend** | Single-page HTML/CSS/Vanilla JS, Chart.js |
| **LLM (optional)** | Anthropic Claude / Google Gemini via `/copilot` |

---

## 🏗️ Architecture

```
                         ┌──────────────────────────────────────────┐
   Browser (SPA)         │            FastAPI  (main.py)            │
 ┌───────────────┐  HTTP │                                          │
 │ index.html    │◀─────▶│  /portfolio  /borrowers  /borrowers/{id} │
 │  • sidebar    │  JSON │  /explain  /model-performance  /copilot  │
 │  • dossier    │       │  /upload  /refresh  /reset               │
 │  • copilot    │       └───────────────┬──────────────────────────┘
 └───────────────┘                       │
        ▲  /dashboard (static)           │ loads at startup
        │  /reports  (eval charts)       ▼
                              ┌────────────────────────────┐
                              │ models/                    │
                              │  xgb_model.joblib (XGBoost)│
                              │  calibrator.joblib         │
                              │  imputer.joblib            │
                              │  shap_explainer.joblib     │
                              └────────────────────────────┘
```

- **`/`** → redirects to `/dashboard/index.html` (the modern console).
- **`/dashboard`** → static mount serving `static/` (SPA + evaluation report page).
- **`/reports`** → static mount serving `evaluation_report/` (model charts).

---

## 📁 Project Structure

```
IDBI-main/
├── main.py                      # FastAPI app: API, scoring, SHAP, copilot, static mounts
├── main_msme_backup.py          # MSME variant (GST/EMI/cashflow behavioural features)
├── nlp_features.py              # officer-notes → note_stress_index (NLP)
│
├── retrain_calibrated.py        # trainer for the SHIPPED model → writes models/*.joblib
├── train_real_credit_nlp.py     # alt trainer: structured + unstructured → models/real/ (not served)
├── train_real_credit.py         # alt trainer: structured-only baseline → models/real/ (not served)
├── tune_model.py                # Optuna hyperparameter tuning
├── evaluate_model.py            # held-out evaluation
├── run_all_models.py            # train/evaluate model zoo
│
├── generate_*_data.py           # synthetic data generators (demo / MSME / realistic)
│
├── models/                      # ✅ shipped model the app loads at startup
│   ├── xgb_model.joblib         #    XGBoost (10 features, monotonic constraints)
│   ├── calibrator.joblib        #    isotonic probability calibration
│   ├── imputer.joblib           #    median imputation
│   ├── shap_explainer.joblib
│   ├── feature_list.joblib
│   ├── performance_report.json
│   └── real/                    #    alt artifacts from train_real_credit*.py (not served)
│
├── evaluation_report/           # ROC/PR curves, confusion matrices, dashboards (served at /reports)
│
├── static/
│   ├── index.html               # modern single-page dashboard (live-wired)
│   ├── index-classic.html       # previous dashboard (kept as backup)
│   └── evaluation_report.html   # model-performance chart gallery
│
├── requirements-runtime.txt     # minimal pinned deps to serve the app
├── requirements.txt             # full deps (training + NLP + LLM)
└── .env.example                 # Copilot LLM keys (optional)
```

---

## 🎯 Problem Statement (Track 04)

Existing MSME risk prediction relies on fragmented rule-based models with only **16–22% accuracy**.
The goal: a robust predictive solution that flags loan stress **12 months in advance** and lifts the
detection/capture rate toward **90%**. This engine reframes the problem from retrospective static
scoring to **forward-looking behavioural-signal monitoring**, surfacing early warnings the moment a
borrower's trajectory bends toward default.

---

## 🔬 Model Notes

- **Data:** synthetic MSME dataset (50,000 borrowers) — 10 leading-indicator features + officer-notes NLP. Defaults are sampled *probabilistically* from a noisy latent risk index, so the classes genuinely overlap (a realistic, non-separable problem — not a label-conditioned toy set).
- **Pipeline:** median imputation → XGBoost (monotonic constraints) → isotonic calibration on a held-out calibration split; evaluated on a separate held-out test split.
- **Headline (calibrated, held-out):** ROC-AUC **0.927**, PR-AUC 0.763, Accuracy 90%, Precision 73%, Recall 65%, Brier 0.074 — structured-only 0.898, **+0.025** from the unstructured NLP signal.
- **MSME variant:** `main_msme_backup.py` preserves the original GST/EMI/cashflow behavioural framing.

> The model bundled in `models/` is trained by `retrain_calibrated.py`; `GET /model-performance`
> serves `models/performance_report.json` — the true held-out metrics for the shipped artifact.

---

## 📄 License

Released under the **MIT License** — see below.

```
MIT License — © 2026 Kushal Mohan
Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files, to deal in the Software without restriction.
```

---

<p align="center"><i>Built for the IDBI Credit Risk Intelligence challenge — Track 04.</i></p>
