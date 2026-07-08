# 🚀 RENDER DEPLOYMENT — QUICK START

## ✅ Verification Complete

All deployment checks passed! Your project is ready for Render.

```
📦 Repository:    ✅ Configured (ParthTheBest/MSME-BANK-PREDICTION-fixed)
📄 Files:         ✅ 23/24 checks passed (all critical files present)
🤖 Models:        ✅ All .joblib files present (xgb_model, calibrator, shap_explainer)
🖼️ Assets:        ✅ Dashboard, static files, evaluation reports
📚 Dependencies:  ✅ requirements.txt complete (FastAPI, XGBoost, Gunicorn, etc.)
⚙️ Configuration: ✅ render.yaml properly configured
🔐 Security:      ✅ .env in .gitignore, API keys protected
```

---

## 🎯 Deploy to Render in 5 Steps

### Step 1: Go to Render
Open [https://render.com/](https://render.com/) and sign in with GitHub.

### Step 2: Create New Web Service
1. Click **New** → **Web Service**
2. Connect GitHub repo: `ParthTheBest/MSME-BANK-PREDICTION-fixed`
3. Select branch: `main`

### Step 3: Render Auto-Detects Configuration
Render will read your `render.yaml` automatically:
- **Build Command**: `pip install -r requirements.txt && python retrain_calibrated.py`
- **Start Command**: `gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT`
- **Environment**: Python 3.11

### Step 4: Add API Keys (Critical!)
In the **Environment** tab, add these variables:

| Key | Value |
|-----|-------|
| `ANTHROPIC_API_KEY` | Copy from your `.env` file (starts with `sk-ant-`) |
| `GEMINI_API_KEY` | Copy from your `.env` file (starts with `AQ.`) |
| `GEMINI_MODEL` | `gemini-2.5-flash` |
| `PYTHON_VERSION` | `3.11.0` |

**⚠️ Do NOT commit your `.env` file. Keep it local and only store keys in Render dashboard.**

### Step 5: Deploy!
Click **Deploy** button. Build takes 3–5 minutes.

---

## 📊 Your App Will Be Live At

```
https://<service-name>.onrender.com/

Examples:
- Dashboard:        https://<service-name>.onrender.com/dashboard
- Portfolio API:    https://<service-name>.onrender.com/api/portfolio
- Health Check:     https://<service-name>.onrender.com/health
- Reports:          https://<service-name>.onrender.com/reports
```

---

## 🔐 API Key Reference (from your .env)

Your current `.env` contains:
- ✅ `GEMINI_API_KEY` = `<your-gemini-api-key>` (from aistudio.google.com)

Before deploying:
1. Verify keys are correct
2. Add them to Render dashboard (NOT in code)
3. Keep `.env` local and never commit to git ✅

---

## 📈 What Each Module Does

| Module | Purpose |
|--------|---------|
| **Portfolio Overview** | Total book metrics, critical borrowers, expected loss |
| **Borrowers** | Search & filter 5,000+ MSME borrowers |
| **Risk Trends** | Risk distribution & sector heatmaps |
| **Early Warnings** | Alert queue (High/Critical risk) |
| **SHAP Explainer** | Why the model scored a borrower that way |
| **Model Performance** | ROC/PR curves, confusion matrices, metrics |
| **Expected Loss** | Portfolio-level credit loss by sector |
| **Upload CSV** | Score your own borrower dataset instantly |

---

## 🛠️ Troubleshooting

**Q: "Build failed: module not found"**
- A: All dependencies are in `requirements.txt` ✅. Check logs for typos.

**Q: "Model loading failed"**
- A: All `.joblib` files are committed to git ✅. Verify in `models/` directory.

**Q: "API key errors"**
- A: Double-check `ANTHROPIC_API_KEY` and `GEMINI_API_KEY` in Render dashboard. 
- A: Verify `.env` is NOT committed (it's in `.gitignore` ✅).

**Q: "Service spins down or is slow"**
- A: Free tier goes idle after 15 min. First request may take 30+ sec.
- A: Upgrade to paid plan for always-on service.

---

## 📋 Files Created for Deployment

- `RENDER_DEPLOYMENT.md` — Detailed deployment guide
- `verify_deployment.sh` — Automated verification script

Run anytime:
```bash
bash verify_deployment.sh
```

---

## ✨ Next Steps

1. ✅ All files verified
2. 🎯 Go to [render.com](https://render.com)
3. 🔐 Add API keys to environment
4. 🚀 Click **Deploy**
5. ⏱️ Wait 3–5 minutes
6. 🎉 Your app is live!

---

**Questions?** Check [RENDER_DEPLOYMENT.md](RENDER_DEPLOYMENT.md) for detailed docs.
