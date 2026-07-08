# Render Deployment Guide — MSME Bank Prediction

## ✅ Pre-Deployment Verification Checklist

### 1. **Git Repository Status**
- ✅ Remote configured: `https://github.com/ParthTheBest/MSME-BANK-PREDICTION-fixed.git`
- ✅ All files committed (working tree clean)
- ✅ Branch: `main` (up-to-date with origin)

### 2. **Project Files Verified**
- ✅ `main.py` — FastAPI app (port 8000)
- ✅ `render.yaml` — Render configuration
- ✅ `requirements.txt` — All dependencies listed
- ✅ `requirements-runtime.txt` — Production runtime (minimal)
- ✅ `.env` — `.gitignore`-d (NOT committed) ✅
- ✅ `.gitignore` — Properly configured (excludes .env, __pycache__, data/)

### 3. **Model & Assets Present**
```
models/
├── xgb_model.joblib          ✅ (1.0 MB)
├── calibrator.joblib         ✅ (1.2 KB)
├── imputer.joblib            ✅ (583 B)
├── feature_list.joblib       ✅ (233 B)
├── shap_explainer.joblib     ✅ (2.6 MB)
├── shap_explainer_real.joblib ✅ (2.3 MB)
├── lr_baseline.joblib        ✅ (2 KB)
├── performance_report.json   ✅
└── bias_audit_results.json   ✅

static/
├── index.html               ✅ (SPA Dashboard)
├── index-classic.html       ✅
└── evaluation_report.html   ✅

evaluation_report/
├── roc_curves.png           ✅
├── pr_curves.png            ✅
├── confusion_matrices.png   ✅
├── score_distributions.png  ✅
├── metrics_comparison.png   ✅
├── all_models_results.json  ✅
└── [15+ visualization files] ✅
```

### 4. **API Keys & Environment Variables**
```yaml
Configured in render.yaml:
- ANTHROPIC_API_KEY        (from .env — secret on Render)
- GEMINI_API_KEY           (from .env — secret on Render)
- GEMINI_MODEL             (value: "gemini-2.5-flash")
- PYTHON_VERSION           (value: "3.11.0")
```

---

## 🚀 Deployment Steps

### Step 1: Connect GitHub to Render
1. Go to [https://render.com/](https://render.com/)
2. Sign in with GitHub
3. Click **New** → **Web Service**
4. Select repository: `ParthTheBest/MSME-BANK-PREDICTION-fixed`
5. Choose branch: `main`

### Step 2: Configure Service
The `render.yaml` is already configured. Render will auto-detect:
- **Environment**: Python
- **Build Command**: `pip install -r requirements.txt && python retrain_calibrated.py`
- **Start Command**: `gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT`
- **Plan**: Free tier (suitable for demo/POC)

### Step 3: Add Environment Variables to Render Dashboard
Go to **Environment** in the web service settings and add:

| Key | Value | Sync |
|-----|-------|------|
| `ANTHROPIC_API_KEY` | `sk-ant-...` (from your `.env`) | ❌ No |
| `GEMINI_API_KEY` | `AQ.Ab8RN6LV...` (from your `.env`) | ❌ No |
| `GEMINI_MODEL` | `gemini-2.5-flash` | ✅ Yes |
| `PYTHON_VERSION` | `3.11.0` | ✅ Yes |

**⚠️ WARNING**: Do NOT commit `.env` to git. Keep API keys secret on Render's dashboard.

### Step 4: Deploy
1. Click **Deploy** in Render dashboard
2. Monitor build logs (should take 3–5 min)
3. Wait for "Live" status

### Step 5: Verify Deployment
Once live, your app will be at: `https://<service-name>.onrender.com/`

Test endpoints:
- Dashboard: `https://<service-name>.onrender.com/dashboard`
- API Health: `https://<service-name>.onrender.com/health` (if defined)
- Portfolio Overview: `https://<service-name>.onrender.com/api/portfolio`

---

## 🔐 API Key Security Best Practices

### ✅ CORRECT (Secure)
```bash
# Store in Render Environment Variables (dashboard)
# Never commit to git
# .env is in .gitignore ✅
```

### ❌ INCORRECT (Unsafe)
```bash
# ❌ Do NOT hardcode in code
# ❌ Do NOT commit to GitHub
# ❌ Do NOT share in messages
```

### Current Status:
- ✅ `.env` is in `.gitignore`
- ✅ `.env` is NOT committed to git
- ✅ `.env` contains sensitive keys (GEMINI_API_KEY)
- ✅ `render.yaml` references these via environment variables

---

## 📦 Dependency Verification

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | latest | Web framework |
| `uvicorn` | latest | ASGI server |
| `gunicorn` | latest | Production WSGI server |
| `xgboost` | latest | ML model |
| `scikit-learn` | latest | ML utilities |
| `joblib` | latest | Model serialization |
| `pandas` | latest | Data processing |
| `numpy` | latest | Numerical computing |
| `shap` | latest | Model explainability |
| `sentence-transformers` | latest | NLP features |
| `anthropic` | latest | Claude API (optional) |
| `google-genai` | latest | Gemini API (optional) |
| `python-dotenv` | latest | Env var loading |
| `reportlab` | latest | PDF generation |

**Install Check:**
```bash
pip install -r requirements.txt
```

---

## 🐛 Troubleshooting

### Build Fails: "module not found"
- Ensure `requirements.txt` is in root directory ✅
- Check Python version: 3.11+ required ✅

### Model Loading Fails
- Verify `models/*.joblib` files are committed ✅
- Check file permissions: `ls -la models/xgb_model.joblib` ✅

### API Key Issues
- Verify `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` are set in Render dashboard
- Check `.env` is NOT committed (and is in `.gitignore`) ✅
- Verify `python-dotenv` is in `requirements.txt` ✅

### Slow Startup (Free Tier)
- Free tier spins down after 15 min inactivity
- First request after spin-down may take 30+ seconds
- Upgrade to paid plan for always-on

---

## 📋 Final Checklist Before Deploy

- [ ] GitHub remote is `https://github.com/ParthTheBest/MSME-BANK-PREDICTION-fixed.git`
- [ ] All files committed: `git status` shows clean tree
- [ ] `render.yaml` exists in root with correct config
- [ ] `requirements.txt` exists in root
- [ ] Model files exist in `models/` directory
- [ ] Static files exist in `static/` directory
- [ ] `.env` is in `.gitignore` (NOT committed) ✅
- [ ] Render account connected to GitHub
- [ ] Environment variables set in Render dashboard
- [ ] Ready to click **Deploy**!

---

## 🎯 Post-Deployment

1. **Monitor**: Check Render logs for errors
2. **Test**: Visit dashboard and verify functionality
3. **Scale**: If needed, upgrade from Free → Pro plan
4. **Update**: Push changes to `main` branch to auto-deploy

---

**Deployment Ready! ✅**
