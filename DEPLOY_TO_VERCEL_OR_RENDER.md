# 🌐 DEPLOY TO VERCEL (or Render)

## ⚡ VERCEL DEPLOYMENT (Fastest)

### 1. Go to Vercel
Open https://vercel.com/ and sign in with GitHub

### 2. Create New Project
- Click **Add New** → **Project**
- Select: `MSME-BANK-PREDICTION-fixed`
- Click **Import**

### 3. Configure Environment
Click **Environment Variables** and add:
```
ANTHROPIC_API_KEY = sk-ant-... (from .env)
GEMINI_API_KEY = <your-gemini-api-key> (from aistudio.google.com)
GEMINI_MODEL = gemini-2.5-flash
```

### 4. Deploy
Click **Deploy** — Done! ✅

**Your app:** `https://<project-name>.vercel.app/dashboard`

---

## ⚡ RENDER DEPLOYMENT (Alternative)

### 1. Go to Render
Open https://render.com/ and sign in with GitHub

### 2. Create New Web Service
- Click **New** → **Web Service**
- Connect: `MSME-BANK-PREDICTION-fixed` repo

### 3. Add Environment Variables
```
ANTHROPIC_API_KEY = sk-ant-...
GEMINI_API_KEY = <your-gemini-api-key>
GEMINI_MODEL = gemini-2.5-flash
```

### 4. Deploy
Click **Deploy** — Done! ✅

**Your app:** `https://<service-name>.onrender.com/dashboard`

---

## 📊 Which Should You Use?

| Aspect | Vercel | Render |
|--------|--------|--------|
| **Ease** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Speed** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Cold Start** | 5-10s | 1-2s |
| **Free Tier** | ✅ | ✅ |
| **Best For** | Quick deployment | Always-on service |

**Recommendation:** Start with **Vercel** (easier setup), upgrade to **Render** if you need faster cold starts.

---

## ✅ What's Configured

Both platforms will auto-detect and use:
- `main.py` (FastAPI app)
- `requirements.txt` (all dependencies including Mangum)
- `vercel.json` or `render.yaml` (configuration)
- `models/` (pre-trained XGBoost model)
- `static/` (dashboard UI)

---

## 🚀 Manual Deploy Script (Optional)

If you want to deploy via command line:

```bash
# Vercel
npm i -g vercel
vercel

# Render
bash deploy-to-render.sh
```

---

## 🔥 Your Dashboard Will Have:

- 📊 Portfolio Overview (metrics, borrowers)
- 🔍 Borrower Search & Filter
- 📈 Risk Trends & Heatmaps
- ⚠️ Early Warnings (alerts)
- 🧠 SHAP Explainer (why each score)
- 📉 Model Performance (ROC, PR curves)
- 💸 Expected Loss (by sector)
- 📤 CSV Upload (score your data)

---

## 🎯 Test Your Deployment

Once live, test the API:

```bash
# Health check
curl https://<your-app>/health

# Portfolio overview
curl https://<your-app>/api/portfolio

# Dashboard
Open: https://<your-app>/dashboard
```

---

## 🆘 Troubleshooting

**Q: Deployment fails with "models not found"**
- A: Models are in `models/` directory and committed to git ✅

**Q: API keys not working**
- A: Verify in platform dashboard (Vercel Settings / Render Environment)
- A: Make sure `.env` is in `.gitignore` ✅

**Q: Cold start timeout**
- A: Normal for first request (5-10s on Vercel, 2-3s on Render)
- A: Upgrade to paid plan for guaranteed performance

**Q: Can't connect to GitHub**
- A: Verify GitHub permissions are granted to Vercel/Render

---

**Ready? Pick Vercel or Render above and deploy in 2 minutes!** 🚀
