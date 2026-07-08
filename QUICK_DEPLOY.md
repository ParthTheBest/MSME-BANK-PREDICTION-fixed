# ⚡ INSTANT DEPLOYMENT GUIDE

## Option 1: Already Connected to Render? (If stuck loading)

### Just Click This in Render Dashboard:
1. Go to https://dashboard.render.com
2. Select your service: `msme-bank-prediction`
3. Click **Redeploy** (top-right button)
4. Wait 2-3 minutes ✅

The optimized config will deploy **much faster** now.

---

## Option 2: Not Yet on Render? (Fresh Deploy)

### Run These Commands:

```bash
# 1. Go to your project
cd /Users/gankai/Desktop/checklist/MSME-BANK-PREDICTION-fixed

# 2. Make deployment executable
chmod +x deploy-to-render.sh

# 3. Run auto-deploy
bash deploy-to-render.sh
```

Then enter:
- **Render API Key** (get from https://dashboard.render.com/settings)
- **Service ID** (find in your Render dashboard URL)

---

## Option 3: Manual (If scripts don't work)

1. Go to https://render.com/
2. **New** → **Web Service**
3. Connect GitHub: `MSME-BANK-PREDICTION-fixed` repo
4. Leave settings as-is (uses our `render.yaml`)
5. Add these secrets:
   - `ANTHROPIC_API_KEY`: (from your .env)
   - `GEMINI_API_KEY`: `<your-gemini-api-key>` (from aistudio.google.com)
   - `GEMINI_MODEL`: `gemini-2.5-flash`
6. Click **Deploy**

---

## ✅ What Changed (Faster Deploy)

| Before | After |
|--------|-------|
| Build: 5-10 min (retrains model) | Build: 1-2 min (uses pre-trained) |
| Start: Slow | Start: 2 workers, optimized |
| No health check | Health check: `/health` |
| Manual deploy | Auto-deploy on git push |

---

## 🔍 Once Live

Your app will be at: `https://<service-name>.onrender.com/dashboard`

Test it:
```bash
curl https://<service-name>.onrender.com/health
# Returns: {"status":"healthy","model_loaded":true}
```

---

**Pick an option above and let me know if you hit any issues!**
