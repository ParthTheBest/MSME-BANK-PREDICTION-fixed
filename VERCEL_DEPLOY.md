# 🚀 VERCEL DEPLOYMENT GUIDE

## ⚠️ Important Notes for Vercel

Vercel works best with Node.js/Next.js apps, but can run FastAPI using serverless functions:

- **Max deployment size**: 250 MB (we're ~100-150 MB with models)
- **Cold start**: First request may take 5-10 seconds
- **Free tier**: Sufficient for POC/demo
- **Model loading**: Models load on first request

---

## ⚡ Deploy to Vercel in 3 Steps

### Step 1: Push to GitHub
```bash
git push origin main
```

### Step 2: Go to Vercel
1. Open https://vercel.com/
2. Sign in with GitHub
3. Click **Add New** → **Project**
4. Select repo: `MSME-BANK-PREDICTION-fixed`
5. Vercel auto-detects Python from `vercel.json`

### Step 3: Add Environment Variables
In **Settings** → **Environment Variables**, add:

| Key | Value |
|-----|-------|
| `ANTHROPIC_API_KEY` | `sk-ant-...` (from .env) |
| `GEMINI_API_KEY` | `<your-gemini-api-key>` (from aistudio.google.com) |
| `GEMINI_MODEL` | `gemini-2.5-flash` |

Click **Deploy**!

---

## 🔗 Your Live App

Once deployed:
```
https://<project-name>.vercel.app/dashboard
```

Test health endpoint:
```bash
curl https://<project-name>.vercel.app/health
```

---

## 📊 Vercel vs Render

| Feature | Vercel | Render |
|---------|--------|--------|
| Cold Start | 5-10s | 1-2s |
| Best For | Frontend-heavy | Always-on services |
| Free Tier | ✅ Good | ✅ Good |
| Model Size | 250 MB max | No limit |
| Scaling | Auto | Auto |

**For this project**: Render is slightly better due to always-on capability, but Vercel works fine for POC!

---

## ✅ Files Added for Vercel

- `vercel.json` — Vercel configuration
- `main.py` — Added Mangum handler
- `requirements.txt` — Added Mangum

All ready to deploy!

---

## 🚨 If Deployment Fails

**Common issue**: Models too large
- **Solution**: Use `.vercelignore` to exclude non-essential files

**Cold start timeout**: 
- **Solution**: Upgrade to Vercel Pro for longer timeouts

**API Key not working**:
- Verify keys in Vercel dashboard
- Check `.env` is in `.gitignore`

---

**Ready? Go to https://vercel.com/ and deploy!**
