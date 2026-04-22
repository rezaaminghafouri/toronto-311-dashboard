# Deploying the Toronto 311 Dashboard Online

This guide covers deploying `dashboard_cloud.py` to **Railway** (recommended — free trial, no credit card for small apps).

---

## What you need

- A [GitHub](https://github.com) account (free)
- A [Railway](https://railway.app) account (free trial — $5 credit, no card required)
- The following 5 files from this folder:
  - `dashboard_cloud.py`
  - `fetch_open_data.py`
  - `ward_boundaries.geojson`
  - `requirements_cloud.txt`
  - `Procfile` (create this — see step 2 below)

---

## Step 1 — Create a GitHub repo

1. Go to [github.com/new](https://github.com/new)
2. Name it something like `toronto-311-dashboard`
3. Set it to **Public** (Railway free tier works with public repos)
4. Click **Create repository**
5. Upload the 5 files listed above (drag and drop them in, then click **Commit changes**)

---

## Step 2 — Add a Procfile

Create a plain text file called `Procfile` (no extension) with this exact content:

```
web: gunicorn dashboard_cloud:server
```

Upload it to the same GitHub repo.

---

## Step 3 — Deploy on Railway

1. Go to [railway.app](https://railway.app) and sign in with GitHub
2. Click **New Project** → **Deploy from GitHub repo**
3. Select your `toronto-311-dashboard` repo
4. Railway auto-detects Python and installs `requirements_cloud.txt`
5. Under **Settings → Environment**, add this variable:
   - Key: `PORT` — Railway sets this automatically, so you can skip this
6. Click **Deploy**

The first deploy takes 3-5 minutes. Railway will show build logs in real time.

---

## Step 4 — Get your public URL

Once deployed, Railway gives you a URL like:

```
https://toronto-311-dashboard-production.up.railway.app
```

That's it — anyone with the link can open the dashboard in their browser. No Python, no installation.

---

## How the data works

On first load, the app downloads all Toronto 311 data (~92 MB compressed) from the City of Toronto Open Data API. This takes about 3-5 minutes. After that, it caches everything as a Parquet file and reloads in ~2 seconds.

The cache resets every 24 hours, or when Railway restarts the container.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Build fails with "No module named X" | Add the missing package to `requirements_cloud.txt` and redeploy |
| App crashes on first load | Check Railway logs — usually a memory issue during data download |
| Map doesn't show ward boundaries | Make sure `ward_boundaries.geojson` is in the repo root |
| "Application error" page | Click **View Logs** in Railway dashboard to see the Python traceback |

---

## Alternative: Render.com

Render is another free option. Steps are nearly identical:

1. Create account at [render.com](https://render.com)
2. New → Web Service → Connect GitHub repo
3. Build command: `pip install -r requirements_cloud.txt`
4. Start command: `gunicorn dashboard_cloud:server`
5. Instance type: **Free**

Render free tier spins down after 15 minutes of inactivity. First visit after idle takes ~30 seconds to wake up. Railway doesn't have this limitation on the trial credit.
