# TruthLens — AI-Generated Media Detection Platform

VIT Mumbai | Department of Information Technology | 2025-26

---

## What actually works right now (verified end-to-end)

- ✅ Upload images, video, or audio through a real drag-and-drop UI
- ✅ File validation (type, size, empty-file checks) with real error messages
- ✅ Full backend pipeline: upload → store → route to image/video/audio detector → save to DB → return result
- ✅ Real SQLite database persisting every upload and result
- ✅ Real dashboard (stats computed from the DB, not hardcoded)
- ✅ Real history log of every scan
- ✅ Real downloadable PDF forensic report (generated server-side with ReportLab)
- ✅ Proper error handling — network errors, validation errors, and server errors all surface as readable messages, with retry
- ✅ No unhandled exceptions — the previous "Something went wrong" bug is fixed (see below)

## What is NOT yet real

The **detection scores themselves** (EfficientNet, XceptionNet, wav2vec2, FFT, LCNN) are currently
**deterministic placeholder values**, not the output of trained models. The architecture, database
schema, API contracts, and UI are all built to receive real model output — but no model has been
trained or plugged in yet. This was always the agreed phased plan (skeleton → image → audio → video),
and training real models is genuinely multi-day work per pipeline (dataset download, GPU training,
evaluation) that cannot be done inside a chat response.

**Where to plug in a real model:** `backend/app/services/{image,video,audio}/detector.py` — each file
has a single clearly-marked block computing the stub scores. Replace that block with real inference
and everything downstream (DB, API, dashboard, PDF report) works unchanged.

---

## The bug that was breaking every upload

The previous version returned a SQLAlchemy object directly from `POST /detect/{id}` without eagerly
loading its related scoring tables. Under async SQLAlchemy, touching a not-yet-loaded relationship
outside of an active DB session throws a `MissingGreenlet` error — a real 500 error that the frontend
had no way to explain, so it just showed "Something went wrong."

**Fix:** all relationships now declare `lazy="selectin"`, so they're always loaded safely regardless
of which route touches them. The main app also has a global exception handler so any future backend
error returns a readable message instead of a silent failure.

---

## Setup

### Backend (Docker)
```bash
docker-compose up --build
```
Runs at `http://localhost:8000` — Swagger docs at `http://localhost:8000/docs`

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Runs at `http://localhost:3000` (or next free port if 3000 is taken)

---

## Testing it end-to-end

1. Open the app, go to the **Scan** tab
2. Drop any JPG/PNG/MP4/WAV file
3. Watch it upload (progress bar) → detect (scan animation) → show a full result: verdict, confidence gauge, pipeline score breakdown, model used, processing time
4. Click **Download Forensic PDF Report** — a real PDF downloads
5. Go to **Dashboard** — see the scan reflected in stats and recent activity
6. Go to **History** — see the full log, download any past report

If any step fails, the UI will show the actual error text (not a generic message) — screenshot it and that tells us exactly what's wrong.

---

## Project structure

```
backend/app/
├── api/routes/       upload.py, detect.py, dashboard.py, report.py
├── core/              config.py, database.py
├── models/            SQLAlchemy models (all relationships eager-loaded)
├── schemas/           Pydantic response schemas
└── services/
    ├── image/         detector.py  ← plug in EfficientNet here
    ├── video/         detector.py  ← plug in XceptionNet here
    ├── audio/         detector.py  ← plug in wav2vec2 here
    └── report/         generator.py (real PDF generation, already working)

frontend/src/
├── components/NavShell.jsx
├── pages/             Dashboard.jsx, Scan.jsx, HistoryPage.jsx
└── api/client.js       all backend calls, centralized error handling
```

## Next real milestone

Train and plug in the **image pipeline** first (EfficientNet-B4 fine-tuned on CIFAKE — see the Dataset
Reference Guide). That's the fastest path to one fully real, trained pipeline, which is usually enough
for a strong FYP demo even before video/audio are trained.

## Developer note (local dev)

- The frontend dev server proxies `/api` to the backend. I fixed a misconfiguration where the
    proxy was pointing to port `8010`; it now targets `http://localhost:8000` which matches the
    backend in `docker-compose.yml`.
- If you're running the frontend directly (not via the dev server), ensure API calls reach
    `http://localhost:8000/api/v1` or update the hosting config appropriately.
- I could not create a git commit from this environment because the workspace is not a git
    repository here. Please commit the `vite.config.js` change locally if you want it tracked.
