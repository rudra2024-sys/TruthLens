# TruthLens — Project Rules & Architecture

TruthLens is an AI-generated media detection platform (FYP, VIT Mumbai, Dept. of IT, 2025-26).
Upload an image/video/audio file → get a verdict (REAL/FAKE/UNCERTAIN or authentic/suspicious/
manipulated), a confidence score, and a downloadable PDF forensic report.

This file is the single source of truth for how the codebase is laid out and which parts are
real ML vs. heuristics vs. dead/placeholder code. **Read it before making changes** — several
past sessions have found the top-level README.md to be stale; trust this file and the actual
code over README claims.

---

## 1. Repo layout

```
tl/
├── backend/            The ACTIVE FastAPI backend (this is what production runs)
│   └── app/
│       ├── api/routes/     upload.py, detect.py, dashboard.py, report.py, auth.py
│       ├── core/           config.py (Settings/.env), database.py, security.py (JWT/bcrypt)
│       ├── models/         models.py — all SQLAlchemy models
│       ├── schemas/        schemas.py — Pydantic response models
│       ├── pipelines/image/   ConvNeXt-Tiny inference (the real trained image model)
│       └── services/
│           ├── image/detector.py    → calls pipelines/image (ConvNeXt-Tiny)
│           ├── video/detector.py    → calls services/heuristics.py (byte-entropy heuristic)
│           ├── audio/detector.py    → calls services/models/model_client.py → AASIST (ONNX)
│           ├── models/               pretrained model code (AASIST, MesoNet) + model_client.py
│           ├── report/generator.py   real PDF generation (ReportLab), not a stub
│           └── ml_pipeline.py        DEAD CODE — see §3
├── frontend/           The ACTIVE React/Vite frontend
│   └── src/
│       ├── App.jsx              the real router — THIS defines which pages are live
│       ├── Home.jsx, Verify.jsx, History.jsx, Reports.jsx, About.jsx   live pages
│       ├── Navigation.jsx, Footer.jsx    live layout chrome
│       ├── api/client.js        the API client App.jsx actually imports and uses
│       ├── client.js            DEAD CODE duplicate at src root — not imported anywhere
│       └── pages/                DEAD CODE — entire directory unused, see §3
├── models/             A SEPARATE, undeployed FastAPI microservice scaffold
│                       ("TruthLense Real Model Inference Service"). Has its own
│                       Dockerfile/requirements.txt but is NOT a service in docker-compose.yml
│                       and is not running. `model_client.py` in backend/ would call it via
│                       MODEL_SERVICE_URL if it were running; since it isn't, calls always
│                       fail and fall back to the local pretrained model code in
│                       backend/app/services/models/. Do not confuse this with backend/.
├── legacy/             Old pretrained weights + model code, superseded by the above. Not
│                       imported by the active backend or the models/ microservice.
├── docker-compose.yml  Defines only `backend` + `frontend` services (no `models` service)
└── .venv/, venv/       Two separate local Python virtualenvs at repo root (not committed)
```

**Rule of thumb for "is this file live?"**: for frontend, check what `App.jsx` imports and
routes to. For backend, check what `main.py` includes and what each detector.py actually
imports — do not assume a file matters just because it looks purposeful or is named
confidently.

---

## 2. API architecture

FastAPI app in `backend/app/main.py`, all routers mounted under `/api/v1`:

| Route | Method | Purpose |
|---|---|---|
| `/api/v1/upload/` | POST | Upload a file (multipart), validates type/size, stores to disk + DB row |
| `/api/v1/upload/{upload_id}` | GET | Fetch upload metadata |
| `/api/v1/detect/{upload_id}` | POST | Run detection for an upload, dispatches by media_type |
| `/api/v1/detect/{upload_id}/result` | GET | Fetch a previously-computed detection result |
| `/api/v1/history` | GET | Recent uploads + their verdicts (dashboard/history page) |
| `/api/v1/stats` | GET | Aggregate counts by verdict/media type (dashboard) |
| `/api/v1/report/{upload_id}` | GET | Generates (or regenerates) and streams the PDF report |
| `/api/v1/auth/signup` | POST | Create account, returns JWT |
| `/api/v1/auth/login` | POST | Returns JWT |
| `/api/v1/auth/me` | GET | Current user (bearer token required) |

CORS is currently wide open (`allow_origins=["*"]`) in `main.py` — acceptable for FYP/dev,
worth tightening before any real production exposure, but out of scope unless asked.

A global exception handler in `main.py` converts any unhandled exception into a JSON 500
instead of a bare crash — keep this when touching `main.py`.

---

## 3. What's real vs. dead/placeholder — READ THIS BEFORE TOUCHING DETECTION CODE

### Genuinely trained (by this team)
- **Image — ConvNeXt-Tiny**: `backend/app/pipelines/image/` (model.py/inference.py/preprocessing.py).
  Checkpoint: **`models/checkpoints/image/convnext_tiny_best.pth`** (~319MB, gitignored — not
  in version control, must be provisioned locally/on the deploy target). Checkpoint dict
  contains `model_state_dict`, `epoch`, `best_val_accuracy` — a real training run, evaluated
  on the CIFAKE benchmark. This is the active image detector, wired in
  `backend/app/services/image/detector.py`.
- **Audio — Wav2Vec2 + LCNN**: exists **separately**, not yet integrated into this codebase.
  Will be integrated later — do not wire it in without explicit instruction.

### Pretrained (not trained by this team, but real trained DNNs)
- **Audio — AASIST** (Jung et al., ICASSP 2022), pretrained on ASVspoof2019-LA, ONNX weights
  at `backend/app/services/models/weights/aasist.onnx` (small, ~1.6MB — kept in git; see
  `.gitignore` comments). This is the currently active audio detector, wired via
  `backend/app/services/audio/detector.py` → `model_client.run_audio_model` →
  `services/models/audio_model.py` → `sessions.py` (`aasist_predict_logits`).
  **Known labeling issue**: the DB (`AudioAnalysis.wav2vec_score` / `.lcnn_score`) and the PDF
  report ("Spectral Entropy Score" / "Amplitude / Packing Score") both mislabel what is
  actually AASIST's spoof probability and its cross-window standard deviation. The numbers
  are real model output — the *labels* are wrong/misleading. Fixing this is a labeling-only
  fix (report text / possibly response schema field names), not an ML change.
- **MesoNet** (Meso4 + MesoInception4, DF-trained, ONNX) — real pretrained model for
  face-forgery detection, present in `backend/app/services/models/{sessions,image_model,
  video_model}.py` and wired into `model_client.run_image_model` / `run_video_model`. **Not
  used by the active detectors** — `image/detector.py` calls ConvNeXt directly and
  `video/detector.py` calls the heuristic directly, bypassing `model_client` entirely for
  both. MesoNet is present but currently dead in the live request path. A further audio model
  is also currently being trained separately and is not yet in this repo.

### Heuristic (real computation, no trained model)
- **Video**: `backend/app/services/heuristics.py::analyze_video()` — samples 12 raw byte
  windows from the file and computes Shannon entropy statistics over those bytes. This does
  **not** decode frames, faces, or any actual video signal. Labeled in the DB/UI as
  `model_used: "Video forensic heuristics v1"` — that label is accurate (it says
  "heuristics"), but the PDF report's "Structure Entropy Score" / "Byte Consistency Score"
  language reads more forensic than the underlying computation is. This is **not** the final
  video ML solution and is expected to be replaced later — do not present it to users as
  equivalent in rigor to the image/audio pipelines.

### Placeholder / mock / dead code — do not wire these in
- `backend/app/services/ml_pipeline.py` — a `LightweightMLPipeline` class with hardcoded,
  fabricated weights (comment literally says "mimicking weights learned from backprop on
  synthetic datasets"). **Not imported anywhere.** Dead code from an earlier phase.
- `frontend/src/pages/*` (Analytics.jsx, AnalyticsPage.jsx, Dashboard.jsx, HistoryPage.jsx,
  Models.jsx, ModelsPage.jsx, Scan.jsx, SettingsPage.jsx) — **none of these are imported by
  `App.jsx`**. The entire `pages/` directory is currently dead/unreferenced.
- `frontend/src/client.js` (root-level, distinct from `src/api/client.js`) — **now fully
  dead**. Until the Phase 3 hardening pass (see git history), `Home.jsx`, `Verify.jsx`,
  `History.jsx`, and `Reports.jsx` imported from this file instead of `src/api/client.js`,
  which meant those four pages' requests never got the JWT `Authorization` header that
  `src/api/client.js`'s axios interceptor attaches, and used a version of `friendlyError()`
  that didn't handle array-shaped 422 validation errors. All four now import from
  `src/api/client.js` (the superset). `client.js` itself was left in place rather than
  deleted — flagged here per the "don't delete without flagging" rule.
- `models/` (top-level microservice) and `legacy/` — see §1. Not part of the active request
  path.

**Do not delete any of the above without flagging it first and getting explicit confirmation**
— some of it may be salvageable for Phase 5 (fusion/ensemble prep) or may be in active use by
a workflow this file doesn't know about.

---

## 4. Authentication architecture

JWT bearer auth, already tested and considered **good enough** — do not rewrite or redesign it.
Only touch it for a concrete, reproducible bug.

- `backend/app/core/security.py`: bcrypt password hashing (`passlib`), JWT via `python-jose`
  (`HS256`), 24h expiry. Secret key comes from `AUTH_SECRET_KEY` env var, falling back to a
  hardcoded dev default (`"change-this-development-secret-key"`) if unset — **must** be set to
  a real secret before any real deployment.
- `backend/app/api/routes/auth.py`: `/signup`, `/login`, `/me`. Signup enforces min 8-char
  password and unique email; login checks `is_active`.
- Frontend: token + user cached in `localStorage` (`truthlens_token`, `truthlens_user`),
  attached to every request via an axios interceptor in `src/api/client.js`. `App.jsx` has
  `ProtectedRoute` (redirects to `/login` if no valid token) and `GuestRoute` (redirects
  authenticated users away from `/login`/`/signup`) wrapper components, plus inline `Login`/
  `SignUp` components (not separate files).

---

## 5. Environment variables

| Variable | Where used | Default | Notes |
|---|---|---|---|
| `DATABASE_URL` | `core/config.py` | `sqlite+aiosqlite:///./truthlens.db` | |
| `UPLOAD_DIR` / `REPORT_DIR` | `core/config.py` | `uploads` / `reports` | relative to backend cwd |
| `MAX_FILE_SIZE_MB` | `core/config.py` | `100` | |
| `FAKE_THRESHOLD` | `core/config.py` | `0.5` | verdict cutoff, used by video/audio `_verdict()` |
| `IMAGE_MODEL_CHECKPOINT` | `pipelines/image/model.py` | `backend/checkpoints/image/convnext_tiny_best.pth` | override to point at `models/checkpoints/image/convnext_tiny_best.pth` when running outside Docker |
| `IMAGE_MODEL_DEVICE` | `pipelines/image/model.py` | `auto` (cuda if available, else cpu) | `cpu`/`cuda` |
| `MODEL_SERVICE_URL` | `services/models/model_client.py` | `http://models:8001` | only relevant if the `models/` microservice is actually deployed as a docker-compose service, which it currently is not |
| `AUTH_SECRET_KEY` | `core/security.py` | insecure dev default | **must** override for any real deployment |
| `DEBUG` | `core/config.py` | `true` | |

No `.env` file currently exists in the repo (confirmed absent as of Phase 1 audit). `Settings`
reads from a `.env` if present (`core/config.py`'s `Config.env_file = ".env"`).

---

## 6. How to run

**Backend** (no Docker required — a `.venv` with all deps already exists at repo root):
```bash
cd backend
../.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Set `IMAGE_MODEL_CHECKPOINT` to `../models/checkpoints/image/convnext_tiny_best.pth` (absolute
path recommended) if running outside Docker, since the default path assumes the Docker bind
mount at `backend/checkpoints/image/`.

**Frontend**:
```bash
cd frontend
npm install   # if node_modules isn't already present
npm run dev   # http://localhost:3000, proxies /api -> http://localhost:8000
```

**Docker** (both services):
```bash
docker-compose up --build
```
Requires Docker Desktop's engine to actually be running — `docker-compose up` will fail to
connect otherwise (seen in this environment).

---

## 7. Testing expectations

- There is no frontend test script (`package.json` has no `test` entry) and no formal backend
  test suite runner configured — existing `test_*.py` files (`backend/test_image_detection.py`,
  root `smoke_test.py`/`test_detection.py`, `frontend/smoke_test.py`,
  `models/test_image_pipeline.py`) are standalone scripts, not wired into CI.
- Before claiming any fix works: actually exercise it. For frontend bugs, that means driving
  the real UI flow (upload → detect → render, or navigate → click) in a browser and checking
  the console for errors — a passing `curl /health` does not prove a UI path works. For backend
  bugs, hit the actual endpoint with a real request and check the response body/status code.
- Do not claim a page/route/feature works unless you observed it working in this session.

---

## 8. Hard rules for future changes

1. **ML model changes require explicit approval.** Do not retrain, swap architectures, change
   weights, change detection thresholds, integrate a new model (including the already-trained
   Wav2Vec2+LCNN audio model or any future model), or activate currently-dead model paths
   (MesoNet, `ml_pipeline.py`) without the user explicitly asking for that specific change in
   that conversation.
2. **Existing working functionality must not be casually rewritten.** Prefer small, targeted
   fixes over rewrites. Before changing a file, understand how it's currently used (check
   imports/callers) rather than assuming from the filename or a docstring.
3. **Do not delete files just because they look unused** (see §3's dead-code list) — report
   them and get confirmation first.
4. **Authentication is not to be redesigned** — bug fixes only, and only for concrete,
   reproduced issues.
5. **Do not fabricate model output, statistics, or forensic explanations in the UI.** If a
   feature isn't backed by real backend data, the UI should say so rather than show invented
   numbers.

---

## 9. Future multi-model integration (Phase 5 notes)

`backend/app/services/detection_interface.py` sketches a common `DetectionResult` shape and
`ImageDetector`/`AudioDetector`/`VideoDetector`/`FusionDetector` `Protocol`s for a future
ensemble layer. **It is not imported anywhere and changes no current behavior** — it exists
only as a design reference.

Actually making the three live detectors conform to it (and building a real fusion layer) was
deliberately **not done**, because `detect.py`'s dispatch is simple if/elif routing to three
detectors that each return a different bespoke shape, and normalizing all of that
simultaneously is a multi-file refactor of currently-correct, working code — real regression
risk, not an additive change. Treat that as a separate, explicitly-approved follow-up task, not
something to pick up incidentally while touching nearby code.
