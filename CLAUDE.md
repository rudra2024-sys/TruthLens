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
│       ├── pipelines/image/       ConvNeXt-Tiny inference (real trained image model)
│       ├── pipelines/image_clip/  CLIP ViT-B/16 + trained head, second opinion (see §3)
│       └── services/
│           ├── image/detector.py    → calls pipelines/image + pipelines/image_clip, averages both
│           ├── video/detector.py    → calls video/backend.py (swappable; default: model_v1/optimized.py)
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
- **Image — ConvNeXt-Tiny + CLIP ViT-B/16 ensemble** (round 3 deployed 2026-09-16, see git
  history / session notes for the full before/after trail): `backend/app/pipelines/image/`
  (ConvNeXt-Tiny) and `backend/app/pipelines/image_clip/` (frozen CLIP backbone + trained MLP
  head), combined in `backend/app/services/image/detector.py` by taking the **max** of each
  model's FAKE probability (changed from averaging on 2026-09-16, see below for why).
  `model_used` on the detection result reads `"ConvNeXt-Tiny + CLIP ViT-B/16 (ensemble)"` when
  both ran.
  - **ConvNeXt-Tiny** checkpoint: **`models/checkpoints/image/convnext_tiny_diversified_v2.pth`**
    (~334MB, gitignored). Fine-tuned from the original CIFAKE-only checkpoint
    (`convnext_tiny_best.pth`, kept on disk as a rollback) on CIFAKE +
    `alessandrasala79/ai-vs-human-generated-dataset` + `xhlulu/140k-real-and-fake-faces` +
    `ayushmandatta1/deepdetect-2025` (StyleGAN3/DALL-E 3/Midjourney/SD3), with random JPEG
    re-compression / blur / resize-degradation augmentation during training. Per-source held-out
    accuracy: 99.4-100%. The once-kept intermediate `convnext_tiny_diversified.pth` (the
    pre-deepdetect-2025 round) was deleted 2026-09-16 during a disk-space cleanup — only the
    original baseline and this v2 checkpoint remain as rollback points. **Not retrained since**
    — still blind to the portrait-app genre below (see round 3).
  - **CLIP second opinion** checkpoint: **`models/checkpoints/image_clip/clip_head_round3_portrait_app.pth`**
    (~530KB, deployed 2026-09-16). Round 2 (`clip_head_best.pth`, same directory) is kept
    alongside it as a rollback. Frozen CLIP ViT-B/16 backbone (downloads via `open_clip_torch` on
    first use, cached in the `clip_backbone_cache` Docker volume), trained head only. Round 3
    added a fifth, fake-only source (`portrait_app_fake_source_v1.zip`, see below) to the same
    four datasets ConvNeXt uses and retrained just the head in
    `TruthLens_CLIP_SecondOpinion.ipynb` (`C:\Users\admin\Downloads\`) — 97.43% best val accuracy,
    held-out per-source: ai_vs_human 97.3%, cifake 95.3%, deepdetect2025 98.2%, faces 98.8%,
    portrait_app 84% (n=25, small holdout since the source itself is only 257 images).
  - `preprocessing.py` resizes directly to 224x224 — **do not reintroduce a 32x32-then-224x224
    "crush" step**. That was tried on 2026-09-11 on the theory that a diversified checkpoint no
    longer needed it, reverted based on a 2-image spot check that (misleadingly) favored keeping
    it, then reverted again once a proper 4-source x 3000-image held-out evaluation showed it is
    actively harmful at scale (dropped `faces` accuracy from 99.8% to 57%, barely better than
    chance, because it destroys real detail in datasets that aren't natively low-resolution like
    CIFAKE is). Any future preprocessing change needs a full per-source evaluation before being
    kept, not a 1-2 image spot check either way.
  - **Portrait-app blind spot: confirmed 2026-09-12, re-confirmed 2026-09-16, FIXED for CLIP,
    ConvNeXt still blind.** Stylized AI "portrait app" images (viral apps that turn a selfie into
    a fake vintage-family-photo or a painterly period-costume/armor portrait) used to be a total
    blind spot for both sub-models (0/8 on the original 8-image hard-case test, both models
    99.9%+ confident REAL, no disagreement). Root-caused to a real data gap: none of the four
    original training sources contain this genre. Closed via a curated fifth, fake-only source —
    `portrait_app_fake_source_v1.zip` (133.5MB, 257 images, currently at
    `C:\Users\admin\Downloads\`, not yet copied into the repo or uploaded anywhere durable):
    found by searching `poloclub/diffusiondb` (CC0, 14M real Stable Diffusion outputs from real
    Discord user prompts, not GAN-generated) **by prompt text** in its `metadata.parquet` index
    (dataset-title search on 2026-09-12 had found nothing) for vintage-photo / period-costume
    prompts (~4,200 precise matches), downloading the 25 highest-yield zip parts (766 candidate
    images), and manually visually reviewing all 766 down to 257 genuine matches (photorealistic
    vintage b/w portraits, painterly period-costume/armor portraits, Renaissance/Baroque
    oil-painting-style portraits, a "Victorian woman from behind" cluster — the last two clusters
    are heavily near-duplicated per the zip's own README, not yet deduplicated). Used to retrain
    only the CLIP head (see above) — raised CLIP's accuracy on this genre from 26.7% to 93.3% on
    a local 15-image sample (84% on Colab's own held-out split). **ConvNeXt was not retrained and
    is still blind** (~6.7% on the same local sample) — this is why the ensemble combination
    changed from average to max: averaging CLIP's fixed signal with ConvNeXt's still-blind one
    dragged the ensemble's accuracy on this genre back down to 66.7% (44% on Colab's held-out
    split), largely cancelling out the fix. Max lets either model's catch through undiluted, at a
    measured cost of ~0.9 points overall accuracy across the four original sources (more false
    FAKE flags on ordinary real photos) per the notebook's own strategy-comparison table — sanity
    checked 2026-09-16 against a 9-image real-photo set through the live API post-deployment,
    9/9 still correctly REAL, no regression observed at that sample size. **Revisit this
    average-vs-max trade-off if ConvNeXt is ever retrained on the same portrait-app source** —
    that would likely let averaging work again without the accuracy-cliff cost max currently
    carries. Test images and a standalone comparison script live at
    `backend/test_data/blind_spot_images/{portrait_app,chatgpt_everyday}/` and
    `backend/test_blind_spot_checkpoints.py` (not wired into CI, run manually).
  - **Separate, newly-confirmed blind spot, 2026-09-16, NOT addressed by round 3**: photorealistic
    "everyday" AI-generated images — ordinary, mundane real-photo recreations (a jewelry
    close-up, a temple selfie, a food photo, a mirror selfie) produced by asking ChatGPT/GPT-4o-
    class image generation to recreate real reference photos, with no stylization cues at all
    (unlike the portrait-app genre's vintage/costume look). Confirmed via 8 such images (in
    `backend/test_data/blind_spot_images/chatgpt_everyday/`): 0/8 correct across every
    model/checkpoint combination tested (ConvNeXt alone, CLIP round 2, CLIP round 3, and both
    ensemble strategies) — all 8 classified REAL at 99.99%+ confidence regardless. Root cause is
    presumably the same category of problem as the portrait-app gap (no training source contains
    this generator's output), but likely a harder data-sourcing problem than DiffusionDB was:
    GPT-4o-class outputs aren't collected in bulk public prompt datasets the way Stable Diffusion
    outputs are via DiffusionDB. No training data search has been attempted for this yet.
- **Audio — Wav2Vec2 + LCNN**: exists **separately**, not yet integrated into this codebase.
  Will be integrated later — do not wire it in without explicit instruction.
- **Video — Video Model v1 (EfficientNet-B0, epoch 11)**: `backend/app/services/video/model_v1/`
  (`common.py`/`baseline.py`/`optimized.py`). Checkpoint: **`backend/checkpoints/video/
  epoch_11_model_only.pt`** (~16MB, gitignored). This is now the **active production video
  detector**, wired via `backend/app/services/video/backend.py`'s `VideoModelV1Backend`
  (selected by default; override with `VIDEO_MODEL_BACKEND=heuristic` to fall back to the old
  byte-entropy heuristic below). Line-by-line verified against a recovered original Celeb-DF-v2
  evaluation script — see git history on `app/services/video/model_v1/` for the full
  parity-verification trail (frame sampling via `np.linspace(...).astype(int)`, Haar
  frontal-face detection with 20% padding and a center-square/empty-crop fallback, PIL
  `BILINEAR` resize — not `cv2.resize`, they differ — ImageNet normalization, 16-frame
  mean-logit pooling, sigmoid, threshold 0.525). `optimized.py` is the fast path (cached model
  + Haar cascade, batched forward pass, `grab()`-skip decoding); `baseline.py` is a slower,
  literal reference implementation kept for regression testing, not used in production.
  Videos shorter than 16 frames raise `common.ShortVideoError` (mapped to a 422 via
  `UnprocessableMediaError`), matching the original evaluator's behavior rather than inventing
  a fallback — the original's real Celeb-DF-v2 run reported 0 such errors across 6,529 videos.

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
  `video/detector.py` goes through `video/backend.py`'s `VideoModelV1Backend` (EfficientNet-B0,
  see above), bypassing `model_client` entirely for both. MesoNet is present but currently dead
  in the live request path. A further audio model is also currently being trained separately
  and is not yet in this repo.

### Heuristic (real computation, no trained model)
- **Video (fallback only, no longer the default)**: `backend/app/services/heuristics.py::
  analyze_video()` — samples 12 raw byte windows from the file and computes Shannon entropy
  statistics over those bytes. This does **not** decode frames, faces, or any actual video
  signal. Still wired as `HeuristicVideoBackend` in `video/backend.py` and selectable via
  `VIDEO_MODEL_BACKEND=heuristic`, but Video Model v1 (above) is now the default. Kept
  registered rather than deleted per the "don't delete without flagging" rule, in case Video
  Model v1 needs a quick rollback.

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
| `CLIP_MODEL_CHECKPOINT` | `pipelines/image_clip/model.py` | `models/checkpoints/image_clip/clip_head_best.pth` | docker-compose points this at `clip_head_round3_portrait_app.pth` as of 2026-09-16; override with an absolute path when running outside Docker |
| `CLIP_MODEL_DEVICE` | `pipelines/image_clip/model.py` | `auto` (cuda if available, else cpu) | `cpu`/`cuda` |
| `VIDEO_MODEL_BACKEND` | `services/video/backend.py` | `model_v1` | `heuristic` to fall back to the old byte-entropy heuristic |
| `VIDEO_MODEL_V1_CHECKPOINT` | `services/video/model_v1/common.py` | `backend/checkpoints/video/epoch_11_model_only.pt` | absolute path recommended when running outside Docker |
| `VIDEO_MODEL_V1_DEVICE` | `services/video/model_v1/common.py` | `auto` (cuda if available, else cpu) | `cpu`/`cuda` |
| `MODEL_SERVICE_URL` | `services/models/model_client.py` | `http://models:8001` | only relevant if the `models/` microservice is actually deployed as a docker-compose service, which it currently is not |
| `AUTH_SECRET_KEY` | `core/security.py` | insecure dev default | **must** override for any real deployment |
| `DEBUG` | `core/config.py` | `true` | |

No `.env` file currently exists in the repo (confirmed absent as of Phase 1 audit). `Settings`
reads from a `.env` if present (`core/config.py`'s `Config.env_file = ".env"`).

---

## 6. How to run

**Backend** (no Docker required — a `.venv` with all deps already exists at repo root):
```powershell
cd backend
$env:IMAGE_MODEL_CHECKPOINT="../models/checkpoints/image/convnext_tiny_diversified_v2.pth"
$env:CLIP_MODEL_CHECKPOINT="../models/checkpoints/image_clip/clip_head_round3_portrait_app.pth"
$env:IMAGE_MODEL_DEVICE="cpu"; $env:CLIP_MODEL_DEVICE="cpu"
../.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Both `IMAGE_MODEL_CHECKPOINT` and `CLIP_MODEL_CHECKPOINT` must be set when running outside
Docker (absolute paths recommended) — the defaults baked into `pipelines/image/model.py` and
`pipelines/image_clip/model.py` assume the Docker bind mounts at `backend/checkpoints/`, which
don't exist locally.

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
