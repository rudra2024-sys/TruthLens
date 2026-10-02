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
| `/api/v1/detect/{upload_id}/jobs` | POST | Start detection as a background job (202); idempotent (see section 13) |
| `/api/v1/jobs/{job_id}` | GET / DELETE | Poll a job's state + real progress / cancel it |
| `/api/v1/history` | GET | Recent uploads + their verdicts (dashboard/history page) |
| `/api/v1/stats` | GET | Aggregate counts by verdict/media type (dashboard) |
| `/api/v1/report/{upload_id}` | GET | Generates (or regenerates) and streams the PDF report |
| `/api/v1/auth/signup` | POST | Create account, returns JWT |
| `/api/v1/auth/login` | POST | Returns JWT |
| `/api/v1/auth/me` | GET | Current user (bearer token required) |
| `/api/v1/detect/{upload_id}/explain` | GET | Grad-CAM / per-frame explanation, computed on demand (section 11) |
| `/api/v1/detect/{upload_id}/provenance` | GET | C2PA/EXIF/ELA provenance signals (section 12) |
| `/api/v1/detect/{upload_id}/feedback` | GET / PUT / DELETE | Read / upsert / withdraw "was this correct?" feedback (section 15) |

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
  - **Known video blind spot (confirmed 2026-09-24)**: consumer AI face-swap/video apps (Akool,
    Magic Hour) — 2 of the 10 local ground-truth videos, both vertical 9:16 phone-format clips
    (464x832, 480x848) — score real-leaning on *every* sampled frame (best-frame logits ~+0.01
    to +0.06). Frame pooling is not the cause: max and top-4-mean pooling were tried and did no
    better (pairwise AUC 0.60 vs 0.64 for mean) and flagged all 5 real videos FAKE. Same class
    of problem as the image portrait-app / ChatGPT gaps: no training source (FF++, Celeb-DF,
    DFDC, WildDeepfake) contains these generators or the vertical format. **Partial root cause
    found for one of the two clips, 2026-10-02**: `fake video 4.mp4` (480x848, Magic Hour) finds
    a Haar face in 0 of its 16 sampled frames — the frame content is encoded rotated 90° from
    upright (no rotation metadata in the container to auto-correct; confirmed by hand), so the
    face detector itself never gets a chance to run on a recognizable face, independent of
    whatever the model would have scored it. See section 19 for the opt-in fix and its measured
    (small, band-swallowed) effect. The other clip's cause is still unknown.
  - **CNN+BiGRU temporal head experiments (Colab, not the production default)**: a full-dataset
    head hit val AUC 0.89 in-distribution but 0.44 on the ground-truth videos; v3 (adds
    WildDeepfake, checkpoint selected by real-world AUC) reached 0.72 vs 0.64 for the deployed
    CNN, but on 10 videos with poor calibration — not enough evidence to swap. **v5** (2026-10-01,
    checkpoint `backend/checkpoints/video/cnn_gru_v5_head.pt`, gitignored; adds RTFS-10k, drops
    the full-dataset pranabkc source down to a 30% sampling fraction, selects by mean of
    real-world/WildDeepfake-held-out/RTFS-held-out AUC): indist AUC 0.87, WildDeepfake held-out
    0.79, RTFS held-out 0.98, self-reported real-world AUC 0.76 on the same 10 local videos —
    **but that real-world number is circular**, since checkpoint_selection explicitly used those
    same 10 videos to pick the epoch, and the underlying per-video probabilities are degenerate:
    9 of 10 videos score within ~0.06 of 0.0 regardless of true label (re-verified 2026-10-01 via
    `eval/video_head_experiments/eval_cnn_rnn.py` pointed at this checkpoint, and again through
    the live `CnnGruV5Backend` wiring — both reproduce the exact same numbers), with the AUC
    propped up almost entirely by one fake video scoring 0.99. At the deployed 0.525 threshold
    this backend catches only 1 of 5 real-world fakes. Wired in as an **opt-in, non-default**
    backend (`VIDEO_MODEL_BACKEND=cnn_gru_v5`, see `app/services/video/cnn_gru_v5.py` and
    `CnnGruV5Backend` in `app/services/video/backend.py`) at the user's explicit request, for
    continued evaluation only — `metadata.validated=False` guards against it silently becoming
    the default. Tests: `backend/tests/test_video_cnn_gru_v5.py`. Production default is
    unchanged (still `VideoModelV1Backend`, verified after wiring this in). Root cause is
    presumably the same training-data gap as the deployed CNN's Akool/Magic Hour blind spot
    above — none of v5's three sources (pranabkc cropped-faces, WildDeepfake subset, RTFS-10k)
    resemble real consumer face-swap app footage. Real fix would be sourcing fake clips from
    consumer face-swap apps as a new training source, and re-selecting the checkpoint on a
    real-world set that is NOT also used for selection; needs explicit approval per §8 rule 1.

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
| `VIDEO_MODEL_BACKEND` | `services/video/backend.py` | `model_v1` | `heuristic` for the old byte-entropy heuristic, or `cnn_gru_v5` for the experimental BiGRU head (§3 — opt-in only, known-weak on real-world footage, not for production use) |
| `VIDEO_MODEL_V1_CHECKPOINT` | `services/video/model_v1/common.py` | `backend/checkpoints/video/epoch_11_model_only.pt` | absolute path recommended when running outside Docker |
| `VIDEO_MODEL_V1_DEVICE` | `services/video/model_v1/common.py` | `auto` (cuda if available, else cpu) | `cpu`/`cuda` |
| `VIDEO_MODEL_CNN_GRU_V5_CHECKPOINT` | `services/video/cnn_gru_v5.py` | `backend/checkpoints/video/cnn_gru_v5_head.pt` | only relevant when `VIDEO_MODEL_BACKEND=cnn_gru_v5` |
| `MODEL_SERVICE_URL` | `services/models/model_client.py` | `http://models:8001` | only relevant if the `models/` microservice is actually deployed as a docker-compose service, which it currently is not |
| `AUTH_SECRET_KEY` | `core/security.py` | insecure dev default | **must** override for any real deployment |
| `DETECTION_JOB_CONCURRENCY` | `services/jobs.py` | `1` | max background detection jobs running at once (inference is CPU-bound; the rest wait in `queued`) |
| `DEBUG` | `core/config.py` | `true` | |
| `CORS_ORIGINS` | `core/config.py` / `core/startup.py` | `*` | comma-separated allowed origins; `*` only logs a warning when `DEBUG=false` (section 16) |
| `BIND_ADDRESS` | `docker-compose.prod.yml` | `0.0.0.0` | set to `127.0.0.1` when a TLS-terminating proxy runs on the same host (section 16) |
| `PUBLIC_PORT` | `docker-compose.prod.yml` | `80` | prod compose only |

No `.env` file currently exists in the repo (confirmed absent as of Phase 1 audit). `Settings`
reads from a `.env` if present (`core/config.py`'s `Config.env_file = ".env"`). An `.env.example`
now documents every variable above for production use (section 16).

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

- **Backend: a pytest suite exists** (`backend/tests/`, config in `backend/pytest.ini`, deps in
  `backend/requirements-dev.txt`). From `backend/`: `../.venv/Scripts/python.exe -m pytest` runs everything (~45 s);
  `-m "not models"` is what CI runs (~20 s, no checkpoints needed); `-m models` runs only the tests that use the
  real checkpoints (they skip themselves when a checkpoint is missing). It uses a temp DB/upload/report dir set
  before `app` is imported (the app builds settings + engine at import time), stubs the detectors for the API
  tests, and covers: auth, upload validation, 404-not-403 ownership on every route, error mapping, PDF content +
  graceful degradation, verdict banding (incl. the protected 0.5 / 0.525 thresholds), explain/provenance logic,
  and the eval metrics (vs scikit-learn). Regression tests exist for two real bugs: repeated `POST /detect/{id}`
  used to store a 2nd result and make `/result`, `/report`, `/explain`, `/provenance` 500 forever (now idempotent),
  and stale heuristic labels on Video-v1 PDFs. A mutation check (breaking each guarded behaviour on purpose)
  confirmed the tests fail when they should.
- **CI:** `.github/workflows/ci.yml` — backend job (Python 3.11, `pytest -m "not models"`) and frontend job
  (`npm ci && npm run build`; there is still no frontend test script). It has not been run on GitHub yet (verified
  locally from a clean virtualenv only, which also caught that `requirements.txt` lacked the `greenlet` SQLAlchemy's async mode needs — now `sqlalchemy[asyncio]`). New backend features should come with tests in `backend/tests/`.
- Older standalone scripts (`backend/test_image_detection.py`, `test_*_smoke.py`, `test_video_model_v1_*.py`,
  `test_blind_spot_checkpoints.py`, root `smoke_test.py`/`test_detection.py`, `frontend/smoke_test.py`,
  `models/test_image_pipeline.py`) are unchanged, are not collected by pytest (`testpaths = tests`), and are not in CI.
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

---

## 10. Evaluation harness (`backend/eval/`) and measured results

Read-only measurement of the **deployed** detectors — changes no model, weight or threshold. Three steps
(`build_manifest.py` → `run_predictions.py` → `make_report.py`, see `backend/eval/README.md`); raw scores and
manifests live in git-ignored `backend/eval/data/`, reports in `backend/eval/results/{image,video,video_lab_lowres}/`,
and the written analysis in `backend/eval/results/EVALUATION_SUMMARY.md`. Metrics are verified against
scikit-learn by `python -m eval.test_metrics`. Evaluation datasets were downloaded to `D:\eval_data\` (not in the
repo; OpenFake's license is "unknown" — keep local).

Headline (2026-09-26): **near-perfect on training-like data, near chance on unseen generators.**
- Image (deployed max ensemble): seen sources (CIFAKE/DeepDetect/140k Faces/AI-vs-Human) 98.4% acc, AUC 1.00;
  unseen (GenImage SD/Midjourney/BigGAN, OpenFake, ChatGPT) 53.2% acc, AUC 0.665, only 27.3% of fakes caught.
  Ensemble ablation confirms the max-vs-average trade-off in section 3.
- Video (Video Model v1): unseen RTFS face swaps (inswapper/uniface) + own 10 videos: 79.4% acc, AUC 0.892
  [0.852, 0.926]. Own 10 videos alone are too few (AUC 0.64); Akool/Magic Hour fakes still missed.
- **The `pranabkc/deepfake-with-cropped-faces-from-video` clips are 112x112 px** — the deployed video model's scores
  collapse to ~0.45-0.50 on them (acc 50.8%), so they are not a valid test set for it, and note they are
  low-resolution when used as training data for the Colab CNN+BiGRU head experiments.
- Caveats to repeat in any report: "seen" sources are train/test folders of datasets used in training (the exact
  hold-out split was not recorded); GenImage real=JPEG vs fake=PNG format bias; RTFS real/fake encode differences.

---

## 11. Explainability (`backend/app/services/explain/`) — added 2026-09-26

Read-only, additive: recomputes attributions with the **deployed** models; changes no model, weight, threshold,
stored result or DB schema. Verified to reproduce production numbers exactly (`backend/test_explain_consistency.py`).

- **Image:** Grad-CAM on ConvNeXt-Tiny's last stage w.r.t. the FAKE logit. **The CLIP sub-model has no heatmap** — when
  CLIP drives the max-ensemble verdict, the API/PDF say so explicitly rather than implying ConvNeXt's map explains it.
  Heatmap strength is scaled by ConvNeXt's own FAKE probability (`core.evidence_strength`) so images judged REAL show
  no fabricated "evidence" (per-image normalised Grad-CAM would otherwise light up noise).
- **Video (Video Model v1 only):** per-frame FAKE logits/probabilities (the verdict is sigmoid of their mean), the exact
  224px crops the model saw, and Grad-CAM on the 3 most suspicious frames. Unavailable under
  `VIDEO_MODEL_BACKEND=heuristic`. Audio: explicitly "not available" (teammate's area).
- **Surfaces:** `GET /api/v1/detect/{upload_id}/explain` (auth + owner check, computed on demand in a threadpool,
  returns data-URI images), a best-effort "Explainability" section in the PDF (failure only skips that section), and
  `frontend/src/components/ExplanationPanel.jsx` (lazy — fetched only when opened) used in `ReportDetail.jsx` and
  `UploadFlow.jsx`. All three verified end to end in a real browser.
- **Faithfulness check** (`python -m eval.explain_faithfulness`, results in `backend/eval/results/explain_faithfulness.md`):
  masking Grad-CAM's top 20% region lowers ConvNeXt's FAKE probability by 0.232 on average vs 0.044 for a random region
  of equal size; Grad-CAM wins on 85% of 195 images. Small drops on saturated (99%+) seen sources are expected.
- **Also fixed (report labels):** with Video Model v1 the PDF's video breakdown used stale heuristic labels
  ("Structure Entropy Score", "Byte Windows Sampled"); it now says "Video Model v1 FAKE Probability" / "Frames Analyzed".
  The audio labelling issue in section 3 is untouched (teammate owns audio).
- **Useful finding:** the video crops expose face-detector failures — for the Akool clip (`fake video 3`) the Haar
  detector locks onto a wall poster/background instead of a face, which plausibly explains that miss.
- Grad-CAM shows what the model's score depends on, not proof of manipulation (the API `note` and PDF say so).

---

## 12. Provenance checks (`backend/app/services/provenance/`) — added 2026-09-26

Supplementary, read-only evidence shown next to the verdict. **It never changes a stored result, threshold or model
score** (fusing provenance into the verdict would be a behaviour change needing explicit approval, section 8 rule 1).

- **C2PA Content Credentials** via the official `c2pa-python` SDK (in `requirements.txt`), validated against the bundled
  C2PA trust list (`provenance/trust/c2pa_trust_list.pem`, retrieved 2026-09-26 from c2pa-org/conformance-public —
  refresh periodically; README alongside). Reported facts: signature valid, **file unchanged since signing**
  (a flipped byte gives `assertion.dataHash.mismatch`), signer on trust list, AI declared (IPTC digital-source-type),
  issuer/generator/actions. OpenAI's signer is currently **not** on the trust list, so its credentials show
  "signature valid, unchanged, signer identity self-asserted" — worded that way on purpose (untrusted != forged).
- **Embedded metadata** (Pillow): camera EXIF (weak signal only), editing software, GPS *presence* (coordinates are never
  returned), AI generation parameters (PNG `prompt`/`seed`/`parameters`/ComfyUI workflow, A1111-style
  Steps/Sampler/CFG layout), XMP AI source type. **ELA** for JPEG only, as a labelled visual aid, never a score.
  Video: C2PA + container facts + encoder tag.
- **Fixed SD/SDXL invisible watermark** (`provenance/watermark.py`, added 2026-09-27; extended to sampled video
  frames 2026-10-02, see section 18): diffusers embeds a
  fixed 48-bit message in every Stable Diffusion / SDXL image by default (`ShieldMnt/invisible-watermark`'s "dwtDct"
  method - a Haar-DWT coefficient trick, not an actual DCT despite the name). Rather than depending on that package
  (its `opencv-python` dependency would collide with this repo's `opencv-python-headless` and risks breaking the
  Docker build), the decode-only path is reimplemented locally with `PyWavelets` (now in `requirements.txt`) plus the
  `cv2`/`numpy` already present. **Verified bit-for-bit identical to the real library** in a throwaway venv before
  being wired in (exact match on both a real-encoder-produced image and on 4 sizes of random/unwatermarked images,
  0 false positives across 30 random-image trials) - the one subtlety that took two passes to match exactly: the
  library's bit threshold is `avg*255 > 127` (~0.498), not `avg > 0.5`, which only shows up on tie-vote blocks
  (i.e. only visible on unwatermarked/near-random content, never on a real watermark's strongly-biased blocks).
  `present` requires an exact 48/48 match (negligible chance of a coincidental hit on an unrelated image); the raw
  `bit_match` fraction is also always reported for transparency. Like everything else here, a **miss proves nothing**:
  many front-ends (AUTOMATIC1111, ComfyUI, most hosted APIs) disable the watermarker, and it does not survive
  resizing, cropping or noticeable recompression (confirmed: gone after a plain resize or JPEG q90 re-save in
  testing) - and unlike a C2PA signature, the fixed pattern is public, so it is theoretically forgeable onto a real
  photo, worded that way in the signal's own detail text. Feeds `_assess()` exactly like a metadata AI declaration
  (same `declared_ai` level / conflict-note path) since a match is comparably strong direct evidence. Not yet run
  through `eval/provenance_study.py`'s 6,532-image harness (see below) - no measured real-world hit-rate yet, unlike
  C2PA/ELA.
- **Wording rules (rule 5):** absent metadata "says nothing either way"; present metadata "can be edited or forged".
  A stripped/re-saved file simply shows nothing — provenance can be removed, so it can add evidence but never clear a file.
- **Conflict note:** when the models say REAL/UNCERTAIN but the file declares itself AI-generated (via metadata or the
  watermark), API + UI + PDF say so explicitly (the ChatGPT "everyday" blind-spot images: models 0/8, all 8 carry
  OpenAI C2PA credentials).
- **Surfaces:** `GET /api/v1/detect/{upload_id}/provenance` (auth + owner check), "Provenance & Metadata" PDF section
  (best-effort - the watermark signal renders through the same generic signal loop as C2PA/metadata, no separate PDF
  code needed), `frontend/src/components/ProvenancePanel.jsx` (auto-fetched, silent on failure) in `ReportDetail.jsx`
  and `UploadFlow.jsx`. Verified end to end (build_provenance -> signals -> PDF-equivalent output) for a real
  watermarked test image, including the REAL-verdict conflict note. Tests: `backend/tests/test_provenance.py`
  (17 checks incl. tamper/strip/EXIF/garbage) and `backend/tests/test_watermark.py` (7 checks incl. exact-match
  detection, 0 false positives on random images, JPEG-recompression fragility, bad/missing files, and that a
  watermark match feeds the same assessment path as a metadata declaration), study: `python -m eval.provenance_study`
  → `backend/eval/results/provenance_study.md` (C2PA/EXIF/ELA only so far, not yet extended to the watermark check).
- **Measured limits (6,532 eval images, C2PA/EXIF/ELA only):** AI declarations exist on only 0.4% of AI images overall
  — benchmark datasets are re-encoded and stripped — but on 9 of the 1,026 AI images the models miss, and all 8
  ChatGPT + 1 portrait-app blind-spot images. So it helps on fresh, unmodified downloads from generators that embed
  credentials, not on scraped/re-uploaded data. ELA separates real/fake with AUC 0.53-0.75 on most sources but
  **0.995 on DeepDetect**: a JPEG-compression-history shortcut in that dataset, which the ML models may be exploiting
  too (consistent with, not proof of, their collapse on unseen generators).


---

## 13. Background detection jobs (`backend/app/services/jobs.py`) — added 2026-09-27

Video scans used to run inside the HTTP request: no progress feedback, and the blocking model call froze the server's
event loop for every other user. The frontend now runs **video** scans as jobs (image/audio stay synchronous — they are fast).

- **API:** `POST /api/v1/detect/{upload_id}/jobs` -> 202 job; `GET /api/v1/jobs/{job_id}` -> `{state: queued|running|done|
  failed|cancelled, progress 0..1, stage, queue_position, result_id, error}`; `DELETE` cancels. When `done`, fetch the
  result from the existing `GET /detect/{upload_id}/result`. Owner-only (404 for anyone else), idempotent (an already-
  scanned upload returns a finished job; an upload with a scan in flight returns that job - no duplicate results).
- **Real progress:** `model_v1.optimized.predict(..., progress=cb)` and `common.decode_selected_frames(..., on_frame=cb)`
  gained optional callbacks (decode 0-70 %, face-crop 70-90 %, scoring 90 %). They are observation-only - verified
  bit-identical predictions with/without a callback (`tests/test_models.py`) and `test_video_model_v1_regression.py` still
  passes. Cancel works by raising inside the callback. The UI shows the server's numbers, not a timer.
- `run_video_detection` now runs the scoring in a worker thread (`run_in_threadpool`), so it no longer blocks the event loop
  - this also applies to the old synchronous `POST /detect/{id}` path. (Image/audio detectors are still blocking calls.)
- **Limits:** job state is in memory of ONE server process (results are durable in the DB; an unknown job id after a restart
  404s and the UI falls back to `/result`). Running several uvicorn workers would need a shared job store. Concurrency is 1
  by default (`DETECTION_JOB_CONCURRENCY`). If the user leaves the page the scan keeps running and lands in the history.
- **Tests:** `backend/tests/test_jobs.py` (13 tests incl. non-blocking, cancel-stops-early, queue position, ownership);
  the mutation tool `backend/tests/mutation_check.py` (17 guarded behaviours as of section 17 below) catches all of them.
- **Known pre-existing quirk (not changed):** for video, `confidence_score` is the raw FAKE probability (Video Model v1's
  protocol), so the UI's confidence gauge shows e.g. "31 %" next to an AUTHENTIC verdict; image and audio use max(p, 1-p).

---

## 14. Robustness suite (`backend/eval/{perturbations,build_robustness_subset,run_robustness,make_robustness_report}.py`)
— added 2026-09-27

Read-only, additive: re-scores the **deployed** detectors on degraded copies of existing eval images. No model, weight
or threshold changed. 300 label-balanced images (180 seen-source, 120 unseen) x 21 settings (clean, JPEG q90-10,
downscale-and-restore, blur, noise, centre crop, screenshot/WhatsApp-style/WebP) — 6,300 scored rows, 0 errors. Full
tables: `backend/eval/results/robustness/report.md`; summarized in `EVALUATION_SUMMARY.md` and
`docs/ETHICS_AND_LIMITATIONS.md` §3.

- **Everyday sharing is fine on seen sources**: screenshot / WhatsApp-style (resize+JPEG q65) / WebP all keep AUC ≥ 0.99.
- **CLIP causes false alarms under noise/compression**: real-photo specificity of the deployed `max` rule falls to
  71.1% (noise std 5), 62.2% (JPEG q10), and even 87.8% at a mild JPEG q90 — ConvNeXt alone stays 96.7-98.9% in all 21
  settings; CLIP alone drops to 63% at q10. This is why `max` inherits CLIP's false-alarm behaviour under degradation.
- **ConvNeXt causes missed fakes under heavy blur**: at blur sigma 3, ConvNeXt alone only catches 50% of AI images (the
  `max` rule recovers to 83% because CLIP still catches its share).
- **Unseen sources stay near chance** (AUC ~0.61) regardless of degradation; degradations mostly shift scores toward
  FAKE without adding real discrimination.
- **No model changed here** — the obvious next step (noise/compression augmentation on retrain) needs explicit
  approval per section 8 rule 1.
- Tests: `backend/tests/test_perturbations.py` (14), `test_robustness_report.py` (3).

---

## 15. Feedback loop (`backend/app/api/routes/feedback.py`, `app/services/feedback_export.py`) — added 2026-09-27

"Was this result correct?" on the result screen and the report-detail page. Purely additive: writes to a new
`feedback` table, never touches a stored detection result, model, or threshold.

- **API:** `GET /api/v1/detect/{upload_id}/feedback` -> the current user's feedback or `null`. `PUT` (same path) upserts
  it — one feedback row per (upload, user), resubmitting replaces it. Body: `agrees: bool`, optional `true_label`
  (`real`/`ai`/`unsure`, ignored when `agrees` is true), optional `comment` (<=500 chars), `allow_reuse: bool` (opt-in,
  default false). Disagreeing without a label is stored as `"unsure"`. `DELETE` withdraws it (idempotent, 204 either
  way) — this also revokes any reuse consent. Owner-only everywhere (404 for anyone else, matching the rest of the API).
  Needs an existing detection result on the upload, else 404.
- **Consent is per-item and explicit**: only feedback with `allow_reuse=true` ever has its file copied or its comment
  exported; everything else contributes anonymous counts only (`app/services/feedback_export.py::derived_label` /
  `export_feedback`). A label is derived only when unambiguous (agree+FAKE->ai, agree+REAL->real, disagree+explicit
  label->that label; agreed UNCERTAIN and "unsure" answers carry no label).
- **Export tool:** `backend/scripts/export_feedback.py --out <dir>` writes `feedback_summary.csv` (all feedback, no
  file paths, comments blanked unless consented), `manifest_feedback.csv` (same columns as
  `eval/build_manifest.py`, consented+labeled rows only, so `python -m eval.run_predictions` can score the deployed
  models on cases users flagged), and copies the consented files into `<dir>/files/`. **For evaluation only** — using
  this to retrain needs explicit approval (section 8 rule 1).
- **Frontend:** `frontend/src/components/FeedbackPanel.jsx` (states: unanswered Yes/No -> "wrong" detail form with
  label buttons + optional comment + consent checkbox (default off) -> saved state with a re-togglable consent
  checkbox, "Change" (reopens the edit form) and "Withdraw feedback"), wired into `UploadFlow.jsx` and
  `ReportDetail.jsx`. Verified end to end in a real browser against the live API: save (disagree/real/comment/
  consent), consent toggle re-PUTs `allow_reuse` immediately, "Change" reopens the edit form, "Cancel" reverts without
  saving, "Withdraw feedback" deletes server-side (`GET .../feedback` returns `null` after), no console errors.
- **Tests:** `backend/tests/test_feedback.py` (17: agree/disagree/unsure, idempotent resubmit, ownership, consent-gated
  export, label derivation, confusion counts).

---

## 16. Production deployment (`backend/Dockerfile.prod`, `docker-compose.prod.yml`, `frontend/nginx.prod.conf`,
`backend/app/core/startup.py`, `backend/scripts/fetch_checkpoints.py`) — added 2026-09-27

Full walkthrough in `docs/DEPLOYMENT.md`. **Statically validated only** — `docker compose config` resolves the prod
compose file against `.env.example`, and every helper (startup guard, checkpoint fetch/verify, file contents) has a
passing test (`backend/tests/test_deploy_helpers.py`, 22 tests), but the images have never actually been built or run
(no Docker engine in this environment) — do the smoke test in `docs/DEPLOYMENT.md` §5 before trusting it in the field.

- **Start-up safety** (`core/startup.py`): with `DEBUG` off, the server refuses to start if `AUTH_SECRET_KEY` is the
  public dev default or under 32 chars; an open `CORS_ORIGINS=*` only logs a warning. Wired into `main.py`'s lifespan.
  `CORS_ORIGINS` (comma-separated origins, default `*`) is a new env var (`core/config.py`), parsed by
  `startup.parse_origins`.
  `BIND_ADDRESS` (default `0.0.0.0`) and `PUBLIC_PORT` (default `80`) are new env vars for the prod compose file only,
  letting the site listen on `127.0.0.1` behind a TLS-terminating reverse proxy.
- **Checkpoint fetch/verify** (`backend/scripts/fetch_checkpoints.py`): downloads the three large checkpoints from a
  URL you host and verifies each against `models/CHECKSUMS.sha256`; a mismatched download is discarded, never left
  in place. `--check` verifies without downloading.
- **Hardened images:** `backend/Dockerfile.prod` runs one non-root uvicorn process without `--reload` (the existing
  dev `backend/Dockerfile` still uses `--reload`, unchanged, for local Docker dev). `frontend/Dockerfile` gained a
  `NGINX_CONF` build arg so the same image can build with either `nginx.conf` (dev) or `nginx.prod.conf` (adds
  security headers and a 10 req/min/IP rate limit on `/api/v1/auth/`).
- **Compose:** `docker-compose.prod.yml` sets `DEBUG=false`, mounts checkpoints read-only, keeps state on a
  `truthlens_data:/data` volume, reads secrets from `.env` (see `.env.example`).
- **Feedback export op:** `docker compose ... exec backend python scripts/export_feedback.py --out /data/feedback_export`
  (see section 15).
- Backend process memory: see the sizing note in `docs/DEPLOYMENT.md` §1 (~265 MB resident after all three image/video
  models are loaded and warmed up; up to ~2.3 GB committed virtual memory from PyTorch's CPU allocator).

---

## 17. Mutation testing coverage (`backend/tests/mutation_check.py`) — updated 2026-09-27

17 guarded behaviours as of this update (run manually: `python tests/mutation_check.py` from `backend/`; each
mutation is applied, the named test is confirmed to fail, then the file is restored byte-for-byte in a `finally`
block). Covers: idempotent detect, provenance ownership, video-v1 report labels, the 0.5/0.525 thresholds,
explainability failure isolation, C2PA tamper detection, the provenance conflict note, Grad-CAM evidence fading, AUC
tie-handling, the non-blocking video job path, job cancellation, monotonic job progress, duplicate-job prevention,
job ownership, **feedback export consent** (added 2026-09-27, section 15), **the production start-up secret guard**
(added 2026-09-27, section 16), and **the Colab v5 cache-fix leakage counter** (added 2026-09-27, see below) — 17/17
caught as of the last run.

### Colab video-head cache fix (`backend/eval/video_head_experiments/v5_cache_fix.py`)

Not part of the live product — a fix for the teammate-adjacent CNN+BiGRU video-head training notebook
(`TruthLens_Video_CNN_RNN_Training_v5.ipynb`, in `C:\Users\admin\Downloads\`), needed because Colab T4 quota exhausts
mid-run and the original `build_feature_cache` returned *every* cached entry regardless of what was actually
requested for the current split — a leakage risk if a later run's train/val split ever differed from an earlier one
sharing the same Drive cache folder.

- **Fix:** `build_feature_cache(items, cache_path, log_every=100)` now returns only the requested items, in requested
  order, reusing whatever is already cached across every `train_features_v5*.pt` / `val_features_v5*.pt` file in the
  shared Drive folder (keyed by `path#view`), and only extracts what's missing.
- **Leakage/consistency check:** `cache_report(train_items, val_items)` prints how many cached entries match this
  run's train list, val list, or neither, and explicitly counts any train-cache items that are this run's validation
  items (`"the old code would have trained on these"`).
- **Split fingerprint:** `split_fingerprint()` prints `zlib.crc32` hashes of the sorted train/val identity lists, so
  the same split can be confirmed across different Google accounts before trusting shared cache reuse. Confirmed
  identical (5,666 train / 999 val identities, fingerprint `1546901497 1575160195`) across the accounts used this
  session — no leakage, cache reuse is safe.
- Tests: `backend/tests/test_v5_cache_fix.py` (11, incl. the leakage-detection counter above).

---

## 18. Video watermark + audio explainability/provenance — added 2026-10-02

Three additive extensions of sections 11/12, same read-only/supplementary rules as everything else in those
sections (no model, weight, threshold or stored result is touched). Covers the gaps those sections previously
called out explicitly: the SD/SDXL watermark check was images-only, and audio had no explainability or
provenance at all.

- **SD/SDXL watermark, extended to video** (`provenance/watermark.py::detect_sd_watermark_video`): samples up
  to 8 evenly-spaced frames from the video and runs the exact same per-frame decode used for images on each one.
  `present` is True if *any* sampled frame matches exactly (re-encoding can destroy the watermark on some frames
  and not others), and the result reports `frames_checked` / `frames_matched` / `best_bit_match` alongside the
  single best-matching frame's decoded hex, same shape as the image case plus these three fields. Wired into
  `provenance/service.py`'s video branch exactly like the image case (`_video_watermark_signals`, same "ai"/
  "strong" signal kind so it feeds `_assess()` and the REAL-verdict conflict note identically). PDF/API/frontend
  (`ProvenancePanel.jsx`) render through the existing generic watermark fields, extended to show the frame counts
  when present. Tests: `backend/tests/test_watermark.py` (added 5, using an FFV1-lossless-encoded AVI so the
  watermark's bits survive the round trip exactly — confirmed byte-identical before being used in tests; MJPG/
  mp4v/etc. all re-encode lossily and would destroy it, same fragility documented in section 12).
- **Audio explainability** (`backend/app/services/explain/audio_explainer.py`): Audio Model v1 (wav2vec2-base)
  has no single late 2D conv feature map the way ConvNeXt/EfficientNet do, so Grad-CAM's own technique doesn't
  apply here — the substitute is a plain **input-gradient saliency curve over time** (Simonyan et al. 2013):
  gradient of the deepfake probability w.r.t. the raw waveform, pooled into 80 time bins, rendered as a
  colour overlay on a log-magnitude spectrogram (computed with `numpy.fft` only, no new dependency) of the single
  most suspicious scored window. Also surfaces each window's own deepfake probability + its position in the
  clip (the verdict is their mean, same convention as video's per-frame mean). Reuses the exact cached production
  model (`audio/model_v1/optimized._get_model_and_device`) and the exact windowing (`optimized._build_windows`),
  so the numbers shown match the verdict exactly (regression-tested, see below) — same "pin explanation to
  production" principle as sections 11's image/video explainers. Fails fast with `available=False` (no model
  load attempted) when `AUDIO_MODEL_BACKEND` isn't `model_v1`, or when the Audio Model v1 checkpoint file isn't
  present — the latter matters because building the model otherwise downloads the wav2vec2-base backbone from
  the Hugging Face Hub *before* checking our own checkpoint, which would silently add a network dependency to
  the CI test suite (which has no checkpoints at all) if not guarded. Surfaces: `GET /api/v1/detect/{id}/explain`
  (extended `ExplanationOut` with `windows`/`windows_above_threshold`/`saliency_window_order`/`spectrogram`/
  `saliency`), the PDF's existing "Explainability" section (window table + spectrogram/saliency image pair),
  `frontend/src/components/ExplanationPanel.jsx`'s new `AudioExplanation` component (same per-item bar-chart +
  heatmap-legend pattern as the video one). Verified end to end over real HTTP (upload → detect → explain →
  PDF) against the real checkpoint, not just unit-tested. Tests: `backend/tests/test_models.py` (3, `-m models`,
  incl. exact-match regression against `optimized.predict`'s own numbers and a real-PDF-render check) +
  `backend/tests/test_detect_api.py` (1, route serialization with a stubbed explainer).
- **Audio provenance** (`provenance/service.py`): audio now gets a provenance check instead of an unconditional
  "not available for this media type" — C2PA Content Credentials only (the C2PA SDK/reader was already
  media-agnostic; audio just wasn't routed to it). No EXIF/ELA/pixel-watermark equivalent exists for audio in
  this codebase, so metadata/ela/watermark are always `None` for audio results. Adds lightweight container facts
  (duration/sample rate/channels/codec) via a container-only probe with PyAV (already a dependency for audio
  decoding) — no full decode, unlike the real detector/explainer. Same `_assess()` / conflict-note path as
  image/video, so an AI-declaring C2PA manifest on an audio file would flag identically. Tests:
  `backend/tests/test_provenance.py` (added 3) + `backend/tests/test_detect_api.py` (updated the test that used
  to assert audio provenance was unavailable) + `backend/tests/test_report.py` (added 1).
- **Full suite**: 240 passed (`pytest` with no marker filter, checkpoints present locally) / 225 passed
  (`-m "not models"`), both re-run after these changes with no regressions.

---

## 19. Video orientation-fallback backend (opt-in, not default) — added 2026-10-02

`app/services/video/model_v1/{orientation,rotation_aware}.py`, registered in `backend.py` as
`VideoModelV1RotationAwareBackend` (`VIDEO_MODEL_BACKEND=model_v1_rotation_aware`). Same "diagnose a specific
miss, fix conservatively, measure honestly, don't flip the default without real evidence" pattern as the CNN+
BiGRU v5 backend (sections 3/17/18) — this is a preprocessing change (same checkpoint, same architecture, same
crop math), not a model change, so section 8 rule 1 doesn't block it, but it still isn't the default until
measured on more than 10 videos.

- **Diagnosis**: `fake video 4.mp4` (480x848, one of the two Akool/Magic Hour blind-spot clips in section 3)
  finds a Haar face in 0 of its 16 sampled frames. Visually confirmed by dumping the decoded frames: the
  content is encoded rotated 90° clockwise from upright (people lying sideways, the app's own watermark text
  sideways too) — rotating each sampled frame 90° clockwise before Haar detection finds a face in 16/16 frames
  instead of 0/16, confirmed with a saved boxed-frame image. `cv2`'s `CAP_PROP_ORIENTATION_META` reads 0 and
  PyAV finds no `DISPLAYMATRIX` side-data on this file, so there is no rotation flag to read and apply
  automatically — the pixels are just stored sideways, apparently an export quirk of this app. This is a
  distinct failure mode from the other documented video miss (`fake video 3.mp4`'s Haar cascade locking onto a
  wall poster, section 11) — that one finds a (wrong) face and is deliberately NOT touched by this fallback.
- **Fix** (`orientation.py::detect_clip_rotation`): tries 0/90/180/270° once per clip (not per frame — a video
  doesn't change orientation mid-clip), and returns 0 immediately the moment the native orientation finds a
  face in *any* of the 16 sampled frames. This means the fallback can only ever activate on a clip where every
  single sampled frame fails to find a face at 0° — exactly the diagnosed failure mode, and the only case where
  changing the crop carries no regression risk against today's production behaviour. Among 90/180/270 it picks
  whichever finds a face in the most frames. `rotation_aware.py::predict()` mirrors `optimized.predict()`
  exactly (same cached model, same `common.preprocess_frame`, same pooling/sigmoid/threshold) with the one
  difference that every frame is rotated by the detected degrees first.
- **Measured (10 local ground-truth videos, 2026-10-02)**: only `fake video 4.mp4` is touched at all (rotation
  = 90°); all other 9 videos are bit-identical to `VideoModelV1Backend` (confirmed via
  `pytest.approx(..., abs=1e-6)` in tests, not just eyeballed). That one video's raw probability moves from
  0.4711 (wrong side of 0.5) to 0.5087 (right side of 0.5) — a real, correct-direction move — but the deployed
  `_band_verdict`'s ±0.2 margin around the 0.525 threshold means **all 10 of these local videos land in
  UNCERTAIN either way**, so the practical verdict on this one clip is unchanged in this specific small sample;
  the fix is directionally real but not yet shown to flip an actual verdict. Honest framing, not oversold: this
  needed the broader `backend/eval/` harness (RTFS / other sources with more vertical/rotated clips) to show a
  verdict-level effect, which has not been run yet.
- **Not investigated**: the other Akool blind-spot clip (`real video`/`fake video` pair at 464x832) is
  untouched by this fix — its own `detect_clip_rotation` returns 0 (native orientation already finds *some*
  face), so whatever its problem is, it isn't "no face found." Root cause still open.
- Tests: `backend/tests/test_video_rotation_fallback.py` (12 — orientation-detection unit tests using a real
  face image rotated by hand, no checkpoint needed; backend-registry wiring tests; and checkpoint-gated tests,
  `-m models`-equivalent via file-existence skip matching `test_video_cnn_gru_v5.py`'s pattern, confirming
  bit-identical output on unaffected clips and the measured move on the affected one).
