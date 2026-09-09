# TruthLens

**AI-generated media detection.** Upload an image, video, or audio clip and get a verdict
(REAL / FAKE / UNCERTAIN), a confidence score, and a downloadable PDF forensic report.

Final-year project — Dept. of Information Technology, VIT Mumbai, 2025–26.

---

## What it does

TruthLens runs three independent detectors, one per media type:

| Media | Model | Trained on | What it actually detects |
|---|---|---|---|
| **Image** | ConvNeXt-Tiny (fine-tuned by this team) | CIFAKE | Real photograph vs. AI-generated image |
| **Video** | EfficientNet-B0 (fine-tuned by this team) | Celeb-DF-v2 | Face-swap deepfake detection |
| **Audio** | AASIST (pretrained, Jung et al. ICASSP 2022) | ASVspoof2019-LA | Real speech vs. synthetic/cloned (TTS) speech |

Only the video model is a "deepfake detector" in the strict sense (manipulation of a real
person's face). Image and audio are real-vs-synthetic classifiers — the umbrella term
"AI-generated media detection" is used deliberately for the product as a whole.

Every result includes a confidence score, and scores close to the decision threshold are
returned as **UNCERTAIN** rather than a forced call — the app never presents a low-confidence
guess as a certain verdict.

## Stack

- **Backend:** FastAPI (Python), SQLAlchemy (async) over SQLite, JWT auth
- **Frontend:** React + Vite, Tailwind
- **ML:** PyTorch (image, video), ONNX Runtime (audio)
- **Reports:** server-generated PDF (ReportLab)
- **Deploy:** Docker Compose (backend + frontend)

## Running it locally

### 1. Docker Desktop
Install and open it — make sure the engine is running.

### 2. Clone
```bash
git clone https://github.com/rudra2024-sys/TruthLens.git
cd TruthLens
```

### 3. Get the model checkpoints
Two files aren't in this repo (too large for git) — ask a teammate who has them for:

| File | Size | Place it at |
|---|---|---|
| `convnext_tiny_best.pth` | ~319 MB | `models/checkpoints/image/convnext_tiny_best.pth` |
| `epoch_11_model_only.pt` | ~16 MB | `backend/checkpoints/video/epoch_11_model_only.pt` |

(Create the `checkpoints/...` folders if they don't already exist — they're gitignored, so a
fresh clone won't have them.) Audio needs no extra setup — its weights are small enough to
already be committed.

Without the checkpoints, the app still runs, but image/video verification will fail with an
error; audio verification works out of the box.

### 4. Start it
```bash
docker-compose up --build
```
First run takes a few minutes. Once it's up:
- App: **http://localhost:3000**
- API docs: **http://localhost:8000/docs**

### 5. Sign up
Each machine has its own local database — sign up for a fresh account. Your history, uploads,
and reports are private to your account and won't be visible to (or from) anyone else's.

To stop: `Ctrl+C`, then `docker-compose down`. To restart later: `docker-compose up` (no
`--build` needed unless the code changed).

## Honest limitations

- The image and audio models were trained on narrow benchmark datasets (CIFAKE, ASVspoof2019-LA)
  and don't fully generalize to arbitrary real-world uploads yet — a real photo or a real audio
  recording can sometimes be misclassified. This is disclosed here deliberately rather than
  overclaiming accuracy.
- The video model's validated accuracy is ROC-AUC 0.695 on Celeb-DF-v2 (verified line-by-line
  against the original evaluator, 6,529 videos, 0 inference errors) — an honest number, not
  inflated.

## Project docs

`CLAUDE.md` in the repo root is the maintained source of truth for architecture, what's real
vs. dead code, environment variables, and detailed dev setup — read it before making changes.
