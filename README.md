# TruthLens

**AI-generated media detection that shows its work.** Upload an image, video or audio clip and get a verdict
(REAL / FAKE / UNCERTAIN), a confidence score, an explanation of *why*, what the file's own metadata says, and a downloadable PDF
forensic report — with the limits of every claim stated plainly.

Final-year project — Dept. of Information Technology, VIT Mumbai, 2025–26.

---

## What it does

| Media | Detector | What it detects |
|---|---|---|
| **Image** | **ConvNeXt-Tiny** (fine-tuned by the team) **+ CLIP ViT-B/16** second opinion (trained head), combined by taking the higher fake score | Real photo vs AI-generated image |
| **Video** | **Video Model v1** — EfficientNet-B0 on 16 face-cropped frames (fine-tuned by the team, verified against the original Celeb-DF-v2 evaluator) | Face-swap deepfakes |
| **Audio** | **AASIST** (pretrained, Jung et al. ICASSP 2022) | Real vs synthetic/cloned speech |

Scores near the decision threshold come back as **UNCERTAIN** rather than a forced call.

### Beyond the verdict
* **Explanations** — Grad-CAM heatmaps for images; per-frame scores and the exact face crops the model saw for video. Grad-CAM explains the
  model's score; it is not proof of manipulation, and the UI says so.
* **Provenance** — C2PA Content Credentials (signature, unchanged-since-signing, trust), embedded metadata and generation parameters. When the
  models say REAL but the file declares itself AI-generated, all three surfaces (UI, API, PDF) flag the conflict. Missing metadata is never
  treated as evidence.
* **Background video scans** with a real progress bar, queue position and cancel — a long video no longer freezes the server.
* **Feedback** — "was this correct?", with an opt-in (off by default) to let a file be kept for evaluation, changeable and withdrawable.
* **PDF report** — verdict, per-model scores, explanation, provenance, caveats.
* **Private by design** — every scan, report, job and feedback item is visible only to its owner (other users get 404, never 403).

## How well does it work? (measured, not marketed)

Evaluated with the harness in [`backend/eval/`](backend/eval/) on 6,500+ images and 300+ videos; details and caveats in
[`backend/eval/results/EVALUATION_SUMMARY.md`](backend/eval/results/EVALUATION_SUMMARY.md).

| | Result |
|---|---|
| Image, on kinds of data it was trained on | **98.4 %** accuracy, AUC 1.00 |
| Image, on generators it **never saw** (GenImage, OpenFake, ChatGPT) | **53 %** accuracy, AUC 0.67 — only ~27 % of AI images caught |
| Video, unseen face swaps (RTFS) | 79 % accuracy, AUC 0.89 |
| Video Model v1 on Celeb-DF-v2 (original protocol, 6,529 videos) | ROC-AUC 0.695 |
| Provenance | catches 9 of the 1,026 AI images the models miss (all 8 ChatGPT ones) — but **only if the file is untouched**: every re-save we tried removed the declaration |
| Robustness (screenshot, WhatsApp-style, WebP) | AUC stays ≥ 0.99 on familiar sources; but faint noise or heavy JPEG makes 30–40 % of *real* photos look fake ([details](backend/eval/results/robustness/report.md)) |

**Bottom line:** near-perfect on data like its training data, weak on generators it hasn't seen. Read
[`docs/ETHICS_AND_LIMITATIONS.md`](docs/ETHICS_AND_LIMITATIONS.md) before relying on a verdict — "REAL" means "no evidence found", and "FAKE" is a lead, not proof.

## Stack
* **Backend:** FastAPI (Python 3.11), async SQLAlchemy over SQLite, JWT auth · **Frontend:** React + Vite, Tailwind
* **ML:** PyTorch (image, video), ONNX Runtime (audio), OpenCLIP · **Reports:** ReportLab · **Provenance:** c2pa-python
* **Deploy:** Docker Compose (dev and production), nginx · **CI:** GitHub Actions (pytest + frontend build)

## Running it locally

### 1. Get the model files
The checkpoints (~335 MB) are not in git. You need three files; their SHA-256 sums are in [`models/CHECKSUMS.sha256`](models/CHECKSUMS.sha256):

| File | Place it at |
|---|---|
| `convnext_tiny_diversified_v2.pth` (~319 MB) | `models/checkpoints/image/` |
| `clip_head_round3_portrait_app.pth` (~0.5 MB) | `models/checkpoints/image_clip/` |
| `epoch_11_model_only.pt` (~16 MB) | `backend/checkpoints/video/` |

If they are hosted somewhere you control: `python backend/scripts/fetch_checkpoints.py --base-url <folder-url>` downloads and verifies them
(`--check` only verifies). The audio weights are small and already in git. Without the image/video files those scans fail with an error; audio still works.

### 2a. With Docker
```bash
docker-compose up --build          # app: http://localhost:3000   API docs: http://localhost:8000/docs
```
### 2b. Without Docker
Backend and frontend commands (with the environment variables for the checkpoints) are in [`CLAUDE.md`](CLAUDE.md) §6.

### 3. Sign up
Each machine has its own local database — create a fresh account. Your history, uploads and reports are private to it.

## Tests and CI
```bash
cd backend && ../.venv/Scripts/python.exe -m pytest              # everything (~1.5 min)
cd backend && python -m pytest -m "not models"                    # what CI runs: no model files needed
```
200+ tests cover auth, access control, upload validation, detection routes, jobs, feedback, PDF content, explain/provenance, verdict thresholds
and the evaluation metrics. `backend/tests/mutation_check.py` deliberately breaks guarded behaviours to prove the tests notice.

## Deploying
See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md): production compose file, safe start-up checks (the server refuses to start without a real
signing key), checkpoint fetching, TLS, backups and a pre-launch checklist.

## Documentation
| | |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System, detector, scan-flow and data-model diagrams |
| [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) | What could go wrong, what protects against it, and the known gaps |
| [`docs/ETHICS_AND_LIMITATIONS.md`](docs/ETHICS_AND_LIMITATIONS.md) | Accuracy limits, blind spots, bias, privacy, responsible use |
| [`docs/DATASETS.md`](docs/DATASETS.md) | Every dataset, its licence and whether the models saw it |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Putting it on a server |
| [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md) | An 8-minute presentation storyboard |
| [`CLAUDE.md`](CLAUDE.md) | Maintained source of truth for internals: what is real ML vs heuristic vs dead code, environment variables, rules for changes |

## Honest limitations (short version)
* Detection generalises poorly to generators it was not trained on (numbers above); consumer face-swap apps such as Akool and Magic Hour are still missed.
* Provenance only helps on untouched files; screenshots, messaging apps and re-saves remove it.
* Audio uses a pretrained model and has not been evaluated by this project; a teammate's separate audio model is not integrated.
* Uploads are stored until you delete them; there is no retention policy or "delete my data" button yet (see the threat model).
* Not a forensic tool for legal decisions: it produces leads and evidence to review, not proof.
