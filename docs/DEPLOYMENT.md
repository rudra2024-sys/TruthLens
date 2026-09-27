# Deploying TruthLens

A step-by-step guide to putting TruthLens on a server, plus an honest note on what has and has not been verified.

> **Verification status.** The compose file validates (`docker compose config`), the start-up safety checks, the checkpoint
> download/verification script and the file contents are covered by tests (`backend/tests/test_deploy_helpers.py`). **The production
> images have not been built or run** — Docker's engine was not available where this was written — and the nginx config has only
> been checked as text. Do the "smoke test" in §5 on your own machine before pointing a domain at it.

## 1. What you are deploying

* **Two containers:** `backend` (FastAPI, one process) and `frontend` (nginx serving the built site and proxying `/api/`).
* **One backend process on purpose** — background-job progress is held in that process's memory. Do not scale it to several workers/replicas
  without first moving job state to a shared store.
* **State on a volume (`/data`):** the SQLite database, uploaded files and PDF reports. Back this up (§6).
* **Models:** the checkpoints (~335 MB) are not in git. You host them yourself and the server fetches and verifies them (§3).
* **CPU only.** No GPU is needed or configured (`*_DEVICE=cpu`).

### Sizing
Measured on the development machine (CPU only, after the models were loaded and one image and one video had been scanned):

| | Measured |
|---|---|
| Memory (backend process) | ~265 MB resident (Working Set) after ConvNeXt-Tiny, CLIP ViT-B/16 and Video Model v1 were all loaded and had scanned one image and one video each; Windows reported ~2.3 GB of *committed* private bytes for the same process (PyTorch's CPU allocator reserves more than it keeps resident) — plan for at least 2 GB free to be safe, but everyday RAM use is closer to the 265 MB figure |
| Image scan (warm) | about 0.5 s; the first request also loads the models |
| Video scan | a few seconds for short clips; long/high-resolution files take longer (progress bar shown) |

## 2. Prerequisites
* A Linux server (2+ vCPU and 4+ GB RAM is a sensible starting point — measure with your own traffic), Docker + the Compose plugin.
* A domain name pointing at the server, and ports 80/443 open.
* Somewhere to host three model files (a GitHub Release, an object-storage bucket, or your own server).

## 3. Get the models onto the server
1. Upload these three files to a folder you control, keeping their names (their SHA-256 sums are in `models/CHECKSUMS.sha256`):
   `convnext_tiny_diversified_v2.pth`, `clip_head_round3_portrait_app.pth`, `epoch_11_model_only.pt`.
2. On the server, from the repo root:
   ```bash
   python3 backend/scripts/fetch_checkpoints.py --base-url https://YOUR-HOST/truthlens-models/
   python3 backend/scripts/fetch_checkpoints.py --check      # prints OK for each file
   ```
   A file whose checksum does not match is **discarded**, never left in place. The audio model (AASIST, ~1.6 MB) is already in git.

## 4. Configure and start
```bash
cp .env.example .env
python3 -c "import secrets; print(secrets.token_urlsafe(48))"    # paste the output as AUTH_SECRET_KEY in .env
# in .env also set:  CORS_ORIGINS=https://your-domain   and, if a proxy on the same machine terminates TLS:  BIND_ADDRESS=127.0.0.1  PUBLIC_PORT=8080
docker compose -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.prod.yml ps
```
* With `DEBUG` off (the production compose sets it), the backend **refuses to start** if `AUTH_SECRET_KEY` is missing, the public dev
  default, or shorter than 32 characters. If it exits immediately, read `docker compose -f docker-compose.prod.yml logs backend`.
* An open `CORS_ORIGINS=*` only logs a warning; set it to your site's origin.

### TLS with Caddy (simplest)
Install Caddy on the host and give it this `Caddyfile` (it obtains and renews the certificate automatically):
```
your-domain.example {
    reverse_proxy 127.0.0.1:8080
    request_body { max_size 100MB }
}
```
Keep the compose site on `BIND_ADDRESS=127.0.0.1` so the plain-HTTP port is not reachable from outside. The bundled nginx already rate-limits
`/api/v1/auth/` (10 requests/minute/IP); when a proxy sits in front, make nginx see the real client address (the `real_ip` module) or all
users share one bucket.

## 5. Smoke test (do this before announcing anything)
1. `curl https://your-domain.example/health` → `{"status":"ok",...}`
2. Open the site, sign up, scan a small image → a verdict appears (the first scan is slower: models load once).
3. Scan a video → a **progress bar** appears; try **Cancel analysis**.
4. Download the PDF report and open the case file's **Provenance**, **Explanation** and **Was this correct?** panels.
5. Restart the stack (`docker compose -f docker-compose.prod.yml restart`) and confirm your history is still there (the volume works).
6. Try 15 wrong logins quickly → you should start receiving HTTP 429 from the proxy.

## 6. Operating it
| Task | How |
|---|---|
| Logs | `docker compose -f docker-compose.prod.yml logs -f backend` |
| Update | `git pull && docker compose -f docker-compose.prod.yml up -d --build` (state is on the volume) |
| Back up | Stop writes or use SQLite's online copy, then archive the volume: `docker compose -f docker-compose.prod.yml exec backend python -c "import sqlite3; s=sqlite3.connect('/data/truthlens.db'); d=sqlite3.connect('/data/backup.db'); s.backup(d)"` and copy `/data/backup.db`, `/data/uploads` and `/data/reports` off the machine |
| Restore | Put the files back into the volume and restart |
| Health | `GET /health` (used by the container health checks) |
| Rotate the signing key | Change `AUTH_SECRET_KEY` and restart — every user is logged out (tokens can't be revoked individually) |
| Export user feedback for evaluation | `docker compose ... exec backend python scripts/export_feedback.py --out /data/feedback_export` (only consenting users' files are copied) |

## 7. Other places to run it
* **Render / Railway / Fly.io:** create two services from `backend/Dockerfile.prod` and `frontend/Dockerfile` (build arg `NGINX_CONF=nginx.prod.conf`),
  attach a **persistent disk** to the backend mounted at `/data` (SQLite and uploads need it), and set the same environment variables. Keep it
  at one instance. The frontend's nginx proxies to a host named `backend`, so either keep both containers on one private network under that
  name or edit `proxy_pass` in `nginx.prod.conf`.
* **Hugging Face Spaces:** a Docker Space exposes one container/port, so it would need a single combined image (nginx + backend in one).
  That image is **not provided and not tested**; treat Spaces as future work rather than a supported target.

## 8. Before you make it public — checklist
- [ ] `AUTH_SECRET_KEY` set (the server enforces this) and `.env` **not** committed
- [ ] `CORS_ORIGINS` = your site only; HTTPS on; `BIND_ADDRESS=127.0.0.1` behind the proxy
- [ ] Backups scheduled and one **restore** tried
- [ ] You have read [`THREAT_MODEL.md`](THREAT_MODEL.md) §4 gaps — especially **no data-retention/delete-my-data** and **no in-app login rate limit**
- [ ] A privacy notice tells users their uploads are stored, and that the "keep this file" feedback box is opt-in
- [ ] The limitations in [`ETHICS_AND_LIMITATIONS.md`](ETHICS_AND_LIMITATIONS.md) are visible to users (verdicts are leads, not proof)
