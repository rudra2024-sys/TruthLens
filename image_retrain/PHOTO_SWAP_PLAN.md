# Photo face-swap detector - training run plan (prepared 2026-10-16, NOT started)

Why: the whole-image models are near chance on face swaps, and the borrowed video model (round 4) reaches only ~66-70%
recall at ~5% false alarms on single photos (CLAUDE.md section 28). Better face detection (done: Haar-then-MTCNN) cannot
fix that; the model itself must get better on STILL photos of the kind users upload.

## Baselines to beat (all measured 2026-10-16, production code path, Haar-then-MTCNN)
| Test | Now |
|---|---|
| 356 held-out video test videos x 4 frames (JPEG q95): AUC / fakes caught at logit>=0.74 / reals wrongly flagged | 0.93-0.95 / 68.5% / 5.4% |
| 80 consumer-style swaps (inswapper_128) + their 80 untouched originals, celeba (faces ~100 px): face-check AUC | 0.859 |
|  - same set: swaps called FAKE / originals called FAKE (p>=0.7) | 70% / 15% |
|  - same set, image models only (AUC / swaps FAKE) | 0.659 / 26% |
Weak spots seen: small faces (~100 px) raise false alarms (13-20% of real celeba photos flagged), hard sets FaceForensics
c23 (49% caught) and DFDC (54-66%), own phone videos (40%).

## What changes vs the video round 4 recipe
1. **Separate checkpoint** `image_face_swap_r1.pt`; the video model stays untouched. `face_swap.py` gets a
   `FACE_SWAP_CHECKPOINT` override (default = video round 4) so swapping is one env var and rolling back is deleting it.
2. **New training data = consumer-style swaps of still photos**: `gen_photo_swaps.py` (inswapper_128, verified working
   locally: 80 pairs in minutes on CPU) run on a GPU at scale over real face photo sets. Each target gives a PAIRED
   real original + swapped copy, both JPEG q95, so the model must learn the swap, not the file. Split by PERSON
   (CelebA identity list / UTKFace has none -> use CelebA only for train/val/test), donors only from the same split.
3. **Keep the video-frame sources** (FF++, Celeb-DF, DFDC, DFD, RTFS, FakeAVCeleb, phone swaps) so video-style fakes
   don't regress; same crops, same degradation augmentation, label smoothing 0.1, init from round-4 best.pt, 4 epochs.
4. **Small-face robustness**: add a random downscale-to-60-130 px-face augmentation before the 224 crop (matches
   phone-gallery and cropped-avatar uploads) applied to real and fake alike.
5. Eval is per FRAME (what production does), not per 16-frame video mean.

## Acceptance criteria (all must hold, on data never used for training/selection/calibration)
- video test frames: AUC >= 0.95 and fakes caught at 5% false alarms >= 75% (now ~66-70%)
- consumer-swap test (new persons): AUC >= 0.93 (now 0.86); originals wrongly FAKE <= 5% (now 15%)
- real-face sets (celeba / 140k faces / UTKFace / coco faces): wrongly FAKE <= 5%
- own phone videos not worse than now; fairness check (UTKFace by race x gender) spread no worse than 8 points
- then refit the Platt calibration on a separate fit split and re-run the live-API test before any swap (needs your OK).

## Honest limits
Only one swap engine (inswapper) is available for generated data; other apps (SimSwap, GAN-based, diffusion-based "AI
headshot" swaps) are not covered, so test results on inswapper overstate generalisation. Real photos from several
actual apps would be the best evidence; 20-30 are enough to start.

## Status (2026-10-16): notebook BUILT and smoke-tested, NOT pushed or run
`kaggle_photo_r1/` holds `photo_swap_r1.py` (the cells), `gen_celeba_swaps.py` (swap generator, runs as a separate process),
`TruthLens_Photo_Swap_r1.ipynb` + `kernel-metadata.json` (built by `build_photo_notebook.py`). Inputs it attaches:
`truthlens-video-finetune-v2` (its saved crops - verified real 3584x224 strips - and `run_r4/best.pt`),
`truthlens-video-ft-inputs` (manifest_r4.csv), `jessicali9530/celeba-dataset`, `kymo9890/identity-celeba`.
Local CPU smoke run (tiny sizes, real inswapper + real production crop code + real r4 checkpoint) passed end to end:
swap generation, photo crops, training step, val/test reports, per-item score CSVs. The GPU path (onnxruntime-gpu install,
Kaggle paths, the git clone) is untested until the first Kaggle run.
Kaggle push was refused 2026-10-16 ("Maximum weekly GPU quota of 30.00 hours reached"), so a **Colab variant** was built:
`kaggle_photo_r1/TruthLens_Photo_Swap_r1_colab.ipynb` (same cells + a setup cell: mounts Drive for resume, reads a Kaggle API
token from the Colab secret `KAGGLE_API_TOKEN`, checks the GPU). It downloads the inputs with the Kaggle CLI (verified here:
dataset names, `manifest_r4.csv` and `identity_CelebA.txt` layout; NOT verified: the `kernels output --file-pattern` download of
the ~7 GB of crops + `best.pt`, and the Colab GPU/onnxruntime-gpu path). Colab has 2 CPU cores, so data loading is slower than
Kaggle (est. 2-3.5 h total on a T4). Kaggle route when quota resets: `kaggle kernels push -p image_retrain/kaggle_photo_r1`.

## Needs from you
A go-ahead to push and run it on Kaggle (GPU quota). Nothing has been uploaded or started.

## Update 2026-10-16 (later): downloaded data replaces generated swaps; Colab-safe
Colab warned about "disallowed code" (its FAQ bans creating deepfakes), so the notebooks NO LONGER generate any swaps.
Photo data is downloaded from Hugging Face instead:
- **Train/val/test: `Sowaiba01/Deepfake`** (MIT): 5,352 real photos (256 px) each paired with its inswapper_128 swap; split
  80/10/10 by connected face groups (pairs sharing any source/target face stay together; asserted leak-free in the notebook).
- **Final test only: `Scam-AI/RWFS`** (CC BY-NC-SA, gated, free login + terms): 500 real-app swaps (Pixlr, Magic Hour,
  Remaker, ...) + 500 real photos. Never used for training or model selection. Needs a read token as Colab secret `HF_TOKEN`
  (skipped, with a message, if absent).
Baselines measured today with the production check (Haar-then-MTCNN) before any retraining:
| Set | Swaps called FAKE | Real called FAKE | AUC |
|---|---|---|---|
| DeepGuard (250 pairs, faces ~170 px) | 91% (face check alone 89%) | 8% | face check 0.970 |
| RWFS (real apps, 500+500) | 38% (UNCERTAIN 28%, REAL 33%) | 16% | combined 0.672, face check alone 0.617 |
So the video-trained face check handles inswapper-style swaps but NOT what real apps produce - the gap this run targets.
The earlier generator (`gen_celeba_swaps.py`, `gen_photo_swaps.py`) is kept in the repo for local use only; it is no longer in
the notebooks. Acceptance criteria above apply, with RWFS as the headline (AUC now 0.67; target >= 0.80 at <= 5% false alarms).
