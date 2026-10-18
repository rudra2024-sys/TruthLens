# Image model retrain — round 1 (started 2026-10-15)

Approved by the user on 2026-10-15 after the baseline measurement below ("yes, start with step 1"). Step 1 = source data and
build a leakage-safe manifest. **Nothing has been trained, and the production image checkpoints are untouched.**

## Baseline that motivated this (deployed max ensemble, 2026-10-15, run locally)
| Set | n | Result |
|---|---|---|
| Real faces (UTKFace, even sample) | 297 | 274 REAL, 12 UNCERTAIN, 11 wrongly FAKE (3.7%) |
| AI-generated, GenImage (7 generators) | 10 | 2 FAKE (both Wukong), 1 UNCERTAIN, 7 REAL |
| AI-generated, ChatGPT "everyday" | 8 | 0 caught |
| Face-swap stills (FakeAVCeleb frames) | 3 | 0 FAKE |
| Portrait-app (CLIP's own training genre) | 15 | 14 FAKE |
ConvNeXt is the blind one; CLIP alone produced every catch. Scores: `testing/README.md` (2026-10-09) matches exactly.

## Sources (see `build_manifest_r1.py` docstring for the exact rules)
| Source | Use | Why |
|---|---|---|
| `cartografia/unbiased-tiny-genimage` (CC BY-NC-SA 4.0) | train/val/test + **ADM held out entirely** | the six generators we miss, JPEG-matched to its ImageNet reals |
| `hiddenplant/sut-project` (MIT) | train/val/test | Midjourney V6 / Flux / NanoBanana: newest generators found; closest to GPT-4o-class |
| `ajaysonicu/deepfakefusion-399k-realfake-faces` (MIT) FF++ + Celeb-DF rows, train/val only | train/val | face-swap stills (the 0/3 miss) |
Rehearsal (keep what works) and eval-only (never trained) lists are in `plan.json`.

## Known problems to handle in step 2 (not fixed by the manifest)
1. **Format shortcut in SUT**: every fake is PNG, reals are 63% PNG / 37% JPEG. Training must JPEG-re-encode both classes
   (random quality) or the model learns "PNG = fake". Same family of bug as DeepDetect's JPEG-history shortcut (CLAUDE.md §10).
2. **DeepFakeFusion frame leakage**: no video id in filenames, so its own test split is unusable; face-swap generalisation must
   be judged on external stills (extract frames from FakeAVCeleb videos; only 3 exist now).
3. **Face crops vs full photos**: DFF stills are tight face crops; users upload full photos. Expect weaker transfer than the
   in-distribution number suggests.
4. **No true GPT-4o-class training data found** on Kaggle (searched ChatGPT/DALL-E/gpt-image/Imagen/Flux). SUT's NanoBanana is the
   nearest. The 8 ChatGPT images stay eval-only; the ChatGPT blind spot may remain after round 1. Provenance (C2PA) already
   catches those 8 (CLAUDE.md §12).
5. **Class balance**: new data is ~29k fake vs ~14k real in train; rehearsal reals (CIFAKE/AI-vs-Human/140k/DeepDetect) rebalance it.
6. **GenImage re-use**: its generators are no longer "unseen" after this; truly unseen evaluation = ADM (held out), OpenFake-20k,
   the ChatGPT 8, FakeAVCeleb stills.

## Files
`build_manifest_r1.py` → `manifest_r1.csv` (54,518 rows: dataset, split, group, label, kind, path, source, fmt), `plan.json`.
`work/` (dataset indexes, git-ignored) holds the zip file listings read over HTTP range requests; no images were downloaded.
Licences: GenImage is CC BY-NC-SA (research only, don't redistribute); add the three sources to `docs/DATASETS.md` when training starts.
