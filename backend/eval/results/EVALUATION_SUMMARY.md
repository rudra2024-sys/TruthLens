# TruthLens — evaluation summary (generated 2026-09-26)

Measured with `backend/eval/` on the **deployed** detectors (same classes the API calls; no model, weight or
threshold was changed). Full tables and figures: `image/report.md`, `video/report.md`,
`video_lab_lowres/report.md`. Metric code is verified against scikit-learn (`eval/test_metrics.py`,
max difference 2e-16).

## Headline

**The models are near-perfect on the kinds of data they were trained on and close to chance on generators
they have not seen.** This is the main finding and the main limitation of the system.

## Images (deployed = max of ConvNeXt-Tiny and CLIP; threshold 0.5)

Data: 6,532 files from 12 sources (up to 500 per class per dataset, taken from each dataset's test folder,
or its train folder where no labeled test split exists), plus the local blind-spot sets and own photos.

| Data | Files | Accuracy | AUC | Fakes caught | Real kept real |
|---|---|---|---|---|---|
| **Seen sources** (CIFAKE, DeepDetect-2025, 140k Faces, AI-vs-Human) | 4,000 | **98.4%** | 1.000 | 100.0% | 96.8% |
| **Unseen sources** (GenImage, OpenFake, ChatGPT, own photos) | 2,517 | **53.2%** | 0.665 | 27.3% | 86.0% |

Unseen, per source (deployed system):

| Source | What it is | Result |
|---|---|---|
| GenImage – Stable Diffusion | 300 fakes | 30.7% caught |
| GenImage – Midjourney | 300 fakes | 20.0% caught |
| GenImage – BigGAN | 300 fakes | 26.0% caught |
| GenImage – real (ImageNet) | 600 real | 92.7% kept real |
| OpenFake (2025 generators) | 500 + 500 | 54.3% accuracy (30.8% fakes caught, 77.8% real kept real) |
| ChatGPT "everyday" images | 8 fakes | 0 / 8 caught (known blind spot) |
| Portrait-app images | 15 fakes | 93.3% caught (CLIP was trained on this genre; ConvNeXt 6.7%) |

**Ensemble ablation** (the reason `max` is deployed): on seen data average is slightly better than max
(99.6% vs 98.4%). On unseen data max catches more fakes (27.3% vs 10.9%) at the price of more false alarms
on real photos (86.0% vs 96.3% real kept real). On the portrait-app genre max keeps CLIP's 93.3% where
average drops to 66.7%. This measures the trade-off recorded in `CLAUDE.md` section 3.

**Calibration:** ECE 0.159. Scores near 0 are wrong for about a quarter of the images (unseen fakes get
confidently "real"), so the confidence number should not be read as a probability of being correct.

## Video (deployed = Video Model v1, EfficientNet-B0, threshold 0.525)

| Data | Files | Accuracy | AUC (95% CI) | Fakes caught | Real kept real |
|---|---|---|---|---|---|
| RTFS face swaps (inswapper + uniface, **unseen**) + own 10 videos | 310 | 79.4% | 0.892 [0.852, 0.926] | 74.2% | 84.5% |

- inswapper 77.3% caught, uniface 72.0% caught, real originals 84.0% kept real.
- Own 10 videos alone: 80% accuracy, AUC 0.64 — too few files to conclude anything. The two commercial-app
  fakes documented earlier (Akool, Magic Hour) are still missed.
- ECE 0.249 (over-confident).

**Not a valid test:** the 600 Celeb-DF / DFDC / FF++ clips of `pranabkc/deepfake-with-cropped-faces-from-video`
(`video_lab_lowres/`). Every clip is 112x112 pixels, pre-cropped by the dataset author; all scores collapse
to 0.45–0.50 (accuracy 50.8%, AUC 0.589). That measures the model's sensitivity to input resolution, not its
skill on Celeb-DF/DFDC/FF++. It also matters for training: features from these clips are low-resolution
upscaled faces, unlike real-world uploads.

## Caveats (state these in the report)

- **"Seen" sources are test/train folders of datasets used in training.** The exact hold-out split used at
  training time was not recorded, so 98–99% is a fit measurement, not proof of generalisation.
- **GenImage subset:** real images are JPEG, fakes are PNG at different resolutions (BigGAN is 128px); such
  format differences can bias a detector in either direction.
- **RTFS:** real originals and generated fakes went through different encodes, which a video model may partly
  exploit.
- Sample sizes: 300–500 per source (CIs are in the reports); the blind-spot sets are only 8 and 15 images.
- OpenFake license is "unknown" and RTFS is CC BY-SA 4.0: keep these datasets local, do not redistribute.

## What this suggests (no action taken; model changes need explicit approval)

1. Any retraining should add generator diversity (Midjourney/SD-XL/Flux/GPT-image outputs) — more of the
   same distribution will not help.
2. For the video head being trained on Colab: the lab clips are 112px; RTFS and WildDeepfake are closer to
   real uploads, so their weight should not be diluted by the 112px data.
3. Report accuracy separately for seen and unseen data everywhere (as this harness does).

## Provenance study (added after the explainability/provenance work)

Full tables: `provenance_study.md`. Across the same 6,532 images:

- **Provenance rescues exactly the images the models miss.** All 8 ChatGPT "everyday" images (models 0/8, scored REAL at
  99.99%+) carry OpenAI Content Credentials declaring them AI-generated; so does the 1 portrait-app image the ensemble
  missed. Overall, 9 of the 1,026 AI images the ensemble misses are declared AI by their own metadata.
- **But coverage is low outside fresh downloads:** only 0.4% of the AI images carry any declaration, because the benchmark
  datasets (CIFAKE, GenImage, OpenFake, ...) were re-encoded and stripped. Provenance adds evidence; it can never clear a
  file, because metadata is trivially stripped.
- **ELA is not a detector.** Within-source AUC of its summary statistics: 0.53-0.75 on most sources, **0.995 on
  DeepDetect**. That near-perfect separation from a trivial compression statistic shows DeepDetect's real and fake images
  differ in JPEG history; the models (99% on DeepDetect) may be learning the same shortcut, which is consistent with —
  but not proof of — their collapse on unseen generators. Worth stating in the report's limitations.

## Robustness (image detectors under real-world degradation)

Full tables and charts: `robustness/report.md`. 300 label-balanced images (180 from sources the models trained on, 120 unseen) x 21 settings
(clean + JPEG q90-10, downscale-and-restore 75-25 %, blur sigma 0.5-3, noise std 5-20, centre crop 80/60 %, screenshot / WhatsApp-style / WebP),
scored by the production classes; the clean baseline reproduces the earlier evaluation exactly. Small groups: differences inside the confidence
intervals are noise.

- **Everyday sharing is survivable on familiar sources:** screenshot, WhatsApp-style (max side 1600 + JPEG q65) and WebP kept the deployed AUC >= 0.99.
- **Faint noise and heavy compression cause false alarms on real photos, and CLIP is why.** Real-photo specificity of the deployed max rule fell from
  95.6 % to 71.1 % at noise std 5, to 62.2 % at JPEG q10, and to 87.8 % already at JPEG q90; it was under 90 % in 9 of 21 settings. ConvNeXt alone stayed
  at 96.7-98.9 % in **all 21**; CLIP dropped to 63 % (q10).
- **Heavy blur hides fakes, and ConvNeXt is why:** at sigma 3, ConvNeXt alone caught 50 % of AI images (max rule: 83 %).
- So the max rule inherits CLIP's false alarms and ConvNeXt's misses at different degradations. Under degradation the earlier "max costs ~0.9 points" trade-off (CLAUDE.md
  section 3) is larger than that for real photos; noise/compression augmentation when retraining is the obvious next step (needs approval).
- **Unseen sources:** the baseline is already near chance (AUC 0.61 [0.51, 0.70] on this subset), and degradations mostly shift all scores toward FAKE
  (e.g. JPEG q10: recall 16.7 % -> 46.7 %, specificity 81.7 % -> 65.0 %) without improving discrimination.

`provenance_survival.md`: an AI declaration in file metadata survives only a byte-for-byte copy (ChatGPT credentials 8/8); every re-encode removed it (0/8). Only
4 of the 15 portrait-app images carried a prompt/seed to begin with.
