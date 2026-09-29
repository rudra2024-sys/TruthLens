# TruthLens — datasets

Every dataset the project trained on or was measured against, what it was used for, and its terms. **Check each licence before
redistributing anything**: this repository does not contain the datasets themselves (they are large and re-downloadable), only
small test samples.

"**Seen**" = the model was trained on data from this source, so accuracy on it measures *fit*. "**Unseen**" = never trained on, so
accuracy measures *generalisation*. Reports must keep the two apart (the evaluation harness tags every source).

## Image

| Dataset | What it is | Size (as used) | Licence / terms | Role |
|---|---|---|---|---|
| **CIFAKE** — Bird & Lotfi, IEEE Access 2024 (`birdy654/cifake-real-and-ai-generated-synthetic-images`) | Real CIFAR-10 vs Stable Diffusion 1.4 images, 32×32 | 120 000 (60k + 60k) | MIT, as CIFAR-10 (cite Krizhevsky & Hinton 2009 and Bird & Lotfi 2024) | **Seen.** Baseline ConvNeXt, then v2 fine-tune and CLIP head |
| **AI vs Human-Generated Images** (`alessandrasala79/ai-vs-human-generated-dataset`) | Shutterstock photos paired with AI equivalents (2025 Women in AI competition) | ~80 000 labelled (train split) | Apache 2.0 | **Seen** |
| **140k Real and Fake Faces** (`xhlulu/140k-real-and-fake-faces`) | Flickr-Faces (real) vs StyleGAN faces, 256 px | 140 000 | "Other" — see the Kaggle description | **Seen** |
| **DeepDetect-2025** (`ayushmandatta1/deepdetect-2025`) | Real vs StyleGAN3 / DALL·E 3 / Midjourney / SD3 images | 100 000+ | Apache 2.0 | **Seen** |
| **Portrait-app source (curated from DiffusionDB)** — `poloclub/diffusiondb` | Vintage-photo / period-costume AI portraits, found by prompt search and reviewed by hand | 257 (fake-only) | CC0 | CLIP head round 3 only. **The zip has no other durable copy — keep `portrait_app_fake_source_v1.zip`** |
| **OpenFake** (`sanketghadge1/openfake-data-20k-img`) | Real vs 2025-era generator images | 500 + 500 sampled | **Unknown** — keep local, do not redistribute | **Unseen** (evaluation only) |
| **GenImage subset** (`renhuang8/genimage-subset-detection`) | ImageNet real photos vs Midjourney / Stable Diffusion / BigGAN | 600 real + 3 × 300 fake sampled | Apache 2.0 on Kaggle; check the original GenImage terms | **Unseen** (evaluation only) |
| **Repository sample sets** (`backend/test_data/blind_spot_images/`) | 8 ChatGPT "everyday" AI photos + 15 portrait-app images | 23 | Team-generated / CC0 | Blind-spot regression and provenance tests |

Known dataset quirks that affect how numbers should be read:
* **DeepDetect:** a trivial JPEG-compression statistic separates its real from fake images with AUC 0.995 — a *compression-history shortcut*.
  Models that score 99 % there may be using the same shortcut.
* **GenImage:** real images are JPEG and fakes are PNG at different resolutions (BigGAN is 128 px) — a format bias in either direction.
* **CIFAKE:** natively 32×32, so preprocessing that crushes resolution does nothing to it but destroys detail in other sources.

## Video

| Dataset | What it is | Licence / terms | Role |
|---|---|---|---|
| **Celeb-DF v2** | Celebrity face-swap deepfakes | Research licence from the authors | Video Model v1 was developed and verified against the original evaluation protocol (per the project README) |
| **RealTimeFaceSwap-10k** (`stplusplus/rtfs-10k`) | 9 772 face swaps (inswapper, uniface) of 1 636 real video-call-style clips | CC BY-SA 4.0 | **Unseen** for the deployed CNN — evaluation (300 clips sampled); training data for the Colab head experiments |
| **WildDeepfake subset** (`naisargirupareliya/wilddeepfake-subset`) | Frames from real internet deepfakes (Zi et al. 2020), face-cropped PNG folders | See Kaggle page | Colab head experiments only |
| **FF++ / Celeb-DF / DFDC face-only clips** (`pranabkc/deepfake-with-cropped-faces-from-video`) | Pre-cropped face clips | See Kaggle page | Colab experiments. **Every clip is 112×112 px** — an invalid test set for the deployed model (its scores collapse to ~0.5) |
| **Local ground-truth videos** (10) | 5 real + 5 fake, including two fakes from consumer apps (Akool, Magic Hour) | Team-owned | Sanity check; **too few to conclude anything** |

## Audio

| Dataset | Role |
|---|---|
| **ASVspoof 2019 LA** (Todisco et al.) | Training data of the **pretrained** AASIST model (Jung et al., ICASSP 2022) that TruthLens ships. Not trained by this team. |
| *(a teammate's separate Wav2Vec2 + LCNN model)* | Not integrated into this codebase; **its training data should be documented here by whoever owns it** |

## Provenance references
* **C2PA trust list** — bundled from `c2pa-org/conformance-public` (retrieved 2026-09-26; refresh periodically). See `backend/app/services/provenance/trust/README.md`.

## Licences to respect
* Do not commit datasets; `backend/eval/data/` and `D:\eval_data` are git-ignored on purpose.
* OpenFake (licence unknown) and the personal photos must stay local. RTFS is CC BY-SA 4.0 (attribution + share-alike).
* Cite: Bird & Lotfi (2024) CIFAKE; Krizhevsky & Hinton (2009) CIFAR-10; Jung et al. (2022) AASIST; Zi et al. (2020) WildDeepfake; Li et al. (2020) Celeb-DF; Selvaraju et al. (2017) Grad-CAM; the C2PA specification.
