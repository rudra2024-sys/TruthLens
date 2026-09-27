# TruthLens — ethics, limitations and responsible use

A detector is only useful if people understand what its answer means. This document says what TruthLens can and cannot tell you, with the
numbers behind each statement (all measured with the harness in [`backend/eval/`](../backend/eval/); sources and caveats in
[`EVALUATION_SUMMARY.md`](../backend/eval/results/EVALUATION_SUMMARY.md) and [`DATASETS.md`](DATASETS.md)).

## 1. How to read a verdict

| Verdict | What it actually means | What it does **not** mean |
|---|---|---|
| **REAL / AUTHENTIC** | The models found no evidence of AI generation. | That the file is genuine. On the ChatGPT-generated photos we tested, the models said REAL at 99.99 % confidence for **8 of 8**. |
| **FAKE / MANIPULATED** | The models' score is high. Treat it as a **lead to check**. | Proof. About 14 % of *real* photos from sources the models never saw were flagged FAKE. |
| **UNCERTAIN** | The score is too close to the threshold to call. | A weaker "fake". It is the honest answer for borderline cases. |

**Confidence is not the chance of being right.** The models are over-confident: a score of 99 % on an unfamiliar generator can still be wrong. (Calibration
error measured at 0.16 for images and 0.25 for video.)

## 2. Where it works and where it does not

**Image** (max of ConvNeXt-Tiny and CLIP; threshold 0.5)

| Data | Accuracy | AUC | AI images caught |
|---|---|---|---|
| Sources the models trained on (CIFAKE, DeepDetect, 140k Faces, AI-vs-Human) | **98.4 %** | 1.00 | 100 % |
| Generators never seen (GenImage: Stable Diffusion, Midjourney, BigGAN; OpenFake) | **≈ 53 %** | 0.67 | **≈ 27 %** |

* **The headline limit:** near-perfect on data like its training data, close to chance on new generators. Any new image model can be a blind spot until the
  detector is retrained on it. The documented blind spots: everyday photorealistic ChatGPT images (0/8 caught) and consumer face-swap apps.
* The "trained on" numbers measure *fit*, not generalisation (the exact training hold-out split was not recorded). Datasets can also carry shortcuts:
  in DeepDetect a trivial JPEG statistic separates real from fake almost perfectly, so 99 % there may partly be a dataset artefact.

**Video** (Video Model v1, face crops, threshold 0.525)

| Data | Accuracy | AUC |
|---|---|---|
| Unseen face swaps (RTFS: inswapper, uniface) | 79 % | 0.89 |
| Celeb-DF-v2, original protocol (6,529 videos) | — | 0.695 |
| Two consumer-app fakes (Akool, Magic Hour), 10 local videos in total | missed | too few videos to conclude anything |

The video model looks only at a detected face; if the face detector locks onto something else (it locked onto a wall poster in one clip), the model never sees the swap.

**Audio** uses a pretrained model (AASIST). This project has not evaluated it on its own data.

## 3. Robustness — what happens to real-world pictures

We re-tested 300 images (label-balanced; 180 from sources the models trained on, 120 from unseen ones) under 20 degradations
([`backend/eval/results/robustness/report.md`](../backend/eval/results/robustness/report.md)). Ranges are wide because the groups are small (90 real + 90 fake images from familiar sources, 60 + 60 from unseen ones); one image moves a rate by about a point, and differences smaller than the confidence intervals in the report are noise.

**Reassuring**
* **Everyday sharing barely matters** for images on familiar sources: a screenshot, a WhatsApp-style resize + JPEG q65 and WebP each kept AUC ≥ 0.99.
* Mild blur (σ ≤ 1), crops to 80 % and 75 % downscaling change the ranking little. **But not even mild JPEG is free:** at quality 90 the ranking barely moves (AUC 0.999) while
  the share of real photos wrongly flagged FAKE already rises from about 4 % to about 12 %.

**Worrying**
* **Faint noise makes real photos look fake.** Adding noise of only ±5/255 raised the share of *real* photos wrongly flagged FAKE from about 4 % to about 29 %; heavy
  JPEG compression (quality 10) raised it to about 38 %. In 9 of the 21 settings the combined rule kept fewer than 90 % of real photos real. The culprit is the CLIP sub-model:
  ConvNeXt alone kept 97–99 % of real photos real under **all 21** settings.
* **Heavy blur hides fakes.** At blur σ = 3, ConvNeXt alone caught only 50 % of AI images (the combined rule rescues that to 83 %).
* The combined `max` rule therefore inherits **CLIP's false alarms** and **ConvNeXt's misses** at different degradations — neither model is strictly safer. This is worth a
  deliberate look at retraining with noise/compression augmentation; **no model was changed here** (model changes need explicit approval).
* On unseen sources the ranking ability starts near chance (AUC 0.61 on this subset), so degradations mostly shift everything toward "FAKE" (more fakes caught *and* more
  real photos flagged) without improving discrimination.

**Provenance is fragile.** An AI declaration embedded in a file's metadata survives only a byte-for-byte copy. Every re-encode we tried — screenshot, resize, JPEG, blur, a plain "Save as" — removed it
(8/8 kept in a straight copy; 0/8 after any re-save). It can add evidence to an untouched file; its absence proves nothing.

## 4. Who can be harmed, and how

| Risk | Example | What TruthLens does about it |
|---|---|---|
| **False accusation** | A real photo or a genuine recording is labelled FAKE and used against its owner | Verdict bands (UNCERTAIN), explanations that are honest about being model behaviour not proof, this document, "lead, not proof" wording |
| **False reassurance** | An AI image labelled REAL is used to lend credibility to a hoax | The evidence layer flags conflicts with file metadata; the UI says REAL means "no evidence found" |
| **Over-trust in explanations** | A heatmap "highlights the fake part" | Grad-CAM shown with the caveat that it is not proof of manipulation; the heatmap fades when the model sees nothing suspicious, instead of implying evidence |
| **Unequal performance across people** | Face-based video detection may work differently across skin tones, ages, genders or camera types | **Not evaluated.** No demographic breakdown exists yet; treat any claim of fairness as unsupported |
| **Evasion** | A forger re-encodes or blurs an image | Measured above; provenance is stripped by re-encoding; detectors are unreliable on new generators |
| **Misuse as a targeting tool** | Using "detected as fake" to harass a creator | Not technically preventable; addressed by the wording and the responsible-use guidance below |
| **Dual use** | Someone tests fakes until one passes | The service can be used as an oracle; rate limits at the proxy limit bulk probing but do not stop it |

## 5. Privacy and consent

* Uploaded files, results and reports are stored on the server you run and are visible only to their owner (other accounts get 404).
* **Feedback is opt-in.** A user can say a verdict was right or wrong without allowing the file to be kept. Only when they tick the (default-off) box can the file and note be
  exported for evaluation; they can change or withdraw it at any time. Withdrawing removes the feedback and the consent.
* GPS location inside a photo is reported only as "present"; coordinates are never shown.
* **Gaps:** there is no automatic deletion and no "delete my data" button yet, and no audit log. See [`THREAT_MODEL.md`](THREAT_MODEL.md) §6.
* Using feedback to **train** the models is a separate decision that needs explicit approval; the export tool is for evaluation.

## 6. Responsible use (suggested wording for users)

1. Treat a verdict as **one input**. Check the source, the context and other evidence.
2. Never treat **REAL** as proof, or **FAKE** as proof.
3. Look at the **Provenance** panel: a declared AI origin is strong evidence; missing metadata is no evidence either way.
4. Do not use TruthLens for legal, employment or safety decisions about a person without human review.
5. If a result looks wrong, use **"No, it is wrong"** — that is how blind spots get found.

## 7. What would make it better (in order)

1. Retrain with **more generators** (including ChatGPT-class images and consumer face-swap apps) and **noise/compression augmentation** — this is where the measurements point. *(Needs approval.)*
2. Evaluate on **demographic slices** and on more real-world videos than the current 10.
3. Add **data retention controls** and a delete-my-data endpoint.
4. Decide deliberately whether provenance should influence the verdict (today it is shown next to it, never merged).
5. Report calibrated probabilities instead of raw scores.
