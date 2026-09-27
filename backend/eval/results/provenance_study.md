# Provenance study

Files: 6532 (from `pred_image.csv`, `pred_b1.csv`, `pred_b2.csv`, `pred_b3.csv`, `pred_b4.csv`). AI = label 1.

## Metadata / Content Credentials per source

| source | files | AI declared | C2PA present | camera EXIF | any EXIF | nothing at all |
|---|---|---|---|---|---|---|
| aivshuman_train | 1000 | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| blind_spot_chatgpt_everyday | 8 | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| blind_spot_portrait_app | 15 | 26.7% | 0.0% | 0.0% | 0.0% | 73.3% |
| cifake_test | 1000 | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| deepdetect_test | 1000 | 0.0% | 0.0% | 0.1% | 0.8% | 99.2% |
| faces140k_test | 1000 | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| genimage_fake_biggan | 300 | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| genimage_fake_midjourney | 300 | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| genimage_fake_sd | 300 | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| genimage_real_imagenet | 600 | 0.0% | 0.0% | 4.0% | 6.2% | 93.8% |
| openfake_test | 1000 | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| personal_real_photos | 9 | 0.0% | 0.0% | 0.0% | 11.1% | 88.9% |

## Does provenance add to the models?

- AI images overall: **12 of 3423** (0.4%) carry an explicit AI declaration.
- AI images the deployed ensemble **misses** (scored < 0.5): 1026; of these, provenance declares **9** as AI (0.9%).

| source | AI images missed by models | ...declared AI by provenance |
|---|---|---|
| blind_spot_chatgpt_everyday | 8 | 8 (100.0%) |
| blind_spot_portrait_app | 1 | 1 (100.0%) |
| cifake_test | 1 | 0 (0.0%) |
| genimage_fake_biggan | 222 | 0 (0.0%) |
| genimage_fake_midjourney | 240 | 0 (0.0%) |
| genimage_fake_sd | 208 | 0 (0.0%) |
| openfake_test | 346 | 0 (0.0%) |

## Is ELA a usable detector? (JPEG sources with both classes)

AUC of each ELA summary statistic for FAKE vs REAL *within the same source* (0.5 = no information; the statistic direction is whichever gives AUC >= 0.5, so values below 0.5 cannot occur).

| source | real | fake | mean error | p95 error | block CV |
|---|---|---|---|---|---|
| aivshuman_train | 500 | 500 | 0.688 | 0.734 | 0.517 |
| cifake_test | 500 | 500 | 0.614 | 0.718 | 0.649 |
| deepdetect_test | 486 | 492 | 0.995 | 0.994 | 0.761 |
| faces140k_test | 500 | 500 | 0.534 | 0.573 | 0.543 |
| openfake_test | 500 | 500 | 0.752 | 0.740 | 0.642 |

Interpretation: values near 0.5 mean ELA carries little class information for that source, so it is shown in the product only as a labelled visual aid and never used as a score.
