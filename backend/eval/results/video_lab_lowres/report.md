# TruthLens video evaluation

- Predictions: `pred_video_lab.csv`  |  files: **600** (300 real, 300 fake)  |  sources: 3
- Decision threshold: **0.525**  |  deployed system: **video_cnn**  |  UNCERTAIN band: +-0.2
- Labels: 0 = real, 1 = fake. All numbers computed from stored model scores (`run_predictions.py`) using the exact production pipeline classes.

## Overall (all sources pooled)

| system | n | accuracy | balanced acc | fake recall | real specificity | precision | F1 | AUC | AP |
|---|---|---|---|---|---|---|---|---|---|
| **video_cnn** | 600 | 50.8% | 50.8% | 5.0% | 96.7% | 60.0% | 0.092 | 0.589 [0.544, 0.633] | 0.572 |

AUC brackets are 95% bootstrap confidence intervals (1000 resamples).

Calibration of the deployed system: ECE = **0.030** (10 bins; lower is better).

## Per source

| source | trained on? | real | fake | video_cnn acc |
|---|---|---|---|---|
| lab_celebdf | unknown | 100 | 100 | 52.5% |
| lab_dfdc | unknown | 100 | 100 | 48.5% |
| lab_faceforensics | unknown | 100 | 100 | 51.5% |

`trained on?` records whether the deployed model saw that source in training. `yes` sources measure fit, not generalisation; only `no` sources are honest held-out tests. Fake-only or real-only sources report recall on that single class.

## Figures

![roc_pr.png](roc_pr.png)

*ROC and precision-recall curves, all sources pooled.*

![confusion_verdicts.png](confusion_verdicts.png)

*Verdict confusion matrix for the deployed system, including the UNCERTAIN band (0.33-0.73).*

![score_hist.png](score_hist.png)

*How separated the deployed system's scores are for real vs fake.*

![reliability.png](reliability.png)

*Calibration: does 90% confidence mean 90% correct? Points below the diagonal are over-confident.*

![per_source_accuracy.png](per_source_accuracy.png)

*Accuracy per data source and system.*

![threshold_sweep.png](threshold_sweep.png)

*Balanced accuracy as the decision threshold moves (dashed = deployed threshold). Descriptive only - no threshold is changed.*
