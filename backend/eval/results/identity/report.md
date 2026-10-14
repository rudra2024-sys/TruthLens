# Identity-match threshold calibration

Source: `jessicali9530/lfw-dataset` on Kaggle (LFW DevTest split, 500 same-person +
500 different-person labeled pairs). Scored with the real, deployed `IdentityPipeline.embed()`
and `detector.py::_cosine_similarity` (see `eval/run_identity_predictions.py`) -- not a
re-implementation.

- Pairs scored: 1000 (500 same-person, 500 different-person)
- Same-person similarity: mean 0.720, std 0.201, min -0.264, max 0.963
- Different-person similarity: mean 0.029, std 0.160, min -0.418, max 0.497
- ROC AUC: 0.9736

## Best-accuracy threshold

| Threshold | Accuracy | False-accept rate | False-reject rate |
|---|---|---|---|
| 0.4411 | 0.9650 | 0.0040 | 0.0660 |

False-accept rate = fraction of different-person pairs scored as a match at this threshold.
False-reject rate = fraction of same-person pairs scored as NOT a match at this threshold.

## Caveats

- LFW is frontal, studio/press-photo-quality, mostly well-lit faces of public figures -- not
  webcam footage. A webcam enrollment/monitoring photo (worse lighting, off-angle, compression)
  will plausibly show more overlap between the two distributions than this measurement does, so
  treat this threshold as a real, data-grounded starting point, not a guaranteed production
  false-accept/false-reject rate.
- The monitoring-session thresholds (`LOOKING_AWAY_*`, `TALKING_SPEECH_RATIO_THRESHOLD`,
  `OBJECT_DETECTION_CONFIDENCE_THRESHOLD`) and item 9's `LIVENESS_MOTION_EPSILON_FRACTION` are
  NOT calibrated by this study -- no comparable labeled dataset was found for any of them, and
  they remain stated placeholders in `config.py`.
