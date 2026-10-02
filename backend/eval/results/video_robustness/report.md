# Video Model v1 -- robustness under per-frame degradation (reduced settings, read-only measurement)

Not part of the deployed pipeline; see CLAUDE.md section 23. Each row perturbs all 16 sampled frames of each video identically (same family/level) before the unmodified Haar-crop + model pipeline.

| family | level | n | accuracy | balanced_acc | fake_recall | real_spec | auc |
|---|---|---|---|---|---|---|---|
| clean | 0 | 30 | 63.3% | 60.0% | 70.0% | 50.0% | 0.665 |
| jpeg | 10 | 30 | 33.3% | 50.0% | 0.0% | 100.0% | 0.585 |
| jpeg | 50 | 30 | 36.7% | 50.0% | 10.0% | 90.0% | 0.625 |
| jpeg | 90 | 30 | 66.7% | 62.5% | 75.0% | 50.0% | 0.745 |
| resize | 0.25 | 30 | 40.0% | 45.0% | 30.0% | 60.0% | 0.465 |
| resize | 0.5 | 30 | 53.3% | 50.0% | 60.0% | 40.0% | 0.570 |
| blur | 1 | 30 | 56.7% | 50.0% | 70.0% | 30.0% | 0.540 |
| blur | 3 | 30 | 40.0% | 45.0% | 30.0% | 60.0% | 0.410 |
| noise | 10 | 30 | 50.0% | 60.0% | 30.0% | 90.0% | 0.685 |
| social | webp | 30 | 40.0% | 47.5% | 25.0% | 70.0% | 0.545 |
| social | whatsapp | 30 | 46.7% | 57.5% | 25.0% | 90.0% | 0.610 |
