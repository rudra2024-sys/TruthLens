# TruthLens evaluation harness

Measures the **deployed** detectors (same classes the API calls) on labeled media and
produces report-ready tables and figures. It changes no model, weight or threshold.

Three steps, run from `backend/` (paths below use the repo `.venv`):

```powershell
# 1. describe the labeled data (label 0 = real, 1 = fake)
../.venv/Scripts/python.exe -m eval.build_manifest --media image --out eval/data/manifest_image.csv `
    --add "cifake_test=D:/data/cifake/test,trained_on=yes,max=500" `
    --fake "chatgpt_everyday=test_data/blind_spot_images/chatgpt_everyday,trained_on=no"

# 2. slow step: run the models, save raw scores (resumable)
../.venv/Scripts/python.exe -m eval.run_predictions --media image `
    --manifest eval/data/manifest_image.csv --out eval/data/pred_image.csv

# 3. fast step: metrics + figures from the stored scores
../.venv/Scripts/python.exe -m eval.make_report --media image `
    --pred eval/data/pred_image.csv --out eval/results/image
```

Same for `--media video` (deployed Video Model v1, threshold 0.525).

Output in `eval/results/<media>/`: `report.md`, `summary.csv`, `per_source.csv`, and
`roc_pr.png`, `confusion_verdicts.png` (incl. UNCERTAIN band), `score_hist.png`,
`reliability.png` (calibration/ECE), `per_source_accuracy.png`, `threshold_sweep.png`.

Image reports compare ConvNeXt alone, CLIP alone, average, and max (deployed) - the ensemble ablation.
Evaluate a rollback checkpoint with `--convnext-ckpt` / `--clip-ckpt` / `--video-ckpt` and a different `--out`.

**Honest-evaluation rules**
- Mark each source `trained_on=yes|no|unknown`. Only `no` sources measure generalisation.
- With few files, read the AUC bootstrap interval, not the point estimate.
- `eval/data/` (manifests + raw scores) is machine-specific and git-ignored.
