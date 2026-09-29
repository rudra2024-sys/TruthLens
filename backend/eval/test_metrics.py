"""Sanity check: eval/metrics.py must agree with scikit-learn on random data.
Run manually:  ../.venv/Scripts/python.exe -m eval.test_metrics
"""
import numpy as np
from eval import metrics as M

try:
    from sklearn import metrics as SK
except Exception as e:                                   # scikit-learn is optional
    raise SystemExit(f"scikit-learn not importable here ({e}); nothing to compare against")

rng = np.random.default_rng(0)
worst = 0.0
for trial in range(200):
    n = int(rng.integers(20, 400))
    y = rng.integers(0, 2, n)
    if y.min() == y.max():
        continue
    s = np.round(rng.random(n), int(rng.integers(1, 4)))          # many ties on purpose
    thr = float(rng.choice([0.3, 0.5, 0.525]))
    pred = (s >= thr).astype(int)
    checks = {
        "auc": (M.roc_auc(y, s), SK.roc_auc_score(y, s)),
        "ap": (M.average_precision(y, s), SK.average_precision_score(y, s)),
        "acc": (M.summary(y, s, thr)["accuracy"], SK.accuracy_score(y, pred)),
        "bal_acc": (M.summary(y, s, thr)["balanced_accuracy"], SK.balanced_accuracy_score(y, pred)),
        "f1": (M.summary(y, s, thr)["f1"], SK.f1_score(y, pred, zero_division=0)),
        "recall": (M.summary(y, s, thr)["fake_recall"], SK.recall_score(y, pred, zero_division=0)),
    }
    fpr, tpr = M.roc_curve(y, s)
    sf, st, _ = SK.roc_curve(y, s, drop_intermediate=False)
    checks["roc_auc_from_curve"] = (float(np.trapezoid(tpr, fpr)), SK.auc(sf, st))
    for k, (a, b) in checks.items():
        if a == a and b == b:                                     # skip NaN vs 0 conventions
            worst = max(worst, abs(a - b))
            assert abs(a - b) < 1e-9, f"{k}: mine={a} sklearn={b} (trial {trial})"
print(f"OK - 200 random trials (with tied scores), max abs difference vs scikit-learn = {worst:.2e}")
