"""Pure-numpy evaluation metrics (no scikit-learn dependency).

Conventions: labels y are 0 = REAL, 1 = FAKE; scores s are P(FAKE) in [0, 1].
"""

from __future__ import annotations

import numpy as np


def _rankdata_avg(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j < len(x) - 1 and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def has_both_classes(y) -> bool:
    y = np.asarray(y)
    return bool((y == 0).any() and (y == 1).any())


def roc_auc(y, s) -> float:
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata_avg(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_curve(y, s):
    """Returns fpr, tpr arrays (thresholds descending, starting at (0, 0))."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    # keep only the last index of each distinct score
    distinct = np.r_[np.where(np.diff(s_sorted))[0], len(s_sorted) - 1]
    tpr = np.r_[0.0, tps[distinct] / max(tps[-1], 1)]
    fpr = np.r_[0.0, fps[distinct] / max(fps[-1], 1)]
    return fpr, tpr


def _precision_recall_points(y, s):
    """(recall, precision) at each DISTINCT score threshold, descending (ties collapsed)."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    n_pos = int((y == 1).sum())
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    distinct = np.r_[np.where(np.diff(s_sorted))[0], len(s_sorted) - 1]
    tp, fp = tp[distinct], fp[distinct]
    return tp / max(n_pos, 1), tp / (tp + fp)


def average_precision(y, s) -> float:
    if int((np.asarray(y) == 1).sum()) == 0:
        return float("nan")
    recall, precision = _precision_recall_points(y, s)
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def pr_curve(y, s):
    return _precision_recall_points(y, s)   # recall, precision


def confusion(y, s, threshold: float) -> dict:
    y = np.asarray(y)
    pred = (np.asarray(s, dtype=float) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def summary(y, s, threshold: float) -> dict:
    """Threshold metrics + AUC/AP. Fields that are undefined come back as NaN."""
    y = np.asarray(y)
    c = confusion(y, s, threshold)
    n = len(y)

    def div(a, b):
        return float(a / b) if b else float("nan")

    recall = div(c["tp"], c["tp"] + c["fn"])            # fake recall / sensitivity
    specificity = div(c["tn"], c["tn"] + c["fp"])       # real recall
    precision = div(c["tp"], c["tp"] + c["fp"])
    f1 = (2 * precision * recall / (precision + recall)
          if precision == precision and recall == recall and (precision + recall) > 0
          else float("nan"))
    parts = [v for v in (recall, specificity) if v == v]
    return {
        "n": n,
        "n_real": int((y == 0).sum()),
        "n_fake": int((y == 1).sum()),
        "accuracy": div(c["tp"] + c["tn"], n),
        "balanced_accuracy": float(np.mean(parts)) if parts else float("nan"),
        "fake_recall": recall,
        "real_specificity": specificity,
        "precision": precision,
        "f1": f1,
        "auc": roc_auc(y, s),
        "ap": average_precision(y, s),
        **c,
    }


def expected_calibration_error(y, p_fake, n_bins: int = 10):
    """Reliability of P(FAKE): returns (ece, bin_confidence, bin_accuracy, bin_count)."""
    y = np.asarray(y)
    p = np.asarray(p_fake, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    conf, acc, cnt = [], [], []
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        cnt.append(int(m.sum()))
        if not m.any():
            conf.append(float("nan"))
            acc.append(float("nan"))
            continue
        c_b = float(p[m].mean())
        a_b = float((y[m] == 1).mean())
        conf.append(c_b)
        acc.append(a_b)
        ece += m.mean() * abs(a_b - c_b)
    return float(ece), np.array(conf), np.array(acc), np.array(cnt)


def bootstrap_ci(fn, y, s, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI of fn(y, s). Skips resamples where fn is undefined."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        v = fn(y[i], s[i])
        if v == v:
            vals.append(v)
    if len(vals) < 20:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
