"""The evaluation metrics behind every number in backend/eval/results must be right."""

import numpy as np
import pytest

from eval import metrics as M


def test_perfect_and_worst_and_chance_auc():
    y = np.array([0, 0, 1, 1])
    assert M.roc_auc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert M.roc_auc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert M.roc_auc(y, [0.5, 0.5, 0.5, 0.5]) == 0.5           # all ties


def test_undefined_cases_are_nan_not_zero():
    assert np.isnan(M.roc_auc([1, 1, 1], [0.1, 0.5, 0.9]))
    assert np.isnan(M.average_precision([0, 0], [0.1, 0.2]))
    assert M.summary([1, 1], [0.9, 0.8], 0.5)["real_specificity"] != M.summary([1, 1], [0.9, 0.8], 0.5)["real_specificity"]  # NaN


def test_summary_counts_and_rates():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.2, 0.6, 0.1, 0.1, 0.2, 0.3])
    r = M.summary(y, s, 0.5)
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (2, 1, 1, 4)
    assert r["accuracy"] == pytest.approx(6 / 8)
    assert r["fake_recall"] == pytest.approx(2 / 3)
    assert r["real_specificity"] == pytest.approx(4 / 5)
    assert r["balanced_accuracy"] == pytest.approx((2 / 3 + 4 / 5) / 2)


def test_threshold_boundary_is_inclusive_for_fake():
    assert M.confusion([1], [0.5], 0.5)["tp"] == 1


def test_calibration_of_a_perfectly_calibrated_predictor_is_near_zero():
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)
    ece, *_ = M.expected_calibration_error(y, p)
    assert ece < 0.02


def test_calibration_flags_an_overconfident_predictor():
    y = np.array([0, 1] * 500)                                  # truth: 50% fake
    ece, *_ = M.expected_calibration_error(y, np.full(1000, 0.99))     # claims 99% fake
    assert ece == pytest.approx(0.49, abs=0.01)


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    s = np.clip(y * 0.3 + rng.random(400) * 0.7, 0, 1)
    lo, hi = M.bootstrap_ci(M.roc_auc, y, s, n_boot=300)
    assert lo <= M.roc_auc(y, s) <= hi and hi - lo < 0.2


def test_matches_scikit_learn_including_tied_scores():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(20, 300))
        y = rng.integers(0, 2, n)
        if y.min() == y.max():
            continue
        s = np.round(rng.random(n), int(rng.integers(1, 3)))    # heavy ties
        assert M.roc_auc(y, s) == pytest.approx(sk.roc_auc_score(y, s), abs=1e-9)
        assert M.average_precision(y, s) == pytest.approx(sk.average_precision_score(y, s), abs=1e-9)
        thr = 0.5
        pred = (s >= thr).astype(int)
        assert M.summary(y, s, thr)["accuracy"] == pytest.approx(sk.accuracy_score(y, pred))
        assert M.summary(y, s, thr)["balanced_accuracy"] == pytest.approx(sk.balanced_accuracy_score(y, pred))
