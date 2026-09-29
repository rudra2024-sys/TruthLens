"""The robustness report generator, on synthetic scores (no models, no datasets)."""

import csv
import sys

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from eval import make_robustness_report as R      # noqa: E402
from eval import perturbations as P               # noqa: E402


def write_predictions(path, images=12, drop_tail=False, damage_jpeg=True):
    """Fake scores where real ~ 0.1 and fake ~ 0.9, degrading toward 0.5 for jpeg quality 10."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(images):
        label = i % 2
        status = "seen" if i < images // 2 else "unseen"
        for fam, lvl in P.settings():
            noise = 0.05 * rng.standard_normal()
            base = 0.9 if label else 0.1
            if damage_jpeg and fam == "jpeg" and lvl == 10:
                base = 0.5                                                  # the model is blind at jpeg q10
            s = float(np.clip(base + noise, 0, 1))
            rows.append({"path": f"img{i}.png", "label": label, "group": "g", "status": status, "source": "s",
                         "family": fam, "level": lvl, "convnext_fake": s, "clip_fake": s})
    if drop_tail:
        rows = rows[:-5]                                                    # an interrupted run: last image incomplete
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run(tmp_path, monkeypatch, **kw):
    pred, out = tmp_path / "p.csv", tmp_path / "out"
    write_predictions(pred, **kw)
    monkeypatch.setattr(sys, "argv", ["make_robustness_report", "--pred", str(pred), "--out", str(out), "--boot", "20"])
    R.main()
    return out


def test_report_files_are_created(tmp_path, monkeypatch):
    out = run(tmp_path, monkeypatch)
    for name in ("report.md", "summary.csv", "auc_by_family.png", "recall_by_family.png", "specificity_by_family.png",
                 "models_seen_auc.png", "models_unseen_auc.png", "heatmap.png"):
        assert (out / name).is_file() and (out / name).stat().st_size > 0, name


def test_report_numbers_reflect_the_damage(tmp_path, monkeypatch):
    out = run(tmp_path, monkeypatch)
    with open(out / "summary.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["system"] == "max" and r["group"] == "all"]
    auc = {(r["family"], r["level"]): float(r["auc"]) for r in rows}
    assert auc[("clean", "0")] == 1.0                                       # perfectly separated synthetic scores
    assert auc[("jpeg", "10")] < 0.75 < auc[("jpeg", "90")]                 # the blind setting is visibly worse
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "jpeg 10" in text and "Most damaging" in text


def test_incomplete_images_from_an_interrupted_run_are_excluded(tmp_path, monkeypatch, capsys):
    run(tmp_path, monkeypatch, images=6, drop_tail=True)
    assert "5 complete images" in capsys.readouterr().out                  # the 6th image lost rows and is dropped
