"""Tests for the video robustness eval scripts (CLAUDE.md section 23): the deterministic subset picker and
the report's metrics computation. Not the model itself -- see test_models.py / CLAUDE.md section 23 for the
measured result these scripts produced.
"""
import csv

from eval.run_video_robustness import pick_subset


def _write_manifest(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "source", "trained_on"])
        w.writerows(rows)


def test_pick_subset_is_deterministic_and_balanced_per_source(tmp_path):
    manifest = tmp_path / "m.csv"
    rows = [(f"/v/{s}_{i}.mp4", "1" if s != "real" else "0", s, "no")
            for s in ("real", "fakeA", "fakeB") for i in range(5)]
    _write_manifest(manifest, rows)

    a = pick_subset(str(manifest), n_per_source=2)
    b = pick_subset(str(manifest), n_per_source=2)
    assert [r["path"] for r in a] == [r["path"] for r in b]          # deterministic
    by_source = {}
    for r in a:
        by_source.setdefault(r["source"], []).append(r["path"])
    assert all(len(v) == 2 for v in by_source.values())              # balanced per source
    # sorted by path within each source
    assert by_source["real"] == sorted(by_source["real"])


def test_pick_subset_caps_at_available_rows(tmp_path):
    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [("/v/only_one.mp4", "1", "rare", "no")])
    out = pick_subset(str(manifest), n_per_source=10)
    assert len(out) == 1


def test_robustness_report_accuracy_matches_hand_count(tmp_path):
    """2 real (both scored REAL, both below threshold) + 2 fake (1 caught, 1 missed) at one setting."""
    from eval.make_video_robustness_report import main
    import sys

    pred = tmp_path / "pred.csv"
    with open(pred, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "source", "family", "level", "video_fake"])
        w.writerow(["/v/r1.mp4", "0", "real", "clean", "0", "0.1"])
        w.writerow(["/v/r2.mp4", "0", "real", "clean", "0", "0.2"])
        w.writerow(["/v/f1.mp4", "1", "fake", "clean", "0", "0.9"])   # caught (>= 0.525)
        w.writerow(["/v/f2.mp4", "1", "fake", "clean", "0", "0.1"])   # missed

    out = tmp_path / "report.md"
    argv = sys.argv
    sys.argv = ["x", "--pred", str(pred), "--out", str(out)]
    try:
        main()
    finally:
        sys.argv = argv

    text = out.read_text(encoding="utf-8")
    assert "| clean | 0 | 4 | 75.0%" in text   # 3 of 4 correct: both reals + 1 of 2 fakes
