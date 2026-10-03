"""Tests for the UTKFace fairness-manifest builder (CLAUDE.md section 25 item 10)."""
import csv

from eval.build_fairness_manifest import GENDER, RACE, build_manifest


def _make_utkface_dir(tmp_path, counts):
    """counts: {(gender_code, race_code): n} -> creates that many empty, uniquely-named files per cell."""
    for (gender, race), n in counts.items():
        for i in range(n):
            (tmp_path / f"{20 + i}_{gender}_{race}_2020010100000{i:02d}.jpg.chip.jpg").write_bytes(b"")
    # a few files that should be skipped
    (tmp_path / "not_a_utkface_file.jpg").write_bytes(b"")
    (tmp_path / "25_9_0_20200101000000.jpg.chip.jpg").write_bytes(b"")  # gender code 9 is invalid
    return tmp_path


def test_build_manifest_is_balanced_across_every_group(tmp_path):
    counts = {(g, r): 10 for g in GENDER for r in RACE}
    d = _make_utkface_dir(tmp_path, counts)
    rows = build_manifest(d, n_per_group=5, seed=1)
    assert len(rows) == 5 * len(GENDER) * len(RACE)
    from collections import Counter
    by_source = Counter(r["source"] for r in rows)
    assert all(v == 5 for v in by_source.values())
    assert set(by_source) == {f"utkface_{GENDER[g]}_{RACE[r]}" for g in GENDER for r in RACE}


def test_build_manifest_labels_are_all_zero(tmp_path):
    counts = {(g, r): 5 for g in GENDER for r in RACE}
    d = _make_utkface_dir(tmp_path, counts)
    rows = build_manifest(d, n_per_group=3, seed=1)
    assert all(r["label"] == 0 for r in rows)


def test_build_manifest_is_reproducible_across_two_independent_runs(tmp_path):
    counts = {(g, r): 20 for g in GENDER for r in RACE}
    d = _make_utkface_dir(tmp_path, counts)
    a = build_manifest(d, n_per_group=10, seed=42)
    b = build_manifest(d, n_per_group=10, seed=42)
    assert sorted(r["path"] for r in a) == sorted(r["path"] for r in b)


def test_build_manifest_warns_but_does_not_crash_on_a_short_group(tmp_path, capsys):
    counts = {(g, r): 10 for g in GENDER for r in RACE}
    counts[(0, 1)] = 2  # male/black has only 2 available, fewer than requested
    d = _make_utkface_dir(tmp_path, counts)
    rows = build_manifest(d, n_per_group=5, seed=1)
    male_black = [r for r in rows if r["source"] == "utkface_male_black"]
    assert len(male_black) == 2
    assert "only 2/5" in capsys.readouterr().out
