"""The Colab cache fix (eval/video_head_experiments/v5_cache_fix.py), tested locally with stub extractors."""

from pathlib import Path

import numpy as np
import pytest
import torch

FIX = Path(__file__).resolve().parents[1] / "eval" / "video_head_experiments" / "v5_cache_fix.py"
D = 6


def make_env(drive_dir, calls):
    """A namespace like the notebook's: stub extractors that record which (path, view) were 'extracted'."""
    def load_views_np(kind, path, views):
        calls.extend((path, v) for v in views)
        return [(v, np.full((16, 3, 2, 2), float(hash((path, v)) % 97), dtype=np.float32)) for v in views]

    def features_from_batch(batch):
        return torch.from_numpy(batch[:, 0, 0, :1].repeat(D, axis=1)).half()      # (16, D), value encodes the item

    ns = {"DRIVE_DIR": drive_dir, "NUM_WORKERS": 2, "SAVE_EVERY_SECONDS": 0, "load_views_np": load_views_np,
          "features_from_batch": features_from_batch, "source_of": lambda kind, path: 0}
    exec(compile(FIX.read_text(encoding="utf-8"), str(FIX), "exec"), ns)
    return ns


def items(n, views=(0,), label_fn=lambda i: i % 2):
    return [("video", f"/vid/{i}.mp4", label_fn(i), v) for i in range(n) for v in views]


def test_extracts_everything_once_and_returns_the_requested_order(tmp_path):
    calls = []
    ns = make_env(tmp_path, calls)
    out = ns["build_feature_cache"](items(6, views=(0, 1)), tmp_path / "train_features_v5_shard0of1.pt")
    assert len(calls) == 12 and len(out["keys"]) == 12
    assert out["keys"] == [f"/vid/{i}.mp4#{v}" for i in range(6) for v in (0, 1)]
    assert out["features"].shape == (12, 16, D)
    assert (tmp_path / "train_features_v5_shard0of1.pt").is_file()


def test_resume_extracts_nothing_more(tmp_path):
    calls = []
    ns = make_env(tmp_path, calls)
    path = tmp_path / "train_features_v5_shard0of1.pt"
    ns["build_feature_cache"](items(5), path)
    calls.clear()
    out = ns["build_feature_cache"](items(5), path)
    assert calls == [] and len(out["keys"]) == 5


def test_stale_entries_from_another_split_are_never_returned(tmp_path):
    """The bug: the old code returned every entry in the file, including items from an older train/val split."""
    calls = []
    ns = make_env(tmp_path, calls)
    path = tmp_path / "train_features_v5_shard0of1.pt"
    ns["build_feature_cache"](items(10), path)                       # an "older split" trained on items 0..9
    out = ns["build_feature_cache"](items(4), path)                  # the current split only wants items 0..3
    assert len(out["keys"]) == 4 and set(out["keys"]) == {f"/vid/{i}.mp4#0" for i in range(4)}
    assert out["features"].shape[0] == 4


def test_a_val_item_cached_in_an_old_train_file_is_reused_not_recomputed_and_not_in_train(tmp_path):
    calls = []
    ns = make_env(tmp_path, calls)
    ns["build_feature_cache"](items(10), tmp_path / "train_features_v5_shard0of1.pt")    # old run: items 0..9 in TRAIN file
    calls.clear()
    # new split: items 0..5 train, items 6..9 validation
    train = ns["build_feature_cache"](items(6), tmp_path / "train_features_v5_shard0of1.pt")
    val = ns["build_feature_cache"](items(10)[6:], tmp_path / "val_features_v5_shard0of1.pt")
    assert calls == [], "everything was already computed; nothing should be extracted again"
    assert not (set(train["keys"]) & set(val["keys"])), "no item may be in both train and validation"
    assert len(train["keys"]) == 6 and len(val["keys"]) == 4


def test_reuses_partial_caches_from_other_accounts(tmp_path):
    calls = []
    ns = make_env(tmp_path, calls)
    ns["build_feature_cache"](items(4), tmp_path / "train_features_v5_shard0of1.pt")           # account A
    other = tmp_path / "train_features_v5_from_account_b.pt"
    ns["build_feature_cache"](items(8)[4:], other)                                             # account B did 4..7
    calls.clear()
    out = ns["build_feature_cache"](items(8), tmp_path / "train_features_v5_shard0of1.pt")
    assert calls == [] and len(out["keys"]) == 8


def test_only_missing_items_are_extracted(tmp_path):
    calls = []
    ns = make_env(tmp_path, calls)
    path = tmp_path / "train_features_v5_shard0of1.pt"
    ns["build_feature_cache"](items(5), path)
    calls.clear()
    ns["build_feature_cache"](items(8), path)
    assert sorted(calls) == sorted((f"/vid/{i}.mp4", 0) for i in range(5, 8))


def test_features_belong_to_the_right_item(tmp_path):
    calls = []
    ns = make_env(tmp_path, calls)
    out = ns["build_feature_cache"](items(5), tmp_path / "train_features_v5_shard0of1.pt")
    again = ns["build_feature_cache"](list(reversed(items(5))), tmp_path / "train_features_v5_shard0of1.pt")
    by_key = {k: out["features"][i] for i, k in enumerate(out["keys"])}
    for i, k in enumerate(again["keys"]):
        assert torch.equal(again["features"][i], by_key[k])          # reordering the request keeps item<->feature pairing
    assert again["labels"].tolist() == [int(k.split("/")[2].split(".")[0]) % 2 for k in again["keys"]]


def test_duplicate_requests_are_returned_once(tmp_path):
    ns = make_env(tmp_path, [])
    dup = items(3) + items(3)
    assert len(ns["build_feature_cache"](dup, tmp_path / "train_features_v5_shard0of1.pt")["keys"]) == 3


def test_atomic_save_keeps_the_previous_file_and_survives_a_crash_mid_run(tmp_path):
    calls = []
    ns = make_env(tmp_path, calls)
    path = tmp_path / "train_features_v5_shard0of1.pt"
    ns["build_feature_cache"](items(3), path)
    ns["build_feature_cache"](items(6), path)
    assert path.with_name(path.name + ".prev").is_file()

    boom = {"n": 0}
    real = ns["load_views_np"]

    def flaky(kind, p, views):
        boom["n"] += 1
        if boom["n"] > 2:
            raise KeyboardInterrupt()
        return real(kind, p, views)
    ns["load_views_np"] = flaky
    with pytest.raises(BaseException):
        ns["build_feature_cache"](items(12), path)
    ns["load_views_np"] = real
    calls.clear()
    out = ns["build_feature_cache"](items(12), path)
    assert len(out["keys"]) == 12 and len(calls) <= 12 - 6           # what was saved before the crash is not redone


def test_cached_subset_returns_only_already_extracted_items_in_order(tmp_path, capsys):
    ns = make_env(tmp_path, [])
    ns["build_feature_cache"](items(5), tmp_path / "train_features_v5_shard0of1.pt")   # items 0..4 cached
    capsys.readouterr()
    out = ns["cached_subset"](items(10))                                              # ask about 0..9
    assert [it[1] for it in out] == [f"/vid/{i}.mp4" for i in range(5)]
    text = capsys.readouterr().out
    assert "5 of 10 available now" in text


def test_cached_subset_feeds_build_feature_cache_with_nothing_left_to_extract(tmp_path):
    calls = []
    ns = make_env(tmp_path, calls)
    path = tmp_path / "train_features_v5_shard0of1.pt"
    ns["build_feature_cache"](items(5), path)
    calls.clear()
    available = ns["cached_subset"](items(10))
    out = ns["build_feature_cache"](available, path)
    assert calls == [] and len(out["keys"]) == 5


def test_build_feature_cache_returns_empty_tensor_instead_of_crashing(tmp_path):
    """Regression: torch.stack([]) used to raise when a split has nothing cached yet (e.g. val_items
    filtered down to zero because extraction hadn't reached the val portion of the queue)."""
    ns = make_env(tmp_path, [])
    ns["build_feature_cache"](items(5), tmp_path / "train_features_v5_shard0of1.pt")  # something cached, elsewhere
    out = ns["build_feature_cache"]([], tmp_path / "val_features_v5_shard0of1.pt")
    assert out["features"].shape == (0, 16, D)
    assert out["labels"].tolist() == [] and out["keys"] == []


def test_cached_identity_split_only_uses_fully_cached_identities(tmp_path):
    ns = make_env(tmp_path, [])
    # identity "a" -> 2 samples both cached; "b" -> 1 sample cached; "c" -> 1 sample NOT cached
    groups = {
        "a": [("video", "/vid/a1.mp4", 0), ("video", "/vid/a2.mp4", 1)],
        "b": [("video", "/vid/b1.mp4", 0)],
        "c": [("video", "/vid/c1.mp4", 1)],
    }
    cached = [("video", "/vid/a1.mp4", 0, 0), ("video", "/vid/a2.mp4", 1, 0), ("video", "/vid/b1.mp4", 0, 0)]
    ns["build_feature_cache"](cached, tmp_path / "train_features_v5_shard0of1.pt")
    train_set, val_set = ns["cached_identity_split"](groups, val_frac=0.5, seed=0)
    all_ids = {p for _, p, _ in train_set + val_set}
    assert "/vid/c1.mp4" not in all_ids                              # identity "c" was never cached, excluded
    assert all_ids == {"/vid/a1.mp4", "/vid/a2.mp4", "/vid/b1.mp4"}   # only fully-cached identities used
    assert not (set(s[1] for s in train_set) & set(s[1] for s in val_set))  # no path in both splits


def test_cached_identity_split_keeps_an_identitys_samples_together(tmp_path):
    ns = make_env(tmp_path, [])
    groups = {str(i): [("video", "/vid/%d.mp4" % i, i % 2)] for i in range(10)}
    cached = [("video", "/vid/%d.mp4" % i, i % 2, 0) for i in range(10)]
    ns["build_feature_cache"](cached, tmp_path / "train_features_v5_shard0of1.pt")
    train_set, val_set = ns["cached_identity_split"](groups, val_frac=0.3, seed=1)
    assert len(val_set) >= 1 and len(train_set) >= 1
    assert len(train_set) + len(val_set) == 10


def test_cache_report_counts_and_flags_leakage(tmp_path, capsys):
    ns = make_env(tmp_path, [])
    ns["build_feature_cache"](items(10), tmp_path / "train_features_v5_shard0of1.pt")
    ns["cache_report"](items(6), items(10)[6:])
    text = capsys.readouterr().out
    assert "match this run's TRAIN list: 6" in text and "match this run's VAL list  : 4" in text
    assert "VALIDATION items in this run: 4" in text


def test_split_fingerprint_is_stable(tmp_path, capsys):
    ns = make_env(tmp_path, [])
    capsys.readouterr()                                                          # discard the "fix installed" banner
    ns["train_ids"], ns["val_ids"] = {"b", "a", "c"}, {"z", "y"}
    ns["split_fingerprint"]()
    first = capsys.readouterr().out
    ns["train_ids"], ns["val_ids"] = {"c", "a", "b"}, {"y", "z"}                # same sets, different insertion order
    ns["split_fingerprint"]()
    assert capsys.readouterr().out == first
