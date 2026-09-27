# Fix for the v5 Colab notebook's feature cache. Run it with exec() AFTER the notebook cells that define
# load_views_np / features_from_batch / source_of / NUM_WORKERS / SAVE_EVERY_SECONDS / DRIVE_DIR (i.e. after the
# "import zlib" extraction-definitions cell), and BEFORE the cell that calls build_feature_cache:
#
#     exec(open("/content/v5_cache_fix.py").read())
#     cache_report(train_items, val_items)          # optional: shows how much of the cache matches this run
#
# What it fixes
#   1. The old build_feature_cache returned EVERY entry in the cache file, including entries left over from an older
#      train/val split. Some of those can be current VALIDATION items, which inflates the validation scores (leakage).
#      This version returns only the items that were asked for.
#   2. It reuses features from ALL cache files in the Drive folder (train, val, other accounts' partial caches, any split),
#      so nothing that was already computed is computed again. A feature depends only on (video path, view), never on
#      the split.
#   3. Saves are atomic (temp file, previous save kept as .prev) and happen on interrupt.
#
# Written without f-strings/braces on purpose: pasting code into Colab has mangled those before.
import os
import shutil
import time
import zlib
from pathlib import Path

import torch


def _key(p, v):
    return str(p) + "#" + str(v)


def _atomic_torch_save(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    prev = path.with_name(path.name + ".prev")
    torch.save(obj, tmp)
    if path.exists():
        os.replace(path, prev)
    os.replace(tmp, path)


def _read_cache(path):
    path = Path(path)
    for cand in (path, path.with_name(path.name + ".prev")):
        if cand.exists():
            try:
                return torch.load(cand)
            except Exception as e:
                print("could not read", cand.name, e, "- trying previous save")
    return None


def _cache_files():
    d = Path(DRIVE_DIR)
    names = set()
    for pattern in ("train_features_v5*.pt", "val_features_v5*.pt"):
        for p in d.glob(pattern):
            names.add(str(p))
    return sorted(names)


def _load_pool():
    """key -> (features (16, D), label, source) from every cache file, whatever split produced it."""
    pool = {}
    for path in _cache_files():
        c = _read_cache(path)
        if c is None:
            continue
        feats = c["features"]
        labels = c["labels"].tolist()
        sources = c["sources"].tolist()
        for i, k in enumerate(list(c["keys"])):
            if k not in pool:
                pool[k] = (feats[i], labels[i], sources[i])
    return pool


def cache_report(train_items, val_items):
    pool = _load_pool()
    tk = set(_key(p, v) for k, p, l, v in train_items)
    vk = set(_key(p, v) for k, p, l, v in val_items)
    in_train = len(set(pool) & tk)
    in_val = len(set(pool) & vk)
    print("cache files:", [os.path.basename(p) for p in _cache_files()])
    print("cached items (all files, unique):", len(pool))
    print("  match this run's TRAIN list:", in_train, "of", len(tk))
    print("  match this run's VAL list  :", in_val, "of", len(vk))
    print("  match neither (from another split/config):", len(pool) - in_train - in_val)
    leak = 0
    for path in _cache_files():
        if os.path.basename(path).startswith("train_features"):
            c = _read_cache(path)
            if c is not None:
                leak += sum(1 for k in c["keys"] if k in vk)
    print("  items in a TRAIN cache file that are VALIDATION items in this run:", leak,
          "(the old code would have trained on these)")


def split_fingerprint():
    """Same numbers on every account <=> same train/val split. Compare them across accounts."""
    tr = sorted(train_ids)
    va = sorted(val_ids)
    print("train identities:", len(tr), "| val identities:", len(va))
    print("split fingerprint:", zlib.crc32("|".join(va).encode()), zlib.crc32("|".join(tr).encode()))


def build_feature_cache(items, cache_path, log_every=100):
    from concurrent.futures import ThreadPoolExecutor
    cache_path = Path(cache_path)

    pool = _load_pool()                                   # everything already computed, by any run
    own = _read_cache(cache_path)                         # this file's own entries, re-saved with any new ones
    if own is not None:
        own_features = list(own["features"])
        own_labels = own["labels"].tolist()
        own_keys = list(own["keys"])
        own_sources = own["sources"].tolist()
    else:
        own_features = []
        own_labels = []
        own_keys = []
        own_sources = []
    print("reusable cached items across all cache files:", len(pool))

    requested = []
    seen = set()
    for k, p, l, v in items:
        key = _key(p, v)
        if key not in seen:
            seen.add(key)
            requested.append((k, p, l, v))

    jobs = {}
    for k, p, l, v in requested:
        if _key(p, v) not in pool:
            jobs.setdefault((k, p, l), []).append(v)
    jobs = [(k, p, l, vs) for (k, p, l), vs in jobs.items()]
    n_items = sum(len(j[3]) for j in jobs)
    print(len(requested), "items requested;", len(requested) - n_items, "already cached;",
          len(jobs), "videos /", n_items, "items to extract")

    def _save():
        if not own_features:
            return
        data = {}
        data["features"] = torch.stack(own_features)
        data["labels"] = torch.tensor(own_labels, dtype=torch.float32)
        data["keys"] = own_keys
        data["sources"] = torch.tensor(own_sources, dtype=torch.long)
        _atomic_torch_save(data, cache_path)

    def _load(job):
        k, p, l, vs = job
        try:
            return job, load_views_np(k, p, vs), None
        except Exception as e:
            return job, None, e

    t0 = time.time()
    last_save = t0
    done_items = 0
    CHUNK = 4 * NUM_WORKERS
    try:
        with ThreadPoolExecutor(NUM_WORKERS) as executor:
            for c0 in range(0, len(jobs), CHUNK):
                for job, res, err in executor.map(_load, jobs[c0:c0 + CHUNK]):
                    k, p, l, vs = job
                    done_items += len(vs)
                    if err is not None:
                        print("skip", p, err)
                        continue
                    for v, batch in res:
                        feats = features_from_batch(batch)
                        src = source_of(k, p)
                        key = _key(p, v)
                        own_features.append(feats)
                        own_labels.append(l)
                        own_keys.append(key)
                        own_sources.append(src)
                        pool[key] = (feats, l, src)
                if (c0 // CHUNK) % max(1, log_every // CHUNK) == 0 and done_items:
                    el = time.time() - t0
                    eta_h = el / done_items * (n_items - done_items) / 3600
                    print("%d/%d items  elapsed=%dmin  eta=%.2fh" % (done_items, n_items, el / 60, eta_h))
                if time.time() - last_save > SAVE_EVERY_SECONDS:
                    _save()
                    last_save = time.time()
                    print("  [checkpoint saved to Drive:", len(own_features), "items in", cache_path.name + "]")
    except BaseException:
        print("stopping early - saving progress to Drive...")
        _save()
        raise
    _save()

    # Return ONLY what was asked for, in the requested order (never leftovers from another split).
    feats_out = []
    labels_out = []
    keys_out = []
    sources_out = []
    missing = 0
    for k, p, l, v in requested:
        key = _key(p, v)
        if key not in pool:
            missing += 1                                     # extraction of this video failed (already printed as "skip")
            continue
        f, lab, src = pool[key]
        feats_out.append(f)
        labels_out.append(l)
        keys_out.append(key)
        sources_out.append(source_of(k, p))
    print("returning", len(keys_out), "items;", missing, "could not be extracted")
    out = {}
    out["features"] = torch.stack(feats_out)
    out["labels"] = torch.tensor(labels_out, dtype=torch.float32)
    out["keys"] = keys_out
    out["sources"] = torch.tensor(sources_out, dtype=torch.long)
    return out


print("v5 cache fix installed: build_feature_cache now reuses every cache file and returns only requested items")
