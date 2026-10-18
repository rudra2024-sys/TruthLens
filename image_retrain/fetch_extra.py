"""Round-1b extra REAL-ONLY sources for the local run: CelebA (178x218 aligned faces, low-res like UTKFace) and COCO
val2017 (everyday photos). Same range-request mechanism as fetch_local_data.py; writes work/local_data/<group>/** and
local_manifest_extra.csv. CelebA split follows its own list_eval_partition (train/val/test by index); COCO is split by
image id hash. usage: python fetch_extra.py <urls.env> <index_dir>"""
import csv, hashlib, io, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor
import fetch_local_data_lib as L  # noqa  (tiny shared helpers, see below)
URLS = dict(l.rstrip("\n").split("=", 1) for l in open(sys.argv[1])); L.URLS.update(URLS)
IDX = sys.argv[2]; HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "work", "local_data")
idx = lambda n: [r["name"] for r in csv.DictReader(open(os.path.join(IDX, n + ".csv")))]
rnd = random.Random(99); JOBS, ROWS = [], []


def want(ds, member, group, split):
    rel = f"{group}/{split}/real_x{len(JOBS):06d}.jpg"; JOBS.append((ds, member, rel))
    ROWS.append({"path": rel, "fake": 0, "group": group, "sub": "real", "split": split, "fmt": member.rsplit(".", 1)[-1].lower()})


cel = sorted(n for n in idx("celeba") if n.endswith(".jpg"))
def part(n):
    i = int(os.path.basename(n).split(".")[0]); return "train" if i <= 162770 else "val" if i <= 182637 else "test"
by = {"train": [], "val": [], "test": []}
for n in cel: by[part(n)].append(n)
for sp, k in (("train", 3000), ("val", 300), ("test", 300)):
    rnd.shuffle(by[sp]); [want("celeba", n, "celeba", sp) for n in by[sp][:k]]
coco = sorted(n for n in idx("coco") if n.endswith(".jpg")); rnd.shuffle(coco)
for i, n in enumerate(coco[:3700]): want("coco", n, "coco", "train" if i < 3200 else "val" if i < 3400 else "test")
print(len(JOBS), "to fetch", flush=True)


def run(j):
    ds, member, rel = j; dest = os.path.join(OUT, rel); os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest): return True
    try: L.save(ds, member, dest); return True
    except Exception as e: print("ERR", member, repr(e)); return False


t0 = time.time()
with ThreadPoolExecutor(14) as ex: ok = list(ex.map(run, JOBS))
rows = [r for r, o in zip(ROWS, ok) if o]
with open(os.path.join(HERE, "local_manifest_extra.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, ["path", "fake", "group", "sub", "split", "fmt"]); w.writeheader(); w.writerows(rows)
print("wrote", len(rows), f"{time.time() - t0:.0f}s")
