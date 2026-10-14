"""Round-3 manifest: round 2's RTFS and FakeAVCeleb rows (unchanged splits), Celeb-DF re-split by person, plus FaceForensics++ c23 and DFDC.

Leakage rules
- FaceForensics++: every identity number in the local final-test videos (work/ff_test, both numbers of a fake) is
  test-only; the identities of 60 further fake pairs are val-only. A fake goes to train only if BOTH of its identities are train
  identities, so no face seen in val/test is ever trained on. All five methods (Deepfakes, Face2Face, FaceSwap,
  NeuralTextures, FaceShifter) are sampled equally. DeepFakeDetection is skipped (its real videos aren't in this copy,
  and a fake-only source would teach "this dataset's look = fake").
- DFDC (pranay22077/dfdc-10, parts 0-9): parts 0-7 train, part 8 val, part 9 test. Each fake is chosen from a
  different original, and only when that original is a sampled real of the same split, so the notebook can drop
  audio-only fakes (video identical to the original) by comparing crops.
usage: python build_manifest_r3.py   (writes manifest_r3.csv next to this file + work/dfdc_test_list.csv)"""
import csv, glob, json, os, random, re

HERE = os.path.dirname(os.path.abspath(__file__)); IDX = os.path.join(HERE, "work", "dataset_index")
rnd = random.Random(3)
r2 = list(csv.DictReader(open(os.path.join(HERE, "manifest_r2.csv"))))
FIELDS = list(r2[0])
rows = [r for r in r2 if r["dataset"] != "celebdf"]   # RTFS / FakeAVCeleb splits are kept as they were

# ---- Celeb-DF, re-split by PERSON (round 2's split let 77 identities and 39 files cross train/val) ----
# Every id in a file name counts (a fake idA_idB shows both people); a fake is kept only if all its ids share a split.
# YouTube-real files carry no identity and are split by file. Crops already extracted in round 2 are reused by key.
old_key = {}
for r in r2:
    if r["dataset"] == "celebdf":
        old_key.setdefault(r["path"], r["key"])
cdf = [l.strip() for l in open(os.path.join(IDX, "celebdf", "all_files.txt")) if l.strip().endswith(".mp4")]
cids = sorted({re.findall(r"id(\d+)", f)[0] for f in cdf if f.startswith("Celeb-real/")}, key=int)
crnd = random.Random(31); crnd.shuffle(cids)
cpart = {**{i: "test" for i in cids[:10]}, **{i: "val" for i in cids[10:18]}, **{i: "train" for i in cids[18:]}}
yt = sorted(f for f in cdf if f.startswith("YouTube-real/")); crnd.shuffle(yt)
ypart = {f: "test" if k < 50 else "val" if k < 90 else "train" for k, f in enumerate(yt)}
cdf_by = {}
for f in cdf:
    if f.startswith("YouTube-real/"):
        s, y = ypart[f], 0
    else:
        ss = {cpart[i] for i in re.findall(r"id(\d+)", f)}
        if len(ss) != 1:
            continue
        s, y = ss.pop(), int(f.startswith("Celeb-synthesis/"))
    cdf_by.setdefault((s, y), []).append(f)
CDF_N = {"train": 600, "val": 110, "test": 30}
celebdf_test = []
for s, n in CDF_N.items():
    for y in (0, 1):
        for f in crnd.sample(sorted(cdf_by[(s, y)]), min(n, len(cdf_by[(s, y)]))):
            if s == "test":
                celebdf_test.append(dict(label=y, file=f))
                continue
            key = old_key.get(f, f"celebdf/r3_{s}_{f.replace('/', '_')[:-4]}")
            rows.append(dict(key=key, dataset="celebdf", split=s, group=f, label=y,
                             kind="fake" if y else "real", path=f))

# ---- FaceForensics++ ----
METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter")
FF_TRAIN_PER_METHOD, FF_VAL_PER_METHOD = 160, 20
meta = [r["File Path"] for r in csv.DictReader(open(os.path.join(IDX, "ffpp", "FF++_Metadata.csv")))]
frames = {r["File Path"]: int(r["Frame Count"]) for r in csv.DictReader(open(os.path.join(IDX, "ffpp", "FF++_Metadata.csv")))}
test_ids = set()
for f in glob.glob(os.path.join(HERE, "work", "ff_test", "*", "*.mp4")):
    test_ids.update(re.findall(r"\d{3}", os.path.basename(f)))
pairs = sorted({tuple(os.path.basename(p)[:-4].split("_")) for p in meta if p.startswith("Deepfakes/")})   # same pairs in every method
free = [pr for pr in pairs if not set(pr) & test_ids]
val_ids = {i for pr in rnd.sample(free, 60) for i in pr}
train_ids = {f"{i:03d}" for i in range(1000)} - test_ids - val_ids
split_of = lambda ids: "test" if ids & test_ids else "val" if ids <= val_ids else "train" if ids <= train_ids else None

for p in meta:
    if p.startswith("original/") and frames[p] >= 16:
        i = os.path.basename(p)[:3]; s = split_of({i})
        if s in ("train", "val"):
            rows.append(dict(key=f"ffpp/{s}_original_{i}", dataset="ffpp", split=s, group=i, label=0, kind="original", path=p))
for m in METHODS:
    by_split = {"train": [], "val": []}
    for p in meta:
        if p.startswith(m + "/") and frames[p] >= 16:
            ids = set(os.path.basename(p)[:-4].split("_")); s = split_of(ids)
            if s in by_split:
                by_split[s].append(p)
    for s, n in (("train", FF_TRAIN_PER_METHOD), ("val", FF_VAL_PER_METHOD)):
        for p in rnd.sample(sorted(by_split[s]), n):
            a, b = os.path.basename(p)[:-4].split("_")
            rows.append(dict(key=f"ffpp/{s}_{m}_{a}_{b}", dataset="ffpp", split=s, group=f"{a}_{b}", label=1, kind=m, path=p))

# ---- DFDC ----
DFDC_REAL = {"train": 800, "val": 140}
dfdc_test = []
for part in range(10):
    md = json.load(open(os.path.join(IDX, "dfdc", f"part{part}", "metadata.json")))
    split = "test" if part == 9 else "val" if part == 8 else "train"
    d = f"dfdc_train_part_{part:02d}/dfdc_train_part_{part}"
    reals = sorted(k for k, v in md.items() if v["label"] == "REAL")
    fakes_of = {}
    for k, v in sorted(md.items()):
        if v["label"] == "FAKE":
            fakes_of.setdefault(v["original"], []).append(k)
    for o in fakes_of.values():
        rnd.shuffle(o)
    if split == "test":
        sel = rnd.sample([r for r in reals if r in fakes_of], 40)
        for r in sel[:20]:
            dfdc_test.append(dict(label=0, file=f"{d}/{r}", original=""))
        for r in sel[20:]:
            dfdc_test.append(dict(label=1, file=f"{d}/{fakes_of[r][0]}", original=f"{d}/{r}"))
        continue
    for r in reals:
        rows.append(dict(key=f"dfdc/{split}_real_{part}_{r[:-4]}", dataset="dfdc", split=split, group=r[:-4], label=0,
                         kind="real", path=f"{d}/{r}"))
        if r in fakes_of:
            f = fakes_of[r][0]
            rows.append(dict(key=f"dfdc/{split}_fake_{part}_{f[:-4]}", dataset="dfdc", split=split, group=r[:-4], label=1,
                             kind=f"fake_of:{r[:-4]}", path=f"{d}/{f}"))

# cap DFDC per split by sampling whole (real, fake) groups so pairs stay together
for split, n in DFDC_REAL.items():
    groups = sorted({r["group"] for r in rows if r["dataset"] == "dfdc" and r["split"] == split})
    keep = set(rnd.sample(groups, min(n, len(groups))))
    rows = [r for r in rows if not (r["dataset"] == "dfdc" and r["split"] == split and r["group"] not in keep)]

with open(os.path.join(HERE, "manifest_r3.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
with open(os.path.join(HERE, "work", "celebdf_test_r3_list.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["label", "file"]); w.writeheader(); w.writerows(celebdf_test)
with open(os.path.join(HERE, "work", "dfdc_test_list.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["label", "file", "original"]); w.writeheader(); w.writerows(dfdc_test)

from collections import Counter
c = Counter((r["dataset"], r["split"], int(r["label"])) for r in rows)
for k in sorted(c):
    print(k, c[k])
print(len(rows), "rows;", len(test_ids), "FF++ test identities,", len(val_ids), "val,", len(train_ids), "train;",
      len(dfdc_test), "DFDC test videos (part 9, local only)")
