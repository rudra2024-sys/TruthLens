"""Round-4 manifest: all of round 3 (same rows, same splits) plus three new sources aimed at real phone footage.

  dfd        Google/Jigsaw DeepFakeDetection (sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset):
             28 paid actors filmed in ordinary scenes + face swaps between them. Split by ACTOR: a fake (AA_BB__...)
             is used only if both actors are in the same split. Test actors are never attached for training.
  phone      Real selfie/liveness videos filmed by many people on their own phones/webcams (5 small datasets, all
             CC BY-NC-ND 4.0, research use). Real only. Split by PERSON.
  phoneswap  Generated in the notebook: for each phone video, its 16 production-sampled frames are written back as a
             16-frame H.264 video (real) and face-swapped onto other people of the SAME split with inswapper (fake),
             encoded identically. Without this, "filmed on a phone" would become a shortcut for "real", the mirror
             image of round 3's reason for skipping fake-only sources.
Test-split rows (held-out actors/persons) are cropped and scored on Kaggle but never trained on (split == "test").
Only the "liveness"/"real_video" files are used; replay/print attacks in the same datasets are not deepfakes and skipped.
usage: python build_manifest_r4.py   (writes manifest_r4.csv, work/r4_test_list.csv)"""
import csv, os, random, re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__)); IDX = os.path.join(HERE, "work", "dataset_index", "r4")
META = os.path.join(HERE, "work", "dataset_index", "r4", "meta")
VID = re.compile(r"\.(mp4|mov|webm|mkv|avi)$", re.I)
r3 = list(csv.DictReader(open(os.path.join(HERE, "manifest_r3.csv"))))
FIELDS = list(r3[0]) + ["source"]
rows = [{**r, "source": ""} for r in r3]


def listing(ref):
    return [l["name"] for l in csv.DictReader(open(os.path.join(IDX, ref.replace("/", "__") + ".csv")))]


def add(dataset, split, group, label, kind, path, source=""):
    key = f"{dataset}/{split}_{len([r for r in rows if r['dataset'] == dataset]):05d}_{kind}"
    rows.append({"key": key, "dataset": dataset, "split": split, "group": group, "label": label, "kind": kind,
                 "path": path, "source": source})
    return key


test_list = []

# ---- DFD, split by actor ----
dfd = [n for n in listing("sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset") if VID.search(n)]
reals = [n for n in dfd if n.startswith("DFD_original")]
fakes = [n for n in dfd if n.startswith("DFD_manip")]
actors = sorted({os.path.basename(n).split("__")[0] for n in reals})
arnd = random.Random(41); arnd.shuffle(actors)
apart = {**{a: "test" for a in actors[:5]}, **{a: "val" for a in actors[5:8]}, **{a: "train" for a in actors[8:]}}
for n in sorted(reals):
    s = apart[os.path.basename(n).split("__")[0]]
    if s == "test":
        test_list.append(("dfd", 0, n))
    add("dfd", s, os.path.basename(n).split("__")[0], 0, "real_original", n, "dfd")
fake_by = {}
for n in sorted(fakes):
    ab = os.path.basename(n).split("__")[0].split("_")
    ss = {apart.get(a) for a in ab}
    if len(ss) == 1 and None not in ss:
        fake_by.setdefault(ss.pop(), []).append(n)
frnd = random.Random(42)
for s, cap in (("train", 700), ("val", 90), ("test", 40)):
    pick = frnd.sample(fake_by.get(s, []), min(cap, len(fake_by.get(s, []))))
    for n in sorted(pick):
        if s == "test":
            test_list.append(("dfd", 1, n))
        add("dfd", s, os.path.basename(n).split("__")[0], 1, "deepfake", n, "dfd")

# ---- phone: real selfie / liveness videos, split by person ----
people = []   # (source_ref, person, path)
for n in listing("tapakah68/selfies-and-video-dataset-4-000-people"):
    if VID.search(n):
        people.append(("tapakah68/selfies-and-video-dataset-4-000-people", "selfies_" + n.split("/")[1], n))
for ref, csvname in (("trainingdatapro/asian-people-liveness-detection-video-dataset", None),
                     ("trainingdatapro/black-people-liveness-detection-video-dataset", None)):
    for n in listing(ref):
        if VID.search(n):
            people.append((ref, ref.split("/")[1][:5] + "_" + n.split("/")[1], n))
worker = {r["video_link"]: r["worker_id"] for r in csv.DictReader(open(os.path.join(META, "real_30.csv")))}
for n in listing("trainingdatapro/anti-spoofing-live"):
    if VID.search(n):
        people.append(("trainingdatapro/anti-spoofing-live", "w_" + worker.get(n.split("/", 1)[1], n), n))
wk = {r["assignment_id"]: r["worker_id"] for r in csv.DictReader(open(os.path.join(META, "file_info.csv")))}
for n in listing("trainingdatapro/full-hd-webcam-live-attacks"):
    if VID.search(n) and n.startswith("live_videos/"):
        people.append(("trainingdatapro/full-hd-webcam-live-attacks", "w_" + wk.get(os.path.splitext(os.path.basename(n))[0], n), n))
rvf = []
for n in listing("trainingdatapro/real-vs-fake-anti-spoofing-video-classification"):
    if VID.search(n) and "/real_video/" in n:
        rvf.append(n)

persons = sorted({p for _, p, _ in people}); prnd = random.Random(43); prnd.shuffle(persons)
k1, k2 = int(len(persons) * 0.2), int(len(persons) * 0.3)
ppart = {p: "test" if i < k1 else "val" if i < k2 else "train" for i, p in enumerate(persons)}
phone = [(ref, p, n, ppart[p]) for ref, p, n in people]
vids = sorted(n for n in rvf if n.startswith("train/")); prnd.shuffle(vids)   # no person ids: each video its own person
for i, n in enumerate(vids):
    phone.append(("trainingdatapro/real-vs-fake-anti-spoofing-video-classification", "rvf_" + n, n, "val" if i < 10 else "train"))
for n in sorted(n for n in rvf if n.startswith("test/")):
    phone.append(("trainingdatapro/real-vs-fake-anti-spoofing-video-classification", "rvf_" + n, n, "test"))

for ref, p, n, s in phone:
    if s == "test":
        test_list.append(("phone", 0, f"{ref}:{n}"))
    add("phone", s, p, 0, "real_phone", f"{ref}:{n}", ref)
    # phoneswap rows for every split (test ones are generated on Kaggle too and brought back for the local test)
    base = re.sub(r"[^A-Za-z0-9]+", "_", f"{p}_{os.path.basename(n)}")
    add("phoneswap", s, p, 0, "real_reencoded", f"{s}/{base}__real.mp4", f"{ref}:{n}")
    for j in range(2):
        add("phoneswap", s, p, 1, f"inswapper_d{j}", f"{s}/{base}__fake{j}.mp4", f"{ref}:{n}")

with open(os.path.join(HERE, "manifest_r4.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, FIELDS); w.writeheader(); w.writerows(rows)
with open(os.path.join(HERE, "work", "r4_test_list.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["dataset", "label", "file"]); w.writerows(test_list)
print(len(rows), "rows")
for k, v in sorted(Counter((r["dataset"], r["split"], r["label"]) for r in rows if r["dataset"] in ("dfd", "phone", "phoneswap")).items()):
    print(" ", k, v)
print("local test:", sorted(Counter((d, y) for d, y, _ in test_list).items()))
# leakage self-check: no group in more than one split, per new dataset
for ds in ("dfd", "phone", "phoneswap"):
    g = {}
    for r in rows:
        if r["dataset"] == ds:
            g.setdefault(r["group"], set()).add(r["split"])
    assert all(len(v) == 1 for v in g.values()), ds
fk = [r for r in rows if r["dataset"] == "dfd" and r["label"] == "1"]
assert all(apart[a] == r["split"] for r in fk for a in r["group"].split("_")), "dfd fake crosses actor split"
print("leakage checks passed")
