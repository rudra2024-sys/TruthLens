"""Image round-1 manifest: the sources that close the gaps measured on 2026-10-15 (see README.md in this folder).

  genimage   cartografia/unbiased-tiny-genimage: 7 generators x 2500 fakes (ADM, BigGAN, GLIDE, Midjourney, VQDM,
             SD 1.5, Wukong) + 5828 ImageNet reals, all JPEG (the "unbiased" re-encode). Split by GROUP = ImageNet class
             index of the fake's filename (reals are hashed by filename). ADM is held out ENTIRELY (split "test_gen") to
             measure an unseen generator; it is never trained on.
  sut        hiddenplant/sut-project: 4752 fakes (Midjourney V6 / Flux / NanoBanana, not labelled per file) + 4752 reals
             (COCO / Unsplash). Fakes are all PNG, reals are mixed PNG/JPEG, so `fmt` is recorded and training MUST
             JPEG-normalise both classes or "PNG" becomes a shortcut for fake. Split by hash of file (no group info).
  dff        ajaysonicu/deepfakefusion-399k-realfake-faces, ONLY the FF++ (5 manipulations + originals) and Celeb-DF v2
             face-swap rows from its train/val folders. Its own test folder is NOT used: filenames carry no video id, so
             frames of one video can sit on both sides of its split (leakage we cannot check). Face-swap generalisation
             is measured on external stills instead (eval_only.json). These are face CROPS, not full photos.
Rehearsal sources (the original four + portrait-app) are listed in plan.json; they are enumerated on Kaggle in step 2.
usage: python build_manifest_r1.py   (writes manifest_r1.csv, plan.json, prints counts)"""
import csv, hashlib, json, os, random, re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__)); IDX = os.path.join(HERE, "work", "dataset_index")
FIELDS = ["key", "dataset", "split", "group", "label", "kind", "path", "source", "fmt"]
rows = []


def listing(name):
    return [r["name"] for r in csv.DictReader(open(os.path.join(IDX, name + ".csv")))]


def bucket(s):  # deterministic 0-99 from a string
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % 100


def split_of(group):
    b = bucket(group)
    return "train" if b < 80 else "val" if b < 90 else "test"


def add(dataset, split, group, label, kind, path, source):
    rows.append({"key": f"{dataset}/{len(rows):06d}", "dataset": dataset, "split": split, "group": group,
                 "label": label, "kind": kind, "path": path, "source": source,
                 "fmt": path.rsplit(".", 1)[-1].lower()})


# ---- GenImage (unbiased tiny) ----
HELD_OUT_GEN = "ADM"
for n in listing("genimage"):
    d = n.split("/")[0]
    if not re.search(r"\.(jpe?g|png)$", n, re.I) or "/" not in n:
        continue
    if d == "Nature":
        add("genimage", split_of("real/" + n), n, 0, "real_imagenet", n, "Nature")
        continue
    m = re.match(r"(\d+)_", os.path.basename(n))
    group = f"{d}/{m.group(1)}" if m else n
    group_key = m.group(1) if m else n  # same class index -> same split across generators (no prompt leakage)
    if d == HELD_OUT_GEN:
        add("genimage", "test_gen", group, 1, "fake_" + d.lower(), n, d)
    else:
        add("genimage", split_of("fake/" + group_key), group, 1, "fake_" + d.lower(), n, d)

# ---- SUT (modern diffusion) ----
for n in listing("sut"):
    if not re.search(r"\.(jpe?g|png)$", n, re.I):
        continue
    fake = "/Fake/" in n
    add("sut", split_of(n), n, int(fake), "fake_modern_diffusion" if fake else "real_coco_unsplash", n,
        "SUT-fake" if fake else "SUT-real")

# ---- DeepFakeFusion: FF++ and Celeb-DF rows only, train/val folders only ----
rnd = random.Random(7)
dff = {}
for n in listing("dff"):
    p = n.split("/")
    if len(p) != 4 or p[1] not in ("train", "val"):
        continue
    m = re.match(r"\w+_(real|fake)_\d+_(baseline_(?:ffpp_\w+|cdfv2))\.jpg$", p[3])
    if m:
        dff.setdefault((p[1], m.group(1), m.group(2)), []).append(n)
CAP_CDF = {"train": 3000, "val": 400}
for (sp, lab, src), names in sorted(dff.items()):
    names = sorted(names)
    if src == "baseline_cdfv2":
        rnd.shuffle(names); names = names[:CAP_CDF[sp]]
    for n in names:
        add("dff", sp, n, int(lab == "fake"), ("fake_" if lab == "fake" else "real_") + src.replace("baseline_", ""),
            n, src.replace("baseline_", ""))

with open(os.path.join(HERE, "manifest_r1.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, FIELDS); w.writeheader(); w.writerows(rows)

json.dump({
    "rehearsal": [  # sampled in step 2 so the model does not forget what it already does well
        {"ref": "birdy654/cifake-real-and-ai-generated-synthetic-images", "note": "CIFAKE, 32px native"},
        {"ref": "alessandrasala79/ai-vs-human-generated-dataset"},
        {"ref": "xhlulu/140k-real-and-fake-faces"},
        {"ref": "ayushmandatta1/deepdetect-2025", "note": "JPEG-history shortcut, cap its weight"},
        {"ref": "portrait_app_fake_source_v1.zip", "note": "257 fakes, local only, keep in training so CLIP/ConvNeXt retain it"}],
    "eval_only_never_trained": [
        {"ref": "sanketghadge1/openfake-data-20k-img", "note": "licence unknown, keep local"},
        {"ref": "renhuang8/genimage-subset-detection", "note": "same generators as genimage, different images"},
        {"ref": "backend/test_data/blind_spot_images/chatgpt_everyday", "note": "8 GPT-4o-class images, the only such data we have"},
        {"ref": "testing/deepfake/images", "note": "3 FakeAVCeleb face-swap stills; too few, extract more from FakeAVCeleb videos"},
        {"ref": "UTKFace real faces", "note": "real-photo specificity / fairness"}],
    "held_out_generator": HELD_OUT_GEN,
}, open(os.path.join(HERE, "plan.json"), "w"), indent=2)

c = Counter((r["dataset"], r["split"], "fake" if r["label"] else "real") for r in rows)
for k in sorted(c): print(f"{c[k]:6d}  {k}")
print(len(rows), "rows;", Counter(r["fmt"] + "/" + ("fake" if r["label"] else "real") for r in rows if r["dataset"] == "sut"))
