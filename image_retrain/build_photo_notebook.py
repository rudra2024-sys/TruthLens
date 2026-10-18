"""Builds image_retrain/kaggle_photo_r1/TruthLens_Photo_Swap_r1.ipynb + kernel-metadata.json from photo_swap_r1.py ("# %%" cells)
(no swap generation any more: the photo data is downloaded from Hugging Face).
usage: python build_photo_notebook.py"""
import json, os, re
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_photo_r1")
src = open(f"{HERE}/photo_swap_r1.py").read()
parts = re.split(r"^# %%(.*)$", src, flags=re.M)
cells = []
for head, body in zip(parts[1::2], parts[2::2]):
    body = body.strip("\n")
    if head.strip() == "[markdown]":
        text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
        cells.append(("markdown", text))
    else:
        cells.append(("code", (f"# {head.strip()}\n" if head.strip() else "") + body))
out = [(kind, text) for kind, text in cells]
nb = {"cells": [{"cell_type": k, "metadata": {}, "source": t.splitlines(True),
                 **({"outputs": [], "execution_count": None} if k == "code" else {})} for k, t in out],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for k, t in out:
    if k == "code":
        compile(t, "cell", "exec")        # syntax check every cell
path = f"{HERE}/TruthLens_Photo_Swap_r1.ipynb"
json.dump(nb, open(path, "w"), indent=1)

# Colab variant: same cells behind a setup cell (Drive for resume, Kaggle token for the input downloads, GPU check)
PRELUDE = """# 0. Colab setup: Runtime > Change runtime type > GPU first. Needs a Kaggle API token (Kaggle > Settings > API > Create
# New Token) stored as a Colab secret named KAGGLE_API_TOKEN (key icon on the left); it is only read, never printed or saved.
# Optional second secret HF_TOKEN (huggingface.co read token, after accepting the terms of Scam-AI/RWFS) enables the final
# real-app test; without it that test is skipped.
import os, subprocess, sys, getpass
from google.colab import drive
drive.mount('/content/drive')
os.makedirs('/content/drive/MyDrive/TruthLens_photo_r1', exist_ok=True)      # resume state + results live here
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'kaggle'], check=True)
if not os.environ.get('KAGGLE_API_TOKEN') and not os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
    try:
        from google.colab import userdata
        os.environ['KAGGLE_API_TOKEN'] = userdata.get('KAGGLE_API_TOKEN')
    except Exception:
        os.environ['KAGGLE_API_TOKEN'] = getpass.getpass('Kaggle API token: ')
r = subprocess.run(['kaggle', 'kernels', 'list', '--mine', '--page-size', '1'], capture_output=True, text=True)
assert r.returncode == 0, 'Kaggle login failed: ' + (r.stdout + r.stderr)[-300:]
print('Kaggle login OK')
import torch
assert torch.cuda.is_available(), 'No GPU: Runtime > Change runtime type > T4 GPU'
print(torch.cuda.get_device_name(0))
"""
cnb = json.loads(json.dumps(nb))
cnb["cells"].insert(1, {"cell_type": "code", "metadata": {}, "source": PRELUDE.splitlines(True), "outputs": [], "execution_count": None})
cnb["metadata"]["accelerator"] = "GPU"
cnb["metadata"]["colab"] = {"gpuType": "T4", "provenance": []}
compile(PRELUDE.replace("from google.colab import drive\ndrive.mount('/content/drive')", "pass"), "prelude", "exec")
json.dump(cnb, open(f"{HERE}/TruthLens_Photo_Swap_r1_colab.ipynb", "w"), indent=1)
json.dump({"id": "maninijaiswal/truthlens-photo-swap-r1", "title": "truthlens-photo-swap-r1",
           "code_file": "TruthLens_Photo_Swap_r1.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True,
           "enable_gpu": True, "enable_internet": True,
           "dataset_sources": ["maninijaiswal/truthlens-video-ft-inputs"],
           "competition_sources": [], "kernel_sources": ["maninijaiswal/truthlens-video-finetune-v2"]},
          open(f"{HERE}/kernel-metadata.json", "w"), indent=2)
print("built", path, len(out), "cells")
