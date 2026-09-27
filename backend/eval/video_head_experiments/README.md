# Video CNN+BiGRU head experiments (Colab, not deployed)

Scripts kept from the Sep 2026 experiments to improve the video model with a temporal head on the frozen
EfficientNet-B0 features. Nothing here is imported by the app. Paths inside are machine-specific (Windows, Downloads).

- `build_notebook_v5.py` — generates `TruthLens_Video_CNN_RNN_Training_v5.ipynb` (v4 recipe: augmented feature views,
  source-balanced sampling, real-world + held-out selection; plus the RTFS dataset, Google-Drive checkpointing,
  multi-account sharding). Writes the notebook to `C:\Users\admin\Downloads\`.
- `eval_cnn_rnn.py` / `eval_v4.py` — compare a downloaded head checkpoint against the deployed CNN on the 10 local
  ground-truth videos (edit the checkpoint path at the top).
- `eval_pooling.py` — mean vs max vs top-k frame pooling of the deployed CNN (no pooling variant helped).
- `inspect_videos.py` — resolution/fps/codec + sample frames of the videos the CNN misses.

Findings and numbers: `backend/eval/results/EVALUATION_SUMMARY.md` and CLAUDE.md sections 3 and 10.
