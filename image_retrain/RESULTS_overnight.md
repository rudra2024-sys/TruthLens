# Image retrain overnight results (2026-10-16) — NOTHING SWAPPED, nothing committed

## Round 2 (local Mac, ConvNeXt only; CLIP round3 head frozen)  — staged: work/staging/convnext_r2_candidate.pt
Rollback = keep models/checkpoints/image/convnext_tiny_diversified_v2.pth (untouched).
Real-photo specificity (higher = fewer false FAKE) / fake recall, deployed(old+clip max) -> r2 max(new+clip):
- UTKFace real 297: 94.3% -> 67% at 0.5; with threshold tuned on val reals only (t=0.787): 93.3%
- GenImage fakes recall 30% -> 67% (tuned t); ADM never-seen generator 17% -> 29-48%; SUT (Flux/MJ6/NanoBanana) 53% -> 76-85%
- ChatGPT-8 0/8 -> 3/8 (tuned) or 6/8 (0.5); portrait-app 14/15 kept (93%)
- Face-swap stills 0-1/3 (still weak); dff face-swap crops recall 12% -> 28-64%
Verdict: big fake-recall gains, real-photo specificity ONLY matches deployed if the FAKE threshold is raised (~0.79). At 0.5 it is
a regression on real faces. Swapping would need: new checkpoint + threshold change + re-calibration (your OK required).
## Round 3 (running, from r2, dff face-swap crops cut to 700, real weight 1.8) -> work/run_local_r3/results.json, work/full3.log
Expected done ~05:30. Compare against the table above before deciding.
