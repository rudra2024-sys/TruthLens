# Grad-CAM faithfulness (deletion test)

- Model: ConvNeXt-Tiny v2 (the model the heatmap explains). Images: 195 fakes it scores FAKE (>= 0.6), sampled across 9 sources.
- Mask: top 20% of pixels by Grad-CAM vs. the same area as random 16x16 blocks (mean of 5 draws), filled with the dataset mean colour.

| | mean FAKE probability |
|---|---|
| original | 0.936 |
| after masking Grad-CAM region | 0.704 |
| after masking random region | 0.892 |

- Mean drop: Grad-CAM **0.232** vs random **0.044**.
- Grad-CAM mask hurt the FAKE score more than the random mask on **85%** of images.

| source | n | mean drop (Grad-CAM) | mean drop (random) | Grad-CAM wins |
|---|---|---|---|---|
| aivshuman_train | 26 | 0.010 | 0.005 | 92% |
| blind_spot_portrait_app | 1 | 0.007 | 0.020 | 0% |
| cifake_test | 26 | 0.053 | 0.017 | 88% |
| deepdetect_test | 26 | 0.050 | 0.026 | 69% |
| faces140k_test | 26 | 0.042 | 0.029 | 73% |
| genimage_fake_biggan | 26 | 0.717 | 0.182 | 96% |
| genimage_fake_midjourney | 20 | 0.256 | 0.024 | 80% |
| genimage_fake_sd | 18 | 0.295 | 0.017 | 83% |
| openfake_test | 26 | 0.470 | 0.042 | 96% |
