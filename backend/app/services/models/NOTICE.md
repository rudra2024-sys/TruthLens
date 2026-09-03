# Third-Party Model Attribution

This project bundles pretrained weights (converted to ONNX) from two external,
open-source research projects. Per their licenses, attribution and license
text are preserved below.

---

## MesoNet (image / video pipeline)

- Paper: Afchar, D., Nozick, V., Yamagishi, J., Echizen, I. "MesoNet: a Compact
  Facial Video Forgery Detection Network." WIFS 2018. https://arxiv.org/abs/1809.00888
- Source: https://github.com/DariusAf/MesoNet
- License: Apache License 2.0
- Weights used: `Meso4_DF.h5`, `MesoInception_DF.h5` (converted to
  `meso4_df.onnx`, `meso_inception_df.onnx`)

```
Apache Mesonet - Facial Video Forgery Detection Network
Copyright 2018 The Apache Software Foundation
Copyright 2018, Darius Afchar, École des Ponts Paristech (France)
Copyright 2018, Vincent Nozick, JFLI (UMI-3527, Japan) and UPEM-LIGM (UMR-8049, France)
Copyright 2018, National Institute of Informatics
All rights reserved.

This product includes software developed at
The Apache Software Foundation (http://www.apache.org/).
```

## AASIST (audio pipeline)

- Paper: Jung, J. et al. "AASIST: Audio Anti-Spoofing Using Integrated
  Spectro-Temporal Graph Attention Networks." ICASSP 2022.
  https://arxiv.org/abs/2110.01200
- Source: https://github.com/clovaai/aasist
- License: MIT
- Weights used: `AASIST.pth` (converted to `aasist.onnx`), pretrained on ASVspoof2019-LA.

```
AASIST
Copyright (c) 2021-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

Neither project endorses or is affiliated with TruthLens; they are used here
under their respective open-source licenses, unmodified apart from the
architecture being re-expressed for ONNX export (weights are untouched).
