# Third-party model weights

The MIT license in `LICENSE` covers the source code of this project. It does
**not** cover the pretrained face models that `recon/setup_models.py`
downloads at setup time. Those weights are distributed by their authors under
their own terms:

| Model | Source | Notes |
|---|---|---|
| InsightFace `buffalo_l` (SCRFD-10G detector + `w600k_r50` recognizer) | [InsightFace model zoo](https://github.com/deepinsight/insightface) | InsightFace source is MIT; the published pretrained model packs are released for **non-commercial research** use. |
| AdaFace IR-101 (ONNX export) | [yakhyo/adaface-onnx](https://github.com/yakhyo/adaface-onnx) | AdaFace source is MIT; check the export repository and the original paper for weight terms. |
| LVFace-B Glint360K (ONNX) | [bytedance-research/LVFace](https://huggingface.co/bytedance-research/LVFace) | Code is MIT; the released weights are intended for **non-commercial research**. |

Before using this project commercially, replace these weights with models you
are licensed to use, or obtain permission from the respective authors. The
project can run with `buffalo_l` alone (`--models w600k`), but even that pack
carries the non-commercial research restriction above.

This project only downloads the models; it never uploads photos or embeddings.
