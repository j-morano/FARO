# FARO

This is the official repository of "FARO: A clinically guided <ins>f</ins>oundation model for generalizable <ins>a</ins>nalysis of <ins>r</ins>etinal <ins>o</ins>ptical coherence tomography", led by José Morano and Hrvoje Bogunović, from the Medical University of Vienna.

FARO is a general-purpose foundation model for retinal OCT that offers high quality representations that generalize well to a wide range of tasks, including volume and B-scan classification and retrieval, longitudinal change detection, and B-scan segmentation.
The model accepts single B-scans or entire OCT volumes with arbitrary number of B-scans as input, up to 256, and provides patch-level, B-scan-level, and volume-level embeddings as output, with sizes of (256, 768), (2, 768), and (16, 768), respectively, with the latter being by default downsampled to (2, 768) and reshaped to (1, 1536).


## Requirements

For managing the dependencies, we used [uv](https://github.com/astral-sh/uv).
With this tool, you can easily install the requirements using the following commands:

```bash
uv venv
source venv/bin/activate
uv sync
```

For using other virtual environment managers, please check the specific dependencies in `pyproject.toml`.


## Inference

The script `infer.py` provides the code for running inference with FARO for whole volumes, returning the embeddings for each volume with the shape (1, 1536).
For more details, please inspect the script itself, which provides a usage example in the main.


## Training and evaluation

Training and evaluation code can be found under `src/` and `src/eval/`, respectively. Please check the README files in those directories for more details.


## TODO

This repository is still under development, and updates will be made in the near future. The following features are planned, in order of priority:

- [x] Evaluation code.
- [x] Training code.
- [x] Plotting utilities.
- [x] Inference code for 3D and weights.
- [ ] Processed benchmark datasets.
- [ ] Inference code for 2D.

## Acknowledgements
FARO code is mainly based on MIRAGE, along with timm, DeiT, DINO, MoCo-v3, BEiT, MAE-priv, MAE, mmsegmentation, MONAI, and RETFound. We thank the authors for making their code available. In addition, the code for UrFound, VisionFM, OCTCube and iBOT was used in our experiments for model comparisons.

- https://github.com/j-morano/MIRAGE
- https://github.com/EPFL-VILAB/MultiMAE
- https://github.com/rwightman/pytorch-image-models/tree/master/timm
- https://github.com/facebookresearch/deit
- https://github.com/facebookresearch/dino
- https://github.com/facebookresearch/moco-v3
- https://github.com/microsoft/unilm/tree/master/beit
- https://github.com/BUPT-PRIV/MAE-priv
- https://github.com/facebookresearch/mae
- https://github.com/open-mmlab/mmsegmentation
- https://github.com/Project-MONAI/MONAI
- https://github.com/rmaphoh/RETFound_MAE
- https://github.com/yukkai/UrFound
- https://github.com/bytedance/ibot
- https://github.com/ABILab-CUHK/VisionFM
- https://github.com/ZucksLiu/OCTCubeM
