# Locality-Aware Inter-and Intra-Video Reconstruction for Self-Supervised Correspondence Learning

This repo contains the unofficial supported code and configuration files to reproduce [LIIR](https://arxiv.org/abs/2203.14333). It is based on [MAST](https://github.com/zlai0/MAST).

## Updates

 - [2022-06-21] Initial commits

## Results and Models

| config | J&F | model | precomputed results |
| :---: | :---: | :---: | :---: |
| APE | 66.9 | [github](https://github.com/shasha521/LIIR_pytorch/releases/download/1.0/APE_669.pt) | [github](https://github.com/shasha521/LIIR_pytorch/releases/download/1.0/APE_669.zip)
| APE + compact prior | 69.0 | [github](https://github.com/shasha521/LIIR_pytorch/releases/download/1.0/APE_compact_690.pt) | [github](https://github.com/shasha521/LIIR_pytorch/releases/download/1.0/APE_compact_690.zip) |
| APE + inter-video reconstruction | 69.9 | [github](https://github.com/shasha521/LIIR_pytorch/releases/download/1.0/APE_intervideo_699.pt) | [github](https://github.com/shasha521/LIIR_pytorch/releases/download/1.0/APE_intervideo_699.zip) |
| APE + inter-video reconstruction + compact prior | 72.2 | [github](https://github.com/shasha521/LIIR_pytorch/releases/download/1.0/APE_compact_intervideo_722.pt) | [github](https://github.com/shasha521/LIIR_pytorch/releases/download/1.0/APE_compact_intervideo_722.zip) |

## Usage

### Requirement

Pytorch == 1.8.0 & torchvision == 0.9.0 & Spatial-correlation-samplar== 0.3.0

We do find verions of Pytorch and Spatial-correlation-samplar affect the results, please stick to our recommend setting.

We also provide the conda environment we used to help the reproduction [[gDrive]()].

### Inference
```
# APE
CUDA_VISIBLE_DEVICES=0 python evaluate_davis.py --resume checkpoints/APE_669.pt

# APE + Spatial Compactness Prior
CUDA_VISIBLE_DEVICES=0 python evaluate_davis.py --resume checkpoints/APE_compact_690.pt --compact

# APE + Inter-video training
CUDA_VISIBLE_DEVICES=0 python evaluate_davis.py --resume checkpoints/APE_intervideo_699.pt --usemomen

# APE + Inter-video training + Spatial Compactness Prior
CUDA_VISIBLE_DEVICES=0 python evaluate_davis.py --resume checkpoints/APE_compact_intervideo_722.pt --usemomen --compact
```

### Training
Step 1, first we need to run:

```
# Baseline + APE
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12222  --lr 1e-3

# Baseline with 1/8 resolution participated in the reconstruction
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12222  --semantic --lr 1e-3

# Baseline + APE + Compactness Prior
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12222  --semantic --compact --lr 1e-3
```

Step 2, and then:
```
# Baseline + APE + Spatial Compactness Prior + Inter-video Reconstruction
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12222  \
    --usemomen --compact --lr 1e-4 --epochs 5 --pretrain [Step 1 checkpoints]
```

### Bag of tricks
We recommend decoupling the first training step by running:
```
# Baseline + APE
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12222  --lr 1e-3

# Baseline + APE + 1/8 resultion reconstruction + Compactness Prior
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12222  \
    --semantic --compact --lr 1e-4 --pretrain [Baseline + APE checkpoints]

```

Freeze BN can also help:
```
# Add Freeze BN
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12222  \
    --freeze_bn --usemomen --compact --lr 1e-4 --epochs 5 --pretrain [checkpoints]
```

**Notes:** 
- We use two Tesla A100 GPUs for training. CUDA version: 11.1.

## TODO
- [x] Code release
- [x] Checkpoint release
- [ ] Evaluation code for YT-VOS, VIP and JHMDB
- [ ] PE Shuffle (we encounter some environment issues)

## Citing HieraSeg
```BibTeX
@article{li2022locality,
  title={Locality-Aware Inter-and Intra-Video Reconstruction for Self-Supervised Correspondence Learning},
  author={Li, Liulei and Zhou, Tianfei and Wang, Wenguan and Lu, Yang and Li, Jianwu and Yang, Yi},
  journal={CVPR},
  year={2022}
}
```

 
