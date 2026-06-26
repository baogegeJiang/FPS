# FPS-UDA

FPS-UDA is a PyTorch library for feature-space unsupervised domain adaptation.
It separates image feature extraction from feature-space training:

1. Build or download a dataset-level H5 feature bank.
2. Select source/target feature views.
3. Train FPS from NumPy/Torch/H5 features.

The package name is `fps-uda`; the import name is `fps_uda`.

- GitHub: [baogegeJiang/FPS](https://github.com/baogegeJiang/FPS)
- Documentation: [baogegeJiang.github.io/FPS](https://baogegeJiang.github.io/FPS/)

<p align="center">
  <img src="docs/assets/shallow-intelligence-group.png" alt="Shallow Intelligence Group" width="260">
</p>

## Paper

This repository accompanies the FPS paper:

**Feature-Space Planes Searcher: A Universal Domain Adaptation Framework for
Interpretability and Computational Efficiency**

[Zhitong Cheng](https://scholar.google.com/citations?hl=zh-CN&user=T4lifpIAAAAJ)+,
[Yiran Jiang](https://scholar.google.com/citations?user=FRCRT-UAAAAJ&hl=zh-CN)+,
Yulong Ge, Yufeng Li, Zhongheng Qin, Rongzhi Lin, and
[Jianwei Ma](https://scholar.google.com/citations?user=6V78tzkAAAAJ&hl=zh-CN)*.

`+` Equal contribution. `*` Corresponding author.

- IEEE Xplore: [document 11568428](https://ieeexplore.ieee.org/abstract/document/11568428)
- Method: Feature-Space Planes Searcher (FPS)
- Task setting: frozen-feature unsupervised domain adaptation

If you use this repository, please cite the IEEE Xplore paper. A temporary
BibTeX entry is provided below; update DOI, volume, pages, and year from IEEE
Xplore if your manuscript requires complete publication metadata.

```bibtex
@article{cheng_fps_uda,
  title  = {Feature-Space Planes Searcher: A Universal Domain Adaptation Framework for Interpretability and Computational Efficiency},
  author = {Cheng, Zhitong and Jiang, Yiran and Ge, Yulong and Li, Yufeng and Qin, Zhongheng and Lin, Rongzhi and Ma, Jianwei},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  url    = {https://ieeexplore.ieee.org/abstract/document/11568428},
  note   = {IEEE Xplore document 11568428}
}
```

In this public release, we slightly simplified the feature-bank extraction
augmentations to reduce stochastic variation across runs. In particular, random
crop and contrast perturbations are replaced with deterministic view generation,
such as five-crop variants, while random pooling is retained. Since the released
feature banks differ slightly from those produced by the original experimental
pipeline, we re-searched the benchmark hyperparameters for the public configs.
We also reorganized the experiment code into a modular library structure so that
feature extraction, feature-bank analysis, training, losses, and configuration
components can be reused more easily.

## Code Contributors

This open-source library includes code contributions from:

- [baogegeJiang](https://github.com/baogegeJiang)
- [re-Gwen](https://github.com/re-Gwen)
- [Long-louis](https://github.com/Long-louis)
- [sayori1698](https://github.com/sayori1698)
- [ZhongH-Qin](https://github.com/ZhongH-Qin)

## Install

For local development:

```bash
pip install -e ".[dev]"
```

Install directly from GitHub:

```bash
pip install "fps-uda[vision,hf] @ git+https://github.com/baogegeJiang/FPS.git"
```

Or clone the repository when you also want the packaged real-H5 example file:

```bash
git clone https://github.com/baogegeJiang/FPS.git
cd FPS
pip install -e ".[vision,hf,dev]"
```

Install vision dependencies when extracting features from images:

```bash
pip install -e ".[vision,dev]"
```

Install Hugging Face download support when using released feature banks:

```bash
pip install -e ".[hf]"
```

If you do not install the package, run the CLI module directly from the repo:

```bash
PYTHONPATH=src python -m fps_uda.cli --help
```

After `pip install -e .`, the console command is available:

```bash
fps-uda --help
```

## Quick Start

The repository includes a compressed real H5 feature bank for Office31
`amazon -> webcam` with ViT features. This example requires no dataset download.

Run the full example config:

```bash
PYTHONPATH=src python -m fps_uda.cli train \
  --config configs/examples/office31_amazon_to_webcam_vit_packaged_h5.yaml \
  --out runs/examples/office31_amazon_to_webcam_vit_packaged
```

For a quick smoke run:

```bash
PYTHONPATH=src python -m fps_uda.cli train \
  --config configs/examples/office31_amazon_to_webcam_vit_packaged_h5.yaml \
  --iter-num 2 \
  --out runs/examples/office31_amazon_to_webcam_vit_smoke
```

The packaged H5 is:

```text
tests/fixtures/office31_amazon_to_webcam_vit_smoke.h5
```

For GitHub hosting, this file is close to the regular Git file-size limit. Git
LFS is recommended if you keep it in the repository. If you prefer a lightweight
Git repository, remove the fixture from Git and download banks from Hugging Face
with `scripts/download_feature_banks.py`.

It keeps all `amazon` and `webcam` samples and the views used by
`configs/training/office31/amazon_to_webcam/vit.yaml`, stored as float16 + gzip.
Regenerate it from the full Office31 ViT bank with:

```bash
python scripts/make_smoke_feature_bank.py --force
```

## Core Concepts

### Feature Bank

FPS-UDA stores image features in a dataset-level H5 bank:

```text
/domains/{domain}/label
/domains/{domain}/views/{view_key}/feature
```

Each domain can be source or target. Each `view_key` represents one deterministic
preprocessing/pooling view, such as:

```text
pad_resize256_input224_center_orig_clean
pad_resize256_input224_center_orig_pool_a
pad_resize256_input224_center_orig_pool_b
```

### Training View Roles

Training uses explicit feature roles:

| Role | Meaning |
| --- | --- |
| `src` | source supervised features and labels |
| `entropy` | target view used by LSE/LCE entropy losses |
| `cr.view1` | first target view used by consistency regularization |
| `cr.view2` | second target view used by consistency regularization |
| `eval` | target view used for evaluation and predictions |

Each role has:

| Field | Meaning |
| --- | --- |
| `key` | one view key or comma-separated multiple view keys |
| `combine: stack` | concatenate samples from multiple views |
| `combine: mean` | average aligned views sample-wise |

## CLI Workflow

FPS-UDA exposes four CLI commands:

```bash
fps-uda extract-feature-bank
fps-uda analyze-feature-bank
fps-uda train
fps-uda sweep
```

Use `PYTHONPATH=src python -m fps_uda.cli ...` instead of `fps-uda ...` when the
package is not installed.

### Download Released Feature Banks

Benchmark banks can be downloaded from the Hugging Face dataset repo
`baogege1995/FPS_H5`. Files are stored under the `banks/` subdirectory.

```bash
python scripts/download_feature_banks.py all
```

Download a subset:

```bash
python scripts/download_feature_banks.py office31_resnet office_home_vit
```

Banks are written to:

```text
fps_h5cache/banks/
```

### Prepare Image Datasets

Use helper scripts to download datasets and generate manifests:

```bash
python scripts/download_datasets.py office31 --root data
python scripts/download_datasets.py office_home --root data
python scripts/download_datasets.py visda17 --root data
```

For existing local copies:

```bash
python scripts/download_datasets.py all --root data --skip-download
```

Generated manifests:

| Dataset | Manifest path |
| --- | --- |
| Office31 | `data/office31/{amazon,dslr,webcam}/annotations/annotations.txt` |
| OfficeHome | `data/office_home/{Art,Clipart,Product,Real World}/annotations/annotations.txt` |
| VisDA17 | `data/visda17/{train,validation}/image_list.txt` |

Download URLs can be overridden with `--office31-url`, `--office-home-url`,
`--visda17-train-url`, and `--visda17-validation-url`.

### Extract a Feature Bank

Dataset YAML controls domains, transforms, backbone, and feature-bank views:

```bash
fps-uda extract-feature-bank \
  --dataset-config configs/datasets/office31_vit.yaml \
  --out fps_h5cache/banks/office31_vit.h5 \
  --device cuda:0 \
  --num-workers 16
```

Regenerate all benchmark banks:

```bash
PYTHON_BIN=/home/jiangyiran/.conda/envs/All/bin/python \
DEVICE=cuda:0 \
NUM_WORKERS=16 \
bash scripts/extract_benchmark_feature_banks.sh
```

Override the backbone from the CLI:

```bash
fps-uda extract-feature-bank \
  --dataset-config configs/datasets/office_home_resnet.yaml \
  --backend torchvision \
  --backbone resnet101 \
  --weights IMAGENET1K_V1 \
  --out fps_h5cache/banks/office_home_resnet101.h5
```

### Analyze Feature-Bank Views

Analysis uses labels for debugging and view selection. It does not modify H5.

```bash
fps-uda analyze-feature-bank \
  --feature-bank fps_h5cache/banks/office_home_resnet50.h5 \
  --source-domain Art \
  --target-domain Product \
  --out runs/analysis/office_home_resnet50
```

Outputs include CSV, JSON, recommended YAML snippets, and plots when matplotlib
is installed.

### Train

Train from a YAML config:

```bash
fps-uda train \
  --config configs/training/office31/amazon_to_webcam/vit.yaml \
  --out runs/office31/amazon_to_webcam/vit
```

Override common settings from the CLI:

```bash
fps-uda train \
  --config configs/training/office31/amazon_to_webcam/vit.yaml \
  --device cuda:0 \
  --iter-num 1000 \
  --base-lr 0.000375 \
  --margin-start-step 100 \
  --out runs/debug/office31_aw_vit
```

Train without a config by specifying the bank and view roles:

```bash
fps-uda train \
  --feature-bank fps_h5cache/banks/office31_vit.h5 \
  --source-domain amazon \
  --target-domain webcam \
  --src-view pad_resize256_input224_center_orig_clean \
  --entropy-view pad_resize256_input224_center_orig_clean \
  --cr-view1 pad_resize256_input224_center_orig_pool_a \
  --cr-view2 pad_resize256_input224_center_orig_pool_b \
  --eval-view pad_resize256_input224_center_orig_clean \
  --feature-transform none \
  --device cuda:0 \
  --base-lr 0.000375 \
  --iter-num 1000 \
  --out runs/debug/cli_only
```

### Sweep

Run alpha/beta sweeps:

```bash
fps-uda sweep \
  --config configs/training/office31/amazon_to_webcam/vit.yaml \
  --alpha-grid 0.6,0.8,1.0 \
  --beta-grid 0.4,0.6 \
  --seeds 0 1 2 \
  --out runs/sweeps/office31_aw_vit
```

## Training YAML

Training configs use a sectioned schema. Flat YAML is rejected.

```yaml
io:
  feature_bank: fps_h5cache/banks/office31_vit.h5
  source_domain: amazon
  target_domain: webcam
  feature_transform: none
  num_classes: 31
  device: cuda:0

views:
  src: {key: pad_resize256_input224_center_orig_clean, combine: stack}
  entropy: {key: pad_resize256_input224_center_orig_clean, combine: mean}
  cr:
    view1: {key: pad_resize256_input224_center_orig_pool_a, combine: stack}
    view2: {key: pad_resize256_input224_center_orig_pool_b, combine: stack}
  eval: {key: pad_resize256_input224_center_orig_clean, combine: mean}

optimization:
  optimizer: sgd
  base_lr: 0.000375
  momentum: 0.9
  nesterov: false
  weight_decay: 0.0
  lr_schedule: linear_step
  min_lr: 0.0

schedule:
  seed: 0
  iter_num: 3000
  alpha: 0.8
  beta: 0.6
  alpha_0: 0.3
  beta_0: 0.15
  schedule_tau: 1000
  dynamic_parameters: true

normalization:
  normalize: cross_norm
  cross_norm_scale: 1
  cross_norm_target_weight: 0.5

losses:
  use_consistency_loss: true
  use_lse: true
  use_lce: true
  use_lcr: true
  lambda_lcr: 1.0
  lcr_loss: mse
  lcr_sample_weight: paper
  use_classes_weight: true
  sparse_weight_a: 5
  pseudo_margin: true
  margin_start_step: 100
  margin_quantile: 0.01
  margin_weight_type: normalized_logit_gap
  margin_convert_mode: by_type
  margin_sigmoid_tau: 0.2
  margin_sigmoid_boundary_weight: 0.5

eval:
  eval_interval: 100
  multi_class: true
  progress: true
```

### Training Parameter Reference

`io`

| Field | Meaning |
| --- | --- |
| `feature_bank` | path to dataset-level H5 bank |
| `source_domain` | domain used for supervised source training |
| `target_domain` | domain used for target entropy/CR/eval roles |
| `feature_transform` | `none` or `sqrt`; `sqrt` is often used for ResNet banks |
| `num_classes` | number of classes |
| `device` | PyTorch device, such as `cpu` or `cuda:0` |

`views`

| Field | Meaning |
| --- | --- |
| `src` | source feature view for supervised CE |
| `entropy` | target feature view for entropy losses |
| `cr.view1`, `cr.view2` | paired target views for LCR |
| `eval` | target feature view for reporting metrics |
| `combine` | `stack` repeats labels per view; `mean` averages aligned views |

`optimization`

| Field | Meaning |
| --- | --- |
| `optimizer` | `sgd` or `adamw` |
| `base_lr` | LR slope for `linear_step`; initial LR for `constant`/`cosine` |
| `momentum`, `nesterov` | SGD parameters |
| `weight_decay` | optimizer weight decay |
| `adamw_betas`, `adamw_eps` | AdamW parameters |
| `lr_schedule` | `linear_step`, `constant`, or `cosine` |
| `min_lr` | final LR floor for cosine |

`schedule`

| Field | Meaning |
| --- | --- |
| `iter_num` | number of optimizer updates |
| `alpha`, `alpha_0` | final/initial entropy balance |
| `beta`, `beta_0` | final/initial supervised-vs-target balance |
| `schedule_tau` | exponential schedule time scale |
| `dynamic_parameters` | if false, use fixed `alpha` and `beta` |
| `src_sample_ratio` | optional source sample fraction per step |
| `target_sample_ratio` | optional entropy/CR sample fraction per step |

`normalization`

| Field | Meaning |
| --- | --- |
| `normalize` | `none`, `cross_norm`, or `self_norm` |
| `cross_norm_scale` | divides the cross-domain standard deviation |
| `cross_norm_target_weight` | target contribution to cross-domain mean/std |
| `self_norm_scale_src`, `self_norm_scale_tgt` | self-normalization scales |

`losses`

| Field | Meaning |
| --- | --- |
| `use_lse`, `use_lce` | enable sample entropy and class entropy terms |
| `use_lcr` | enable consistency regularization between CR views |
| `lambda_lcr` | LCR weight |
| `lcr_loss` | logits-space `mse` or `l2` |
| `lcr_sample_weight` | `none`, `density`, `margin`, `density_margin`, or `paper` |
| `use_classes_weight` | use sparse density weights for target samples |
| `sparse_weight_a` | density softmax scale, or `class_aware` when labels exist |
| `pseudo_margin` | enable pseudo-margin sample weighting |
| `margin_start_step` | first step where pseudo-margin can be active |
| `margin_quantile` | quantile threshold for margin conversion |
| `margin_sigmoid_tau` | sigmoid temperature for margin weights |
| `margin_sigmoid_boundary_weight` | weight at the quantile boundary |
| `use_shift_constraint` | enable LDelta; also enables correction |
| `ldelta_weight`, `ldelta_decay_steps` | LDelta strength and decay |
| `sample_entropy_type` | `shannon`, `tsallis`, or `adaptive_temp_shannon` |

`eval`

| Field | Meaning |
| --- | --- |
| `eval_interval` | evaluate every N steps |
| `multi_class` | report multiclass metrics, macro-F1, CWC, ECE |
| `progress` | show tqdm progress bar |

## Dataset/Extraction YAML

Dataset configs are explicit; `name` is descriptive and does not drive hidden
dataset/model branches.

```yaml
name: office31_vit
root_dir: data/office31
backbone:
  backend: timm
  name: vit_base_patch16_224.augreg2_in21k_ft_in1k
  pretrained: true
  checkpoint: null
  in_features: 768
  kwargs:
    class_token: true
    global_pool: token
  pooling:
    feature_type: token
    random_strategy: token_channel_squared
loader:
  batch_size: 32
  num_workers: 16
  seed: 0
transform:
  interpolation: bicubic
  antialias: true
  pad_fill: 127
  mean: [0.5, 0.5, 0.5]
  std: [0.5, 0.5, 0.5]
domains:
  - name: amazon
    kind: manifest
    image_root: amazon
    manifest: amazon/annotations/annotations.txt
    path_column: 0
    label_column: 1
feature_bank:
  water_level: 0.0
  mute_padding_in_pool: true
  views:
    - key: pad_resize256_input224_center_orig
      pad_to_square: true
      resize_size: 256
      input_size: 224
      crop: center
      flip: orig
```

### Extraction Parameter Reference

| Section | Field | Meaning |
| --- | --- | --- |
| root | `root_dir` | dataset root used by domain paths |
| `backbone` | `backend` | `torchvision`, `timm`, `hf_vit`, or `clip` |
| `backbone` | `name` | model name passed to the backend |
| `backbone` | `weights` | torchvision weights enum name, when applicable |
| `backbone` | `checkpoint` | optional local checkpoint path |
| `backbone.pooling` | `feature_type` | `spatial`, `token`, or `flat` |
| `backbone.pooling` | `random_strategy` | ResNet uses `spatial_shared`; ViT uses `token_channel_squared` |
| `loader` | `batch_size`, `num_workers` | DataLoader settings |
| `transform` | `mean`, `std` | image normalization |
| `transform` | `interpolation`, `antialias`, `pad_fill` | deterministic preprocessing controls |
| `domains` | `kind: manifest` | read image path/label columns from a text file |
| `domains` | `kind: class_folder` | infer labels from class subfolders |
| `feature_bank` | `water_level` | random pooling water level |
| `feature_bank` | `mute_padding_in_pool` | exclude padding from pooling masks when possible |
| `feature_bank.views` | `pad_to_square`, `resize_size`, `input_size`, `crop`, `flip` | deterministic view geometry |

For each configured base view, extraction writes `clean` plus random pooling
views such as `pool_a` and `pool_b`.

## Python API

Train directly from NumPy or Torch arrays:

```python
from fps_uda import FeatureSet, FPSConfig, train_fps

features = FeatureSet(
    src_features=src_x,
    src_labels=src_y,
    entropy_features=tgt_x_eval,
    entropy_labels=tgt_y_eval,
    cr_features_1=tgt_x_view1,
    cr_features_2=tgt_x_view2,
    cr_labels=tgt_y_eval,
    eval_features=tgt_x_eval,
    eval_labels=tgt_y_eval,
)

config = FPSConfig(
    num_classes=31,
    feature_dim=src_x.shape[1],
    device="cuda:0",
    base_lr=0.000375,
    iter_num=3000,
)

result = train_fps(features, config, output_dir="runs/python_api")
print(result.best_metric, result.best_score)
```

Load a feature bank in Python:

```python
from fps_uda.io import load_feature_bank_h5

features = load_feature_bank_h5(
    "fps_h5cache/banks/office31_vit.h5",
    source_domain="amazon",
    target_domain="webcam",
    src_view="pad_resize256_input224_center_orig_clean",
    entropy_view="pad_resize256_input224_center_orig_clean",
    cr_view1="pad_resize256_input224_center_orig_pool_a",
    cr_view2="pad_resize256_input224_center_orig_pool_b",
    eval_view="pad_resize256_input224_center_orig_clean",
)
```

## Customization

### Custom Loss

Custom Python losses can be added without modifying the trainer. A loss receives
a dict-like context with `batch`, `outputs`, `weights`, `schedules`, `step`, and
mutable `state` entries:

```python
from fps_uda import LossOutput

def extra_entropy_logit_penalty(ctx):
    logits = ctx["outputs"]["entropy.logits"]
    value = logits.sum() * 0.0 if ctx["step"] < 100 else 0.01 * logits.pow(2).mean()
    return LossOutput(
        "extra_entropy_logit_penalty",
        value,
        {"loss_extra": float(value.detach().cpu())},
    )

result = train_fps(features, config, extra_loss_terms=[extra_entropy_logit_penalty])
```

Use `loss_terms=[...]` to replace the default FPS loss list, or
`extra_loss_terms=[...]` to append custom losses.

Runnable example:

```bash
PYTHONPATH=src python examples/custom_loss_training.py
```

### Custom Backbone

For private models or non-standard feature extractors, use the pure Python
example instead of the YAML/CLI extraction path:

```bash
PYTHONPATH=src python examples/advanced_custom_backbone_feature_bank.py
```

The example defines a custom `torch.nn.Module`, wraps it with `PoolableBackbone`,
and writes a standard FPS-UDA feature bank. A custom model may return:

| Shape | Meaning |
| --- | --- |
| `[B, D]` | flat features |
| `[B, C, H, W]` | spatial feature maps |
| `[B, L, C]` | token features |

## Benchmark Utilities

Run every benchmark Office31, OfficeHome, and VisDA17 config:

```bash
PYTHON_BIN=/home/jiangyiran/.conda/envs/All/bin/python \
DEVICE=cuda:0 \
RESUME=1 \
KEEP_GOING=1 \
bash scripts/run_benchmarks.sh
```

Useful filters:

```bash
DATASETS=office31 BACKBONES=vit DRY_RUN=1 bash scripts/run_benchmarks.sh
TASKS=amazon_to_webcam ITER_NUM=1000 bash scripts/run_benchmarks.sh
```

Historical tex-table comparison helpers and local run outputs are archived under
`bak/` and are not part of the public library workflow.

## Repository Layout

```text
src/fps_uda/          Python package
configs/datasets/    feature-bank extraction configs
configs/training/    benchmark training configs
configs/examples/    small runnable examples
examples/            pure Python customization examples
scripts/             dataset, bank, and benchmark helpers
tests/fixtures/      packaged real-H5 example fixture
```

Historical experiment folders are archived under `bak/` and are not public
entry points.
