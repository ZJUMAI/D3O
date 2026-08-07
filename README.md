# D3O — Adience release

This repository is the Adience-only cleanup of the experimental code used for
**D3O: Dynamic Distribution Distillation for Ordinal Regression**.

The model, loss computation, EMA teacher, shallow-layer distillation, ConLE
implementation, optimizer grouping, metrics, and Lightning training loop are
retained from the original `OrdinalCLIP` experiment. Dataset switches and
configuration files for MORPH II, Aesthetics, Historical Color Images,
Diabetic Retinopathy, and Tongue datasets have been removed.

## Data

Download Adience separately. The loader expects the original experiment layout:

```text
data/adience/
├── aligned/
└── splits/
    ├── test_fold_is_0/
    │   ├── age_train.txt
    │   └── age_test.txt
    ├── test_fold_is_1/
    └── ...
```

Each split line is:

```text
relative/image/path.jpg LABELS
```

`LABELS` follows the original loader convention: a single class digit or
multiple annotator class digits such as `233`. Training samples one digit at
random; evaluation uses the middle digit. Classes are numbered `0` through `7`.

Edit the six paths in `configs/adience.yaml`, or override them on the command
line.

## Installation

```bash
conda create -n d3o python=3.11 -y
conda activate d3o
pip install -r requirements.txt
pip install -e .
```

The original experiment instantiated both DINOv3 and CLIP vision encoders. DINOv3
is retained for compatibility with the original checkpoints, although the
released forward pass uses the CLIP hidden states.

## Train

Fold 0:

```bash
python train.py
```

Another fold and external data paths:

```bash
python train.py --cfg_options \
  data_cfg.fold=1 \
  data_cfg.train_images_root=/path/to/aligned \
  data_cfg.val_images_root=/path/to/aligned \
  data_cfg.test_images_root=/path/to/aligned \
  data_cfg.train_data_file=/path/to/splits \
  data_cfg.val_data_file=/path/to/splits \
  data_cfg.test_data_file=/path/to/splits
```

Run all five folds by repeating with `data_cfg.fold=0,1,2,3,4` and a distinct
`--output_dir`.

## Test a checkpoint

```bash
python train.py --test_only --test_dir /path/to/model.ckpt \
  --cfg_options data_cfg.fold=0
```

## What was intentionally changed

- All core Python files were copied from the actual experimental project and
  then reduced to the Adience execution path.
- Private absolute paths and the forced Hugging Face mirror were removed.
- The hard-coded Adience fold was exposed as `data_cfg.fold`.
- Unused dataset branches, dataset loaders, hyperbolic alternatives, notebooks,
  result files, and exploratory visualizations were removed.
- The original numerical class prompt file `data/classnames_8.txt` and final
  Adience hyperparameters were retained.

The release does not include Adience images, pretrained model weights, or
experimental checkpoints.
