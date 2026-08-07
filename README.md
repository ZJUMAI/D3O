# D3O

This repository is the experimental code used for
**D3O: Dynamic Distribution Distillation for Ordinal Regression**.


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

