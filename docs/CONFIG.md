# Configuration Guide

## Overview

Configuration is composed of three YAML files merged in order. Later files override earlier ones:

```
evaluation.yaml (defaults) → dataset config → model config (highest priority)
```

## File Layout

```
configs/
├── evaluation.yaml          # Global evaluation defaults
├── dataset/
│   ├── ml100k.yaml
│   ├── ml1m.yaml
│   └── ...
└── model/
    ├── general/             # Baseline models (EASE, LightGCN, MF, ...)
    └── csar/                # LIRA and ASPIRE family
```

---

## 1. `evaluation.yaml` — Global Defaults

Applied to all models unless overridden by a model config.

```yaml
evaluation:
  batch_size: 2048
  validation_method: "sampled"  # "full", "sampled", "uni99"
  final_method: "full"
  metrics:
    - "NDCG"
    - "Recall"
    - "HitRate"
  top_k: [5, 10, 20, 50]
  main_metric: "NDCG"
  main_metric_k: 10
  long_tail_percent: 0.8
```

---

## 2. Dataset Config

### Required fields

| Field | Description | Example |
|---|---|---|
| `dataset_name` | Dataset identifier | `"ml-1m"` |
| `data_path` | Path to the raw data file | `"/data/ml-1m/ratings.dat"` |
| `separator` | Column separator | `"::"`, `"\t"`, `","` |
| `columns` | Column names | `["user_id", "item_id", "rating", "timestamp"]` |
| `min_user_interactions` | Minimum interactions per user (k-core) | `5` |
| `min_item_interactions` | Minimum interactions per item (k-core) | `5` |
| `split_method` | How to split data | see below |
| `data_cache_path` | Cache directory | `"./data_cache/"` |

### Optional fields

| Field | Description | Default |
|---|---|---|
| `rating_threshold` | Only keep interactions at or above this rating | `null` (keep all) |
| `train_ratio` | Train fraction for ratio-based splits | `0.8` |
| `valid_ratio` | Validation fraction for ratio-based splits | `0.1` |
| `has_header` | Whether the data file has a header row | `false` |
| `dedup` | Remove duplicate (user, item) interactions | `true` |

### Split methods

**`loo` (Leave-One-Out):**

```yaml
split_method: "loo"
```

The last interaction per user (by timestamp, or by item_id as tie-breaker) goes to test; the second-last to validation. All models use this for fair comparison. With LOO, `HitRate@K == Recall@K` because each user has exactly one test item.

**`temporal_ratio`:**

```yaml
split_method: "temporal_ratio"
train_ratio: 0.8
valid_ratio: 0.1
```

Requires a timestamp column. Interactions are sorted by time; the last 10% per user goes to test, next 10% to validation.

**`random`:**

```yaml
split_method: "random"
train_ratio: 0.8
valid_ratio: 0.1
```

Same as `temporal_ratio` but without requiring timestamps. Use for datasets that lack timestamp data.

**`presplit`:**

```yaml
split_method: "presplit"
train_file: "/data/gowalla/train.txt"
test_file: "/data/gowalla/test.txt"
```

Loads externally provided train/test files in LightGCN format (each line: `user_id item_id item_id ...`). No k-core filtering or ID remapping is applied — IDs must be pre-processed and zero-indexed. No validation set is created.

### Example: MovieLens 1M

```yaml
dataset_name: "ml-1m"
data_path: "/data/ml-1m/ratings.dat"
separator: "::"
columns: ["user_id", "item_id", "rating", "timestamp"]
rating_threshold: 4
min_user_interactions: 5
min_item_interactions: 5
split_method: "temporal_ratio"
train_ratio: 0.8
valid_ratio: 0.1
data_cache_path: "./data_cache/"
```

---

## 3. Model Config

### Closed-form models (no training loop)

```yaml
model:
  name: ease
  reg_lambda: 500.0

device: "auto"
```

No `train:` block means the Trainer calls `fit()` directly and skips gradient descent.

### Gradient-based models

```yaml
model:
  name: lightgcn
  embedding_dim: 64
  n_layers: 3

train:
  epochs: 500
  batch_size: 1024
  lr: 0.001
  optimizer: "adam"
  loss_type: "pairwise"
  num_negatives: 1
  early_stop_patience: 40
  embedding_l2: 1.0e-4

device: "auto"
```

### Per-model evaluation override

A model config can override the global evaluation settings:

```yaml
evaluation:
  metrics: ["NDCG", "HitRate"]
  top_k: [10, 50]
  validation_method: "uni99"
```

---

## Data Cache

Preprocessed data is cached in `data_cache/` with the key:

```
{dataset_name}_{split_method}_rt{rating_threshold}_mu{min_u}_mi{min_i}_tr{train_ratio}_vr{valid_ratio}_dedup{0|1}.pkl
```

Any change to preprocessing parameters produces a new cache file. The old file is not deleted automatically.

---

## HitRate vs. Recall

| | LOO | temporal\_ratio / random |
|---|---|---|
| Ground truth size | 1 per user | N per user (variable) |
| HitRate@K | 1 if the test item is in top-K | 1 if any test item is in top-K |
| Recall@K | Equal to HitRate | (# test items in top-K) / (total test items) |
| NDCG@K | `1/log2(rank+1)` | Multi-item DCG/IDCG |
