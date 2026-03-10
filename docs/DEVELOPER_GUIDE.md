# Developer Guide

This guide explains how to add new models, understand the training and evaluation pipeline, and work with the framework's internal systems.

---

## BaseModel Interface

All models inherit from `src/models/base_model.py::BaseModel`. The required methods are:

### `calc_loss(self, batch_data) -> (tuple, dict | None)`

Called each training step for gradient-based models.

```python
def calc_loss(self, batch_data):
    """
    Args:
        batch_data (dict): {
            'user_id':     [B, 1],
            'pos_item_id': [B, 1],
            'neg_item_id': [B, K],
        }

    Returns:
        losses (tuple): (main_loss, reg_loss, ...)  — Trainer sums and calls backward()
        log_params (dict | None): values to log per step, e.g. {'loss': value}
    """
```

Loss weights are applied inside the model before returning. Use `self.get_l2_reg_loss(*tensors)` for L2 regularization (reads `train.embedding_l2` from config).

### `forward(self, users) -> Tensor[B, n_items]`

Full-item scoring for evaluation. Returns a score matrix over all items.

### `predict_for_pairs(self, user_ids, item_ids) -> Tensor[N]`

Pair-level scoring for sampled evaluation (uni99). Returns scores for each (user, item) pair.

### `get_final_item_embeddings(self) -> Tensor | None`

Returns the final item embedding matrix for ILD, novelty, and diversity metrics. Return `None` if not applicable.

---

## Adding a New Model

**Step 1 — Create the model file:**

```python
# src/models/general/my_model.py
from ..base_model import BaseModel

class MyModel(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.dim = config['model'].get('embedding_dim', 64)
        # self.device is set by BaseModel

    def calc_loss(self, batch_data): ...
    def forward(self, users): ...
    def predict_for_pairs(self, user_ids, item_ids): ...
    def get_final_item_embeddings(self): ...
```

---

## Parameter Tracking

Any scalar values returned as the second element of `calc_loss` are automatically tracked per epoch and saved as a plot and JSON file.

**How it works:**

1. `calc_loss` returns `(loss_tuple, dict | None)`.
2. The Trainer collects the dict values every step, averages them per epoch, and stores them in `self.tracked_params`.
3. After training, it writes `params_history.json` and renders `params_plot.png` — both saved alongside the model checkpoint.

**Example: tracking individual loss components and an internal parameter:**

```python
def calc_loss(self, batch_data):
    # ... compute losses ...
    bpr_loss = ...
    cl_loss  = ...
    l2_loss  = self.get_l2_reg_loss(self.user_emb.weight, self.item_emb.weight)

    params_to_log = {
        'loss_bpr': bpr_loss.item(),
        'loss_cl':  cl_loss.item(),
        'loss_l2':  l2_loss.item(),
    }
    return (bpr_loss, cl_loss, l2_loss), params_to_log
```

**Example: tracking a learnable scale parameter (CSAR-style):**

```python
params_to_log = {
    'scale':      self.attention_layer.scale.item(),
    'loss_main':  loss.item(),
    'loss_orth':  orth_loss.item(),
}
return (loss, self.lamda * orth_loss), params_to_log
```

**Example: tracking KL annealing schedule (MultiVAE-style):**

```python
anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)
params_to_log = {
    'nll':    nll_loss.item(),
    'kl':     kl_loss.item(),
    'anneal': anneal,
}
return (nll_loss, anneal * kl_loss), params_to_log
```

**Output files (in `trained_model/{dataset}/{model}/`):**

| File | Contents |
|---|---|
| `params_history.json` | `{key: [epoch_0_avg, epoch_1_avg, ...]}` for every tracked key |
| `params_plot.png` | Line plot of all tracked values over epochs |
| `loss_plot.png` | Per-component loss curves (one line per `loss_N` in the tuple) |
| `metrics_plot.png` | Validation metric curves over epochs |

If `calc_loss` returns `None` as the second element, no parameter tracking is performed.

**Step 2 — Register in the model registry:**

```python
# src/models/__init__.py
from .general.my_model import MyModel

MODEL_REGISTRY = {
    ...
    'my_model': MyModel,
}
```

**Step 3 — Create a config file:**

```yaml
# configs/model/general/my_model.yaml
model:
  name: "my_model"
  embedding_dim: 64

train:
  epochs: 200
  batch_size: 1024
  lr: 0.001
  optimizer: "adam"
  loss_type: "pairwise"
  num_negatives: 1
  early_stop_patience: 40
  embedding_l2: 1.0e-5

device: "auto"
```

**Step 4 — Run:**

```bash
uv run python scripts/main.py \
  --dataset_config configs/dataset/ml1m.yaml \
  --model_config configs/model/general/my_model.yaml
```

---

## Closed-form (Non-gradient) Models

If a model computes parameters analytically rather than via gradient descent, implement `fit()` instead of `calc_loss()`. The Trainer calls `fit()` before evaluation if it exists, and skips the gradient loop entirely if no `train:` block is present in the config.

```python
class MyClosedFormModel(BaseModel):
    def fit(self, data_loader):
        # Compute item similarity matrix, SVD, etc.
        ...

    def calc_loss(self, batch_data):
        # Return a zero loss — the trainer requires this to exist
        return (torch.tensor(0.0, device=self.device),), None
```

**Trainer execution flow:**

```
trainer.run()
  ├── model.fit() exists → call it once
  ├── config has 'train:' → run _train_loop() (gradient steps)
  └── evaluate()
```

---

## GPU Acceleration in Models

Use `self.device` (set by `BaseModel`) for all tensor allocation. For SVD or Gram matrix operations, use the utilities in `src/utils/gpu_accel.py`:

```python
from src.utils.gpu_accel import SVDCacheManager, gpu_gram_solve

# In __init__:
manager = SVDCacheManager(device=self.device.type)
self.register_cache_manager('svd', manager)

# In fit() or __init__:
u, s, v, total_energy = manager.get_svd(X_sparse, k=200, dataset_name=dataset_name)
```

See [GPU_ACCEL.md](GPU_ACCEL.md) for full documentation.

---

## Trainer Flow

```
Trainer.run()
  ├── model.fit()         (if defined)
  ├── _train_loop()       (if train config exists)
  │   ├── calc_loss() per batch
  │   ├── optimizer.step()
  │   └── evaluate() every N epochs (early stopping)
  └── final evaluate()
```

Metrics, losses, and model checkpoints are saved automatically to `trained_model/{dataset}/{model}/`.

---

## Loss Functions

`src/loss.py` implements:

| Class | Type | Description |
|---|---|---|
| `BPRLoss` | Pairwise | Standard BPR: `-log σ(s_pos - s_neg)` |
| `DynamicMarginBPRLoss` | Pairwise | BPR with adaptive margin scaled by score difference |
| `MSELoss` | Pointwise | Mean squared error for explicit feedback |
| `SampledSoftmaxLoss` | Listwise | InfoNCE with temperature and optional log-Q correction |

Usage inside a model:

```python
from src.loss import BPRLoss

class MyModel(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.loss_fn = BPRLoss()

    def calc_loss(self, batch_data):
        pos_scores = ...
        neg_scores = ...
        loss = self.loss_fn(pos_scores, neg_scores)
        reg = self.get_l2_reg_loss(self.user_emb.weight, self.item_emb.weight)
        return (loss, reg), {'bpr': loss.item()}
```

---

## Evaluation

Three evaluation protocols are supported, configurable per model or globally:

| Protocol | Description |
|---|---|
| `full` | Score all items, mask train interactions, rank |
| `sampled` | Evaluate on a random subset of users |
| `uni99` | 1 positive + 99 uniform random negatives |

Metrics: `NDCG`, `HitRate`, `Recall`, `Precision`, `Coverage`, `ILD`, `Novelty`, `GiniIndex`, `LongTailCoverage`, `LongTailRatio`.

---

## Data Pipeline

```
data_loader.py
  ├── _load_or_process_data()
  │   ├── cache hit  → _load_from_cache()
  │   └── cache miss → _process_data() → _save_to_cache()
  ├── get_train_loader()
  ├── get_validation_loader()
  └── get_interaction_graph()
```

Data cache key format: `{dataset}_{split_method}_rt{threshold}_mu{min_u}_mi{min_i}_tr{train_ratio}_vr{valid_ratio}_dedup{0|1}.pkl`

Changing any preprocessing parameter automatically creates a new cache file.

---

## Batch HPO

Run Bayesian hyperparameter optimization over multiple models and datasets:

```bash
uv run python scripts/run_all_smart_searches.py \
  --config configs/paper_baselines_search.yaml \
  --output_dir output/results
```

Search config structure:

```yaml
datasets:
  - "configs/dataset/ml1m.yaml"

seeds: [42, 43, 44]

searches:
  - name: "EASE"
    model_config: "configs/model/general/ease.yaml"
    params:
      - name: "model.reg_lambda"
        type: "float"
        range: "0.1 100000"
        log: true
    n_trials: 60
    metric: "NDCG@20"
```

Parameter types: `float`, `int`, `categorical`.

Output: `output/{dataset}/{model}/best_params.json`, `best_val_metrics.json`, per-seed model directories.
