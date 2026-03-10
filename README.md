# RecSys Framework

A lightweight recommender systems research framework designed for fast prototyping and rigorous evaluation. It targets researchers who need to implement and test new models quickly without wrestling with framework boilerplate, and supports GPU acceleration on CUDA and Apple Silicon (MPS) out of the box.

---

## Overview

The framework provides:

- **Device-aware GPU acceleration** — `CUDA → MPS → CPU` automatic fallback for all expensive linear algebra operations (SVD, Gram matrix solve, Cholesky).
- **SVD and Gram eigen caching** — computed decompositions are stored on disk and reused across runs. Reloading a previously computed SVD is near-instant.
- **YAML-driven configuration** — a 3-tier config system (`evaluation → dataset → model`) controls every aspect of the pipeline without code changes.
- **Simple model interface** — `BaseModel` defines a minimal four-method contract. Closed-form models implement `fit()`; gradient-based models implement `calc_loss()`.
- **Batch HPO** — Optuna-based Bayesian hyperparameter search over multiple models and datasets, with multi-seed averaging.
- **Comprehensive evaluation** — full ranking and sampled evaluation, with accuracy, diversity, novelty, and long-tail metrics.

---

## Architecture

```
recsys_framework/
├── configs/                  # YAML configs (evaluation, dataset, model)
├── data/                     # Raw dataset files
├── data_cache/               # Preprocessed data and SVD/Gram cache (auto-managed)
├── docs/                     # Technical documentation
├── output/                   # HPO search results and evaluation logs
├── scripts/
│   ├── main.py               # Single-run entry point
│   └── run_all_smart_searches.py  # Batch HPO runner
└── src/
    ├── utils/
    │   ├── gpu_accel.py      # GPU-accelerated linear algebra (SVD, Cholesky, Gram)
    │   └── cache_manager.py  # Cache lifecycle management
    ├── models/
    │   ├── base_model.py     # Abstract base class
    │   ├── general/          # Baseline models (EASE, LightGCN, MF, ...)
    │   └── csar/             # LIRA and ASPIRE family
    ├── data_loader.py        # Data loading, filtering, and train/val/test split
    ├── evaluation.py         # Top-K ranking metrics
    ├── trainer.py            # Training and evaluation orchestration
    └── loss.py               # BPR, SampledSoftmax, MSE, DynamicMarginBPR
```

---

## Quick Start

### Install

```bash
uv pip install -r docs/requirements.txt
```

Python 3.12+ recommended. `uv` is preferred for environment management.

### Run a single experiment

```bash
uv run python scripts/main.py \
  --dataset_config configs/dataset/ml1m.yaml \
  --model_config configs/model/general/ease.yaml
```

### Run batch HPO

```bash
uv run python scripts/run_all_smart_searches.py \
  --config configs/paper_baselines_search.yaml \
  --output_dir output/paper_baselines
```

---

## Implemented Models

### Closed-form (no gradient descent)

| Model | Description |
|---|---|
| `ease` | EASE — item-item weight matrix via closed-form ridge regression |
| `lira` | LIRA — dual ridge regression via user-user Gram solve |
| `light_lira` | LightLIRA — SVD-based spectral approximation of LIRA |
| `aspire` | ASPIRE — popularity-debiased item similarity (MNAR correction) |
| `item_knn` | Item-based cosine KNN |
| `most_popular` | Popularity baseline |
| `pure_svd` | Truncated SVD recommendation |
| `svd_ease` | SVD-approximated EASE |
| `slim` | SLIM — sparse linear model |

### Gradient-based

| Model | Description |
|---|---|
| `mf` | Matrix Factorization with BPR/Softmax loss |
| `neumf` | Neural MF (GMF + MLP) |
| `lightgcn` | LightGCN — graph convolution collaborative filtering |
| `multivae` | Multinomial VAE for implicit feedback |
| `protomf` | Prototype-based MF with orthogonality regularization |
| `ultragcn` | UltraGCN — constraint-based graph CF |

---

## Documentation

- **[Developer Guide](docs/DEVELOPER_GUIDE.md)** — How to add a model, BaseModel interface, Trainer flow
- **[Configuration Guide](docs/CONFIG.md)** — YAML config system, all parameters
- **[GPU Acceleration](docs/GPU_ACCEL.md)** — SVDCacheManager, Cholesky solver, Gram matrix solver, device dispatch
- **[Evaluation Protocol](docs/EVALUATION_PROTOCOL.md)** — Data splits, negative sampling, metrics
- **[Models Reference](docs/general_models_summary.md)** — Implemented model descriptions
- **[Loss Functions](docs/loss_functions_summary.md)** — BPR, SampledSoftmax, MSE

---

## Device Support

All computationally intensive operations dispatch automatically:

```
CUDA (if available) → MPS (Apple Silicon, if available) → CPU
```

This applies to SVD computation, Cholesky factorization, and Gram matrix eigendecomposition. Large-matrix SVD on CUDA uses a native randomized algorithm over sparse CSR tensors (`torch.sparse.mm` + `torch.linalg.qr`). MPS uses a batched version that works around MPS QR instability.
