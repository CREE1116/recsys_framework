# Models Reference

This document describes the models implemented in `src/models/`.

---

## Closed-form Models

### EASE

**File:** `src/models/general/ease.py`

Embarrassingly Shallow Autoencoder. Computes an item-item weight matrix `B` as a closed-form ridge regression solution:

```
B = (X^T X + λI)^{-1} X^T X,  diag(B) = 0
```

Scores are `X @ B`. No training epochs. Fast and surprisingly competitive on sparse datasets. Uses `gpu_gram_solve` for GPU-accelerated computation.

### LIRA

**File:** `src/models/csar/LIRA.py`, layer: `src/models/csar/LIRALayer.py::LIRALayer`

Linear Interest covariance Ridge Analysis. Computes a dual ridge regression via user-user Gram matrix:

```
K = X X^T,  CX = (K + λI)^{-1} X,  S = X^T CX
```

Scores are `X @ S`. Dense computation; suitable for small-to-medium datasets.

### LightLIRA

**File:** `src/models/csar/LightLIRA.py`, layer: `LIRALayer.py::LightLIRALayer`

SVD-based spectral approximation of LIRA. Avoids the `n_users x n_users` matrix inversion by working in the low-rank SVD subspace:

```
filter = σ^2 / (σ^2 + λ),  score = (X V) * filter @ V^T
```

O(nk) inference. Suitable for large datasets. Uses `SVDCacheManager` for SVD caching.

### ASPIRE

**File:** `src/models/csar/ASPIRE.py`, layer: `ASPIRELayer.py::ASPIRELayer`

Popularity-debiased item similarity model with MNAR (Missing Not At Random) correction. Estimates exposure bias parameters and applies a Gamma correction to the item interaction matrix before computing similarities. Uses SVD-based low-rank approximation.

### ItemKNN

**File:** `src/models/general/item_knn.py`

Memory-based collaborative filtering. Precomputes a cosine item-item similarity matrix and scores items based on the user's interaction history weighted by similarity. Simple but often competitive.

### Most Popular

**File:** `src/models/general/most_popular.py`

Non-personalized baseline. Recommends the globally most interacted items to all users. Used as a lower-bound reference.

### Pure SVD

**File:** `src/models/general/pure_svd.py`

Recommendation via truncated SVD of the interaction matrix. Projects users into the latent space and ranks items by projection score.

### SVD-EASE

**File:** `src/models/general/svd_ease.py`

EASE approximated using SVD: applies the spectral filter in the low-rank SVD subspace instead of solving the full Gram matrix. Scales better than EASE for large item sets.

### SLIM

**File:** `src/models/general/slim.py`

Sparse Linear Method. Learns a sparse item-item weight matrix via coordinate descent with L1+L2 regularization. Can be slow to compute but produces interpretable sparse weights.

### GF-CF

**File:** `src/models/general/gf_cf.py`

Graph-based closed-form recommendation. Applies a graph frequency filter to the interaction matrix using SVD.

---

## Gradient-based Models

### MF (Matrix Factorization)

**File:** `src/models/general/mf.py`

User and item embeddings trained with BPR or SampledSoftmax loss. Prediction: `u · i^T`. The baseline for latent factor models.

### NeuMF

**File:** `src/models/general/neumf.py`

NCF-style model combining GMF (generalized MF) and MLP paths. Each path has its own embeddings; outputs are concatenated before the final prediction layer.

### LightGCN

**File:** `src/models/general/lightgcn.py`

Graph convolution collaborative filtering without nonlinear activation or feature transformation. Aggregates neighbor embeddings via normalized weighted sum across L layers. Strong baseline for graph-based recommendation.

### Multi-VAE

**File:** `src/models/general/multivae.py`

Variational autoencoder trained with multinomial likelihood. Input is the full user interaction history (bag-of-words); the model reconstructs it via a stochastic latent representation. Performs well on datasets with dense interaction histories.

### ProtoMF

**File:** `src/models/general/protomf.py`

Prototype-based MF. Users and items are represented as weighted combinations of K shared prototype vectors with orthogonality regularization. Provides some interpretability via prototype assignments.

### UltraGCN

**File:** `src/models/general/ultragcn.py`

Constraint-based graph CF that approximates infinite-layer graph convolution. Uses a precomputed constraint matrix over the user-item graph to regularize the embedding learning.

### SimGCL

**File:** `src/models/general/simgcl.py`

Graph contrastive learning model that generates augmented views by adding uniform noise to graph embeddings and applies InfoNCE to align them.

---

## Loss Functions

See [loss_functions_summary.md](loss_functions_summary.md) for details on `BPRLoss`, `SampledSoftmaxLoss`, `MSELoss`, and `DynamicMarginBPRLoss`.
