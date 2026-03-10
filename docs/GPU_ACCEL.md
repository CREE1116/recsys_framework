# GPU Acceleration Utilities

`src/utils/gpu_accel.py` provides device-aware linear algebra operations for recommendation models. All functions automatically dispatch to CUDA, MPS, or CPU depending on what is available.

---

## Device Dispatch

The priority order is always:

```
CUDA → MPS (Apple Silicon) → CPU
```

The `get_device(preference='auto')` function implements this:

```python
from src.utils.gpu_accel import get_device

device = get_device('auto')   # returns torch.device('cuda'), 'mps', or 'cpu'
device = get_device('cpu')    # forces CPU
```

Most internal functions accept a `device='auto'` argument that follows this same logic.

---

## SVDCacheManager

Handles truncated SVD computation with disk-based caching. Multiple models sharing the same dataset will reuse the same cached SVD rather than recomputing it.

### Cache key

SVD results are stored as `data_cache/svd_{dataset}_{matrix_hash}_k{k}.pt`. The matrix hash is an MD5 over the matrix shape, nnz, and a sample of the data and index arrays. This ensures that different preprocessing configurations (rating threshold, k-core filtering, etc.) produce different cache files even if the dataset name is the same.

### Usage

```python
from src.utils.gpu_accel import SVDCacheManager

manager = SVDCacheManager(cache_dir='data_cache', device='auto')

# Compute or load cached SVD (k top singular values/vectors)
u, s, v, total_energy = manager.get_svd(X_sparse, k=200, dataset_name='ml-1m')

# Load from cache without recomputing (X_sparse can be None if cache exists)
u, s, v, total_energy = manager.get_svd(dataset_name='ml-1m', k=200)
```

### Parameters of `get_svd`

| Parameter | Description |
|---|---|
| `X_sparse` | scipy sparse matrix (CSR). Can be `None` if cache exists. |
| `k` | Number of singular values to compute. |
| `target_energy` | Alternative to `k`: keep enough components to capture this fraction of total energy. |
| `dataset_name` | Used in cache filename. |
| `force_recompute` | If `True`, ignore existing cache and recompute. |

### Cache reuse

If the cache contains a result for `k' > k`, it is truncated and returned immediately. This avoids recomputation when experimenting with different rank values.

### SVD backends

| Device | Condition | Backend |
|---|---|---|
| CUDA | `min(M, N) >= 5000` | Randomized SVD (Halko et al.) over native sparse CSR — `torch.sparse.mm` + `torch.linalg.qr` |
| MPS | `min(M, N) >= 5000` | Randomized SVD (batched) — avoids MPS QR instability via eigendecomposition |
| CPU / small matrices | always | `scipy.sparse.linalg.svds` (iterative) or `scipy.linalg.svd` (dense) |

The CUDA randomized SVD works as follows:

1. Build sparse CSR tensors on GPU from the scipy sparse matrix.
2. Sketch: `Y = X @ G` where `G` is a random Gaussian matrix.
3. Orthonormalize with `torch.linalg.qr`.
4. Power iterations to improve accuracy.
5. Project to low-dimensional space and run a small dense SVD with `torch.linalg.svd`.
6. Recover full-space singular vectors.

### Invalidating cache

```python
manager.invalidate()                        # remove all SVD cache files
manager.invalidate(key='ml-1m')             # remove only ml-1m entries
```

---

## GramEigenCacheManager

In-memory cache for eigendecomposition of Gram matrices (`X^T X`). Used by `gpu_gram_solve` to avoid recomputing the eigen factorization across multiple lambda values.

The cache key is a hash of the matrix structure (shape, nnz, data/index samples), so two different preprocessing results for the same dataset name will not collide.

This cache lives in Python process memory and is not persisted to disk.

```python
from src.utils.gpu_accel import GramEigenCacheManager

# Typically used indirectly via gpu_gram_solve
# Manual access (rarely needed):
result = GramEigenCacheManager.get(X_sparse)  # returns (V, eigvals) or None
GramEigenCacheManager.put(X_sparse, V, eigvals)
GramEigenCacheManager.clear()
```

---

## gpu_gram_solve

Computes `(X^T X + λI)^{-1} @ rhs` efficiently. Used by EASE, LIRA, and similar closed-form models.

```python
from src.utils.gpu_accel import gpu_gram_solve

# Compute inverse of Gram matrix (returns X^T X + λI)^{-1}
P = gpu_gram_solve(X_sparse, reg_lambda=500.0, device='auto')

# Solve a system: (X^T X + λI)^{-1} @ rhs
P = gpu_gram_solve(X_sparse, reg_lambda=500.0, rhs=rhs_np, device='auto')

# Return as a GPU tensor
P = gpu_gram_solve(X_sparse, reg_lambda=500.0, return_tensor=True, device='auto')
```

### Dispatch strategy

| Condition | Method |
|---|---|
| `M <= 15000` (first call) | Full eigendecomposition via `scipy.linalg.eigh`, cached for reuse with different λ |
| `M <= 15000` (subsequent calls) | Load from `GramEigenCache`, apply new λ instantly |
| `M > 15000` | Cholesky factorization (`gpu_cholesky_solve`) |

The eigen path is beneficial when the same dataset is evaluated with many λ values (HPO), since the factorization is computed once and only the diagonal scaling changes.

---

## gpu_cholesky_solve

Solves `G @ X = rhs` or computes `G^{-1}` via Cholesky factorization. Used as a fallback for large Gram matrices.

```python
from src.utils.gpu_accel import gpu_cholesky_solve

# Invert G (returns G^{-1} as numpy array)
G_inv = gpu_cholesky_solve(G_np, device='auto')

# Solve G @ X = rhs
X = gpu_cholesky_solve(G_np, rhs_np=rhs, device='auto')

# Return as GPU tensor
X = gpu_cholesky_solve(G_np, device='auto', return_tensor=True)
```

On CUDA or MPS: uses `torch.linalg.cholesky` + `torch.cholesky_solve`. On OOM, falls back to CPU.

On CPU: uses `scipy.linalg.cho_factor` with `overwrite_a=True` to minimize memory usage. For matrix inversion, solves in blocks to avoid allocating a large identity matrix at once.

---

## CacheRegistry

Models register cache managers with `BaseModel.register_cache_manager()`. The `Trainer` uses the `CacheRegistry` to log aggregate cache status at the end of a run:

```
[Cache] 3 entries, ~45 MB on disk
```

Only the total count and total disk/memory size are shown, not individual file paths or dataset names.

```python
# In a model's __init__:
self.register_cache_manager('svd', SVDCacheManager(device=self.device.type))
```

The registry calls `summary()` on each registered manager and aggregates the results.
