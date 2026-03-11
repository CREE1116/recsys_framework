"""
GPU-accelerated linear algebra utilities for recommendation models.
MPS (Apple Silicon) / CUDA 가속 + CPU fallback.

통합 유틸리티:
- SVDCacheManager: SVD 캐싱 + MPS randomized SVD
- gpu_cholesky_solve: GPU-accelerated Cholesky solver
- gpu_gram_solve: EASE-family Gram matrix solver
"""
import torch
import numpy as np
import os
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from .cache_manager import GlobalCacheManager


# ============================================================
# Device Detection
# ============================================================

def get_device(preference='auto'):
    """Get the best available device."""
    if preference == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    if isinstance(preference, str):
        return torch.device(preference)
    return preference


# ============================================================
# GPU-Accelerated Cholesky Solver
# ============================================================

def gpu_cholesky_solve(G_np, rhs_np=None, device='auto', return_tensor=False):
    """
    Solve G @ X = rhs via Cholesky.
    Args:
        G_np: (M, M) numpy array
        rhs_np: (M, K) numpy array
        return_tensor: If True, returns torch.Tensor on device.
    """
    dev = get_device(device)
    M = G_np.shape[0]
    
    if dev.type in ('mps', 'cuda'):
        try:
            t0 = time.time()
            G_t = torch.from_numpy(G_np).float().to(dev)
            L = torch.linalg.cholesky(G_t)
            if rhs_np is None:
                I = torch.eye(M, device=dev, dtype=torch.float32)
                X_t = torch.cholesky_solve(I, L)
                del I
            else:
                rhs_t = torch.from_numpy(rhs_np).float().to(dev)
                X_t = torch.cholesky_solve(rhs_t, L)
            del L, G_t
            
            if return_tensor:
                print(f"[gpu_accel] {dev.type.upper()} Cholesky Solve ({M}x{M}) [Tensor]: {time.time()-t0:.2f}s")
                return X_t
                
            result = X_t.cpu().numpy()
            print(f"[gpu_accel] {dev.type.upper()} Cholesky Solve ({M}x{M}) [NumPy]: {time.time()-t0:.2f}s")
            return result
        except RuntimeError as e:
            print(f"[gpu_accel] {dev.type.upper()} cholesky OOM ({e}), CPU fallback...")
    
    res_np = _cpu_cholesky(G_np, rhs_np)
    return torch.from_numpy(res_np).float().to(dev) if return_tensor else res_np


def _cpu_cholesky(G_np, rhs_np, block_size=2000):
    """
    CPU Cholesky solve via scipy.
    [Optimization] Use overwrite_a=True and block-wise solve for identity to save memory.
    """
    import gc
    t0 = time.time()
    M = G_np.shape[0]
    from scipy.linalg import cho_factor, cho_solve
    
    # Factorize in-place to save memory (M x M x 4 bytes)
    cho, low = cho_factor(G_np, lower=True, overwrite_a=True, check_finite=False)

    if rhs_np is None:
        result = np.empty((M, M), dtype=np.float32)
        for i in range(0, M, block_size):
            end = min(i + block_size, M)
            # Create a localized block of identity matrix
            rhs_block = np.zeros((M, end - i), dtype=np.float32)
            for j in range(end - i):
                rhs_block[i + j, j] = 1.0
            
            result[:, i:end] = cho_solve((cho, low), rhs_block, check_finite=False)
            del rhs_block
            gc.collect() # Force garbage collection of old blocks
        print(f"[gpu_accel] CPU Cholesky inversion ({M}x{M}): {time.time()-t0:.2f}s")
    else:
        result = cho_solve((cho, low), rhs_np, check_finite=False)
        print(f"[gpu_accel] CPU Cholesky Solve ({M}x{rhs_np.shape[1]}): {time.time()-t0:.2f}s")
    
    return result


def gpu_gram_solve(X_sparse, reg_lambda, rhs=None, device='auto', return_tensor=False):
    """
    Compute (X^T X + λI)^-1 @ rhs.
    """
    M = X_sparse.shape[1]
    EIGEN_THRESHOLD = 15000
    dev = get_device(device)
    
    # 1. Check eigen cache
    cache = _GramEigenCache.get(X_sparse)
    if cache is not None:
        V_np, eigvals_np = cache
        t0 = time.time()
        if return_tensor and dev.type in ('cuda', 'mps'):
            V = torch.from_numpy(V_np).float().to(dev)
            eigvals = torch.from_numpy(eigvals_np).float().to(dev)
            inv_eig = 1.0 / (eigvals + reg_lambda)
            if rhs is None:
                P = (V * inv_eig.unsqueeze(0)) @ V.t()
            else:
                rhs_t = torch.from_numpy(rhs).float().to(dev) if isinstance(rhs, np.ndarray) else rhs.to(dev)
                P = (V * inv_eig.unsqueeze(0)) @ (V.t() @ rhs_t)
            print(f"[gpu_accel] Eigen solve [Tensor] on {dev}: {time.time()-t0:.2f}s")
            return P
        else:
            inv_eigvals = (1.0 / (eigvals_np + reg_lambda)).astype(np.float32)
            if rhs is None:
                P = (V_np * inv_eigvals[None, :]) @ V_np.T
            else:
                P = (V_np * inv_eigvals[None, :]) @ (V_np.T @ rhs.astype(np.float32))
            print(f"[gpu_accel] Eigen solve [NumPy] on CPU: {time.time()-t0:.2f}s")
            return torch.from_numpy(P).float().to(dev) if return_tensor else P

    # 2. Eigen Path
    if M <= EIGEN_THRESHOLD:
        print(f"[gpu_accel] Gram ({M}x{M}) eigendecomposition (first call) on {dev.type}...")
        t0 = time.time()
        
        # Optimization: compute Gram matrix directly.
        # If CUDA/MPS, do it on device to massively speed up X^T * X and Eigendecomposition.
        if dev.type in ('cuda', 'mps'):
            try:
                if isinstance(X_sparse, csr_matrix):
                    # For sparse, we can convert to dense if small enough, or use sparse mm.
                    # Since M <= 15000, M*M is up to 225M floats (~900MB), which easily fits.
                    X_torch_dense = torch.from_numpy(X_sparse.toarray()).float().to(dev)
                    G_t = torch.mm(X_torch_dense.t(), X_torch_dense)
                    del X_torch_dense
                else:
                    # Fallback if it's already a tensor or numpy
                    X_torch = torch.tensor(X_sparse, device=dev, dtype=torch.float32)
                    G_t = torch.mm(X_torch.t(), X_torch)
                    
                eigvals_t, V_t = torch.linalg.eigh(G_t)
                del G_t
                
                # Cache as NumPy to save VRAM and keep cache manager agnostic
                eigvals = eigvals_t.cpu().numpy()
                V = V_t.cpu().numpy()
                del eigvals_t, V_t
                
                print(f"[gpu_accel] {dev.type.upper()} torch.linalg.eigh done: {time.time()-t0:.2f}s")
            except Exception as e:
                print(f"[gpu_accel] {dev.type.upper()} Eigen failed ({e}), fallback to Scipy CPU...")
                G = (X_sparse.T @ X_sparse).toarray().astype(np.float32)
                from scipy.linalg import eigh
                eigvals, V = eigh(G)
                del G
        else:
            # CPU Path
            G = (X_sparse.T @ X_sparse).toarray().astype(np.float32)
            from scipy.linalg import eigh
            eigvals, V = eigh(G)
            del G
            
        _GramEigenCache.put(X_sparse, V.astype(np.float32), eigvals.astype(np.float32))
        return gpu_gram_solve(X_sparse, reg_lambda, rhs, device, return_tensor)

    # 3. Cholesky Path
    print(f"[gpu_accel] Gram ({M}x{M}) + Exact Cholesky on {dev.type}...")
    t0 = time.time()
    
    if dev.type in ('cuda', 'mps'):
        try:
            if isinstance(X_sparse, csr_matrix):
                X_torch_dense = torch.from_numpy(X_sparse.toarray()).float().to(dev)
                G_t = torch.mm(X_torch_dense.t(), X_torch_dense)
                del X_torch_dense
            
            # Diagonal shift for torch tensor
            G_t.diagonal().add_(reg_lambda)
            P = gpu_cholesky_solve(G_t.cpu().numpy(), rhs, device=device, return_tensor=return_tensor)
            del G_t
            return P
        except Exception as e:
            print(f"[gpu_accel] {dev.type.upper()} Gram+Cholesky prep failed ({e}), fallback to CPU...")

    # Fallback / CPU
    G = (X_sparse.T @ X_sparse).toarray().astype(np.float32)
    G[np.diag_indices(M)] += reg_lambda
    
    P = gpu_cholesky_solve(G, rhs, device=device, return_tensor=return_tensor)
    del G
    return P


class GramEigenCacheManager(GlobalCacheManager):
    """
    Module-level cache for Gram matrix eigendecomposition. Global scope.
    Keyed by (shape, nnz, data_checksum) to identify the same dataset.
    """
    _cache = {}  # key -> (V, eigvals)
    
    @classmethod
    def _key(cls, X_sparse):
        d = X_sparse.data
        idx = X_sparse.indices
        ptr = X_sparse.indptr
        checksum = hash((X_sparse.shape, X_sparse.nnz, 
                        d[:10].tobytes() if len(d) >= 10 else d.tobytes(),
                        idx[:10].tobytes() if len(idx) >= 10 else idx.tobytes(),
                        ptr[:10].tobytes() if len(ptr) >= 10 else ptr.tobytes()))
        return checksum
    
    @classmethod
    def get(cls, X_sparse):
        key = cls._key(X_sparse)
        return cls._cache.get(key)
    
    @classmethod
    def put(cls, X_sparse, V, eigvals):
        key = cls._key(X_sparse)
        cls._cache[key] = (V, eigvals)
        print(f"[gpu_accel] Gram eigen cached (M={V.shape[0]}, {V.nbytes/1024**2:.0f} MB)")
    
    @classmethod
    def clear(cls):
        cls._cache.clear()

    # --- CacheManager Interface ---
    def summary(self):
        total_bytes = sum(v.nbytes + e.nbytes for v, e in self._cache.values()) if self._cache else 0
        return {"type": "GramEigen", "entries": len(self._cache), "size_mb": round(total_bytes / 1e6, 1)}

    def invalidate(self, key=None):
        self._cache.clear()

# Backward compat
_GramEigenCache = GramEigenCacheManager




# ============================================================
# SVD Cache Manager
# ============================================================



class SVDCacheManager(GlobalCacheManager):
    """SVD 결과를 캐싱하고 MPS 가속을 제공하는 관리자 클래스 (Global Scope)"""
    
    def __init__(self, cache_dir='data_cache', device='auto'):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.device = get_device(device)
        print(f"[SVD-Manager] Device: {self.device}")

    # --- CacheManager Interface ---
    def summary(self):
        import glob as _glob
        files = _glob.glob(os.path.join(self.cache_dir, "svd_*.pt"))
        total_size = sum(os.path.getsize(f) for f in files) if files else 0
        return {"type": "SVD", "files": len(files), "size_mb": round(total_size / 1e6, 1)}

    def invalidate(self, key=None):
        import glob as _glob
        if key:
            pattern = os.path.join(self.cache_dir, f"{key}*.pt")
        else:
            pattern = os.path.join(self.cache_dir, "svd_*.pt")
        for f in _glob.glob(pattern):
            os.remove(f)
            print(f"[SVD-Manager] Invalidated: {os.path.basename(f)}")


    @staticmethod
    def _generate_matrix_id(X_sparse):
        """행렬의 구조와 데이터를 기반으로 고유한 ID(해시) 생성"""
        if not hasattr(X_sparse, 'shape'): return "unknown"
        
        # Shape, nnz
        meta = (X_sparse.shape, X_sparse.nnz)
        
        # Sample data for hashing (first 100 values to be fast yet robust)
        d = X_sparse.data[:100].tobytes() if len(X_sparse.data) >= 100 else X_sparse.data.tobytes()
        idx = X_sparse.indices[:100].tobytes() if len(X_sparse.indices) >= 100 else X_sparse.indices.tobytes()
        
        import hashlib
        h = hashlib.md5()
        h.update(str(meta).encode())
        h.update(d)
        h.update(idx)
        return h.hexdigest()[:12]

    @staticmethod
    def get_analysis_dir(config):
        """config 기반 분석 디렉토리 경로 생성"""
        dataset_name = config.get('dataset_name', 'unknown')
        model_name = config['model']['name']
        run_name = config.get('run_name', 'default')
        
        if run_name and run_name != 'default':
            folder = f"{model_name}__{run_name}"
        else:
            folder = model_name
        return os.path.join('trained_model', dataset_name, folder, 'analysis')

    def get_svd(self, X_sparse=None, k=None, target_energy=None, dataset_name=None, force_recompute=False):
        """
        캐시에서 SVD 로드 또는 새로 계산 (최적화 버전).
        - 행렬 해시를 기반으로 모델 간 캐시를 공유합니다.
        - X_sparse가 None일 경우 dataset_name을 기반으로 최신 캐시를 찾습니다.
        - 요청된 k보다 큰 캐시가 있다면 잘라서 재사용(Truncate)합니다.
        """
        if X_sparse is None and dataset_name is None:
             raise ValueError("Either X_sparse or dataset_name must be provided to get_svd")

        if X_sparse is not None:
            M, N = X_sparse.shape
            if M == 0 or N == 0:
                return torch.empty(M, 0), torch.empty(0), torch.empty(N, 0), 0.0
            matrix_id = self._generate_matrix_id(X_sparse)
        else:
            matrix_id = "*"  # Wildcard for glob
        
        dataset_name = dataset_name or "unknown"
        
        # 1. Cache Search and Reuse Logic
        if not force_recompute:
            import glob
            # Pattern: svd_{dataset}_{matrix_id}_k*.pt
            pattern = os.path.join(self.cache_dir, f"svd_{dataset_name}_{matrix_id}_k*.pt")
            cache_files = glob.glob(pattern)
            
            if cache_files:
                # Find best candidate: smallest k that is >= requested k
                candidates = []
                for f in cache_files:
                    try:
                        f_k = int(f.split("_k")[-1].replace(".pt", ""))
                        candidates.append((f_k, f))
                    except ValueError: continue
                
                candidates.sort() # Small k first
                
                best_file = None
                if k is not None:
                    for f_k, f_path in candidates:
                        if f_k >= k:
                            best_file = (f_k, f_path)
                            break
                elif target_energy is not None:
                    # For target_energy, we need the largest available or any that satisfies it
                    # Just pick the largest one to be safe
                    best_file = candidates[-1]

                if best_file:
                    f_k, f_path = best_file
                    print(f"[SVD] Cache hit (k={f_k}, dataset={dataset_name})")
                    cp = torch.load(f_path, map_location='cpu')
                    u, s, v = cp['u'], cp['s'], cp['v']
                    
                    if X_sparse is not None:
                        total_energy = cp.get('total_energy', float(np.sum(X_sparse.data ** 2)))
                    else:
                        total_energy = cp.get('total_energy', 1.0) # Fallback if missing
                    
                    # Truncate if k is specified
                    if k is not None and len(s) > k:
                        print(f"[SVD] Truncating: k={len(s)} -> k={k}")
                        u, s, v = u[:, :k], s[:k], v[:, :k]
                    
                    # Check energy if target_energy is specified
                    if target_energy is not None:
                        s2 = s ** 2
                        cum_energy = torch.cumsum(s2, dim=0) / total_energy
                        mask = (cum_energy >= target_energy).nonzero()
                        if len(mask) > 0:
                            final_k = mask[0].item() + 1
                            print(f"[SVD] Cache satisfied target energy (needs k={final_k})")
                            return u[:, :final_k], s[:final_k], v[:, :final_k], total_energy
                        else:
                            if k is None:
                                print(f"[SVD] Cache k={len(s)} insufficient for target energy. Recomputing...")
                            else:
                                return u, s, v, total_energy # Return what we have if k was also fixed
                    
                    # [Fixed] Always return if we found a best_file and it wasn't filtered out by energy check
                    return u, s, v, total_energy
                
        # 2. Computation needed (with Energy Target Loop)
        if X_sparse is None:
            raise RuntimeError(f"SVD Cache NOT found for {dataset_name} and X_sparse is None. Cannot compute.")

        # 시작 사이즈를 크게 상향 (기존 min_dim // 10 → min_dim // 4)
        compute_k = k or (min(M, N) // 4 if target_energy else 256)
        compute_k = min(compute_k, min(M, N) - 1)
        total_energy = float(np.sum(X_sparse.data ** 2))
        
        while True:
            print(f"[SVD] Computing (shape={X_sparse.shape}, k={compute_k}, dataset={dataset_name})...")
            start = time.time()
            min_dim = min(M, N)
            if self.device.type == 'cuda' and min_dim >= 2000:
                u, s, v = self._cuda_randomized_svd(X_sparse, compute_k)
            elif self.device.type == 'mps' and min_dim >= 2000:
                u, s, v = self._mps_randomized_svd(X_sparse, compute_k)
            else:
                u, s, v = self._cpu_svd(X_sparse, compute_k)
            print(f"[SVD] Done ({time.time()-start:.2f}s, k={compute_k})")

            # Check if energy target is met (only when target_energy is specified and k was not fixed)
            if target_energy is not None and k is None:
                s2 = s ** 2
                cum_energy = torch.cumsum(s2, dim=0) / total_energy
                # If we've reached the target or max possible rank, break and assign final_k
                mask = (cum_energy >= target_energy).nonzero()
                if len(mask) > 0:
                    final_k = mask[0].item() + 1
                    print(f"[SVD] Target energy met, k={final_k}")
                    u, s, v = u[:, :final_k], s[:final_k], v[:, :final_k]
                    break
                elif compute_k >= min_dim - 1:
                    print(f"[SVD] Reached max rank k={compute_k}, energy={cum_energy[-1].item()*100:.1f}%")
                    break
                else:
                    # 스텝 점프도 기존 * 2 에서 더 급격하게 키움 (* 3)
                    new_k = min(min_dim - 1, int(compute_k * 3))
                    print(f"[SVD] Energy not met ({cum_energy[-1].item()*100:.1f}%), expanding k: {compute_k} -> {new_k}")
                    compute_k = new_k
            else:
                # Target energy not tracked or k is manually fixed
                break
            
            
        # 3. Save Cache
        save_k = len(s)
        cache_path = os.path.join(self.cache_dir, f"svd_{dataset_name}_{matrix_id}_k{save_k}.pt")
        torch.save({'u': u, 's': s, 'v': v, 'total_energy': total_energy}, cache_path)
            
        return u, s, v, total_energy

    def _cpu_svd(self, X_sparse, k):
        """CPU SVD: Dense for small, Sparse iterative for large."""
        min_dim = min(X_sparse.shape)
        
        if min_dim < 2000:
            print(f"[CPU-SVD] Dense SVD ({X_sparse.shape})...")
            from scipy.linalg import svd
            X_dense = X_sparse.toarray()
            u, s, vt = svd(X_dense, full_matrices=False)
            u, s, vt = u[:, :k], s[:k], vt[:k, :]
        else:
            print(f"[CPU-SVD] Sparse iterative SVDS ({X_sparse.shape})...")
            u, s, vt = svds(X_sparse, k=k)
            idx = np.argsort(s)[::-1]
            s, u, vt = s[idx], u[:, idx], vt[idx, :]
        
        return (torch.from_numpy(u.copy()).float(),
                torch.from_numpy(s.copy()).float(),
                torch.from_numpy(vt.T.copy()).float())

    def _orthonormalize(self, Y, device):
        """Orthonormalize columns of Y using Eigen decomposition."""
        if Y.shape[1] == 0:
            return Y
        C = torch.mm(Y.t(), Y)
        try:
            C_cpu = C.cpu()
            S, V = torch.linalg.eigh(C_cpu)
            S_inv_sqrt = torch.where(S > 1e-12, 1.0 / torch.sqrt(S), torch.zeros_like(S))
            V, S_inv_sqrt = V.to(device), S_inv_sqrt.to(device)
            return torch.mm(Y, V * S_inv_sqrt.unsqueeze(0))
        except Exception as e:
            print(f"[SVD-Manager] Orthonormalization failed ({e}), simple scaling fallback")
            return Y / Y.norm(dim=0, keepdim=True).clamp(min=1e-12)

    @torch.no_grad()
    def _cuda_randomized_svd(self, X_sparse, k, n_iter=2, oversampling=10):
        """CUDA-accelerated Randomized SVD (Halko et al., 2011) using native sparse CSR.

        Unlike the MPS path (which batches due to memory/op constraints), CUDA supports
        torch.sparse.mm on CSR tensors natively and efficiently, so no batching is needed.
        Uses torch.linalg.qr for orthonormalization (stable on CUDA).
        """
        device = torch.device("cuda")
        M, N = X_sparse.shape
        q = min(k + oversampling, M, N)
        if q < 1:
            q = 1

        print(f"[CUDA-SVD] k={k}, q={q}, n_iter={n_iter}")

        # Build CUDA sparse CSR tensors from scipy sparse
        X_coo = X_sparse.tocoo()
        indices = torch.from_numpy(np.vstack((X_coo.row, X_coo.col))).long()
        values = torch.from_numpy(X_coo.data.copy()).float()
        X_t = torch.sparse_coo_tensor(
            indices, values, (M, N), device=device
        ).coalesce().to_sparse_csr()
        Xt_t = torch.sparse_coo_tensor(
            torch.stack([indices[1], indices[0]]), values,
            (N, M), device=device
        ).coalesce().to_sparse_csr()

        # Phase 1: Random sketch
        G = torch.randn(N, q, device=device, dtype=torch.float32)
        Y = torch.sparse.mm(X_t, G)        # (M, q)
        Q, _ = torch.linalg.qr(Y)          # (M, q), orthonormal columns

        # Phase 2: Power iterations (improves approximation quality)
        for i in range(n_iter):
            Z = torch.sparse.mm(Xt_t, Q)   # (N, q)
            Q_z, _ = torch.linalg.qr(Z)    # (N, q)
            Y = torch.sparse.mm(X_t, Q_z)  # (M, q)
            Q, _ = torch.linalg.qr(Y)      # (M, q)

        # Phase 3: Project into low-dimensional space
        B = torch.sparse.mm(Xt_t, Q).t()   # (q, N)

        # Phase 4: Small dense SVD on projected matrix
        U_hat, S_vals, Vh = torch.linalg.svd(B, full_matrices=False)  # (q,q), (q,), (q,N)

        # Recover full-space singular vectors
        U = torch.mm(Q, U_hat)             # (M, q)
        V = Vh.t()                         # (N, q)

        U, S_vals, V = U[:, :k].cpu(), S_vals[:k].cpu(), V[:, :k].cpu()
        print(f"[CUDA-SVD] Done! σ range: [{S_vals[-1]:.4f}, {S_vals[0]:.4f}]")
        return U, S_vals, V

    @torch.no_grad()
    def _mps_randomized_svd(self, X_sparse, k, n_iter=2, oversampling=10, batch_size=2000):
        """MPS-accelerated Randomized SVD (Halko et al., 2011)"""
        device = torch.device("mps")
        M, N = X_sparse.shape
        q = min(k + oversampling, M, N)
        if q < 1: q = 1
        
        print(f"[MPS-SVD] k={k}, q={q}, n_iter={n_iter}, batch={batch_size}")
        
        # Phase 1: Sketching + Power Iteration
        G = torch.randn(N, q, device=device, dtype=torch.float32)
        Y = self._sparse_mm_batched(X_sparse, G, batch_size, device)
        Q = self._orthonormalize(Y, device)
        
        for i in range(n_iter):
            Z = self._sparse_mm_transposed_batched(X_sparse, Q, batch_size, device)
            Q_z = self._orthonormalize(Z, device)
            Y = self._sparse_mm_batched(X_sparse, Q_z, batch_size, device)
            Q = self._orthonormalize(Y, device)
            del Z, Q_z
        
        # Phase 2: Projection (B = Q^T @ A)
        B = self._sparse_mm_transposed_batched(X_sparse, Q, batch_size, device).t()

        # Phase 3: Small SVD via eigen trick
        C = torch.mm(B, B.t())
        try:
            S2, U_hat = torch.linalg.eigh(C.cpu())
            idx = torch.argsort(S2, descending=True)
            S2, U_hat = S2[idx], U_hat[:, idx]
            S_vals = torch.sqrt(torch.clamp(S2, min=0.0))
        except Exception:
            U_hat, S_vals, _ = torch.linalg.svd(C.cpu(), full_matrices=False)

        U_hat, S_vals = U_hat.to(device), S_vals.to(device)
        
        # Recover V and U
        S_inv = torch.where(S_vals > 1e-12, 1.0 / S_vals, torch.zeros_like(S_vals))
        V = torch.mm(B.t(), U_hat) * S_inv
        U = torch.mm(Q, U_hat)
        
        U, S_vals, V = U[:, :k].cpu(), S_vals[:k].cpu(), V[:, :k].cpu()
        print(f"[MPS-SVD] Done! σ range: [{S_vals[-1]:.4f}, {S_vals[0]:.4f}]")
        return U, S_vals, V

    @staticmethod
    def _sparse_mm_batched(X_sparse, Y_dense, batch_size, device):
        """Batched X_sparse @ Y_dense (memory efficient)."""
        M = X_sparse.shape[0]
        q = Y_dense.shape[1]
        result = torch.zeros(M, q, device=device)
        if Y_dense.device.type != device:
            Y_dense = Y_dense.to(device)
        for i in range(0, M, batch_size):
            end = min(i + batch_size, M)
            A_batch = torch.from_numpy(X_sparse[i:end].toarray()).float().to(device)
            result[i:end] = torch.mm(A_batch, Y_dense)
            del A_batch
        return result
    
    @staticmethod
    def _sparse_mm_transposed_batched(X_sparse, Y_dense, batch_size, device):
        """Batched X_sparse^T @ Y_dense."""
        M, N = X_sparse.shape
        q = Y_dense.shape[1]
        result = torch.zeros(N, q, device=device)
        if Y_dense.device.type != device:
            Y_dense = Y_dense.to(device)
        for i in range(0, M, batch_size):
            end = min(i + batch_size, M)
            A_batch = torch.from_numpy(X_sparse[i:end].toarray()).float().to(device)
            result += torch.mm(A_batch.t(), Y_dense[i:end])
            del A_batch
        return result

    def clear_cache(self, dataset_name=None):
        """캐시 삭제"""
        if dataset_name:
            import glob
            for f in glob.glob(os.path.join(self.cache_dir, f"svd_{dataset_name}_*.pt")):
                os.remove(f)
                print(f"[SVD-Manager] Deleted: {f}")
        else:
            import shutil
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir)
            print(f"[SVD-Manager] All cache cleared")
