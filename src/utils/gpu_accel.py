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
                print(f"[gpu_accel] {dev.upper()} Cholesky Solve ({M}x{M}) [Tensor]: {time.time()-t0:.2f}s")
                return X_t
                
            result = X_t.cpu().numpy()
            print(f"[gpu_accel] {dev.upper()} Cholesky Solve ({M}x{M}) [NumPy]: {time.time()-t0:.2f}s")
            return result
        except RuntimeError as e:
            print(f"[gpu_accel] {dev.upper()} cholesky OOM ({e}), CPU fallback...")
    
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
    print(f"[gpu_accel] Factorizing {M}x{M} matrix in-place...")
    cho, low = cho_factor(G_np, lower=True, overwrite_a=True, check_finite=False)
    
    if rhs_np is None:
        # Memory-efficient inversion: avoid creating large identity matrix
        print(f"[gpu_accel] Pre-allocating result matrix ({M}x{M})...")
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
        print(f"[gpu_accel] CPU Cholesky Block-wise Inversion ({M}x{M}): {time.time()-t0:.2f}s")
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
        print(f"[gpu_accel] Gram ({M}x{M}) eigendecomposition (first call)...")
        # Optimization: compute Gram matrix directly if small
        G = (X_sparse.T @ X_sparse).toarray().astype(np.float32)
        from scipy.linalg import eigh
        eigvals, V = eigh(G)
        del G
        _GramEigenCache.put(X_sparse, V.astype(np.float32), eigvals.astype(np.float32))
        return gpu_gram_solve(X_sparse, reg_lambda, rhs, device, return_tensor)

    # 3. Cholesky Path
    print(f"[gpu_accel] Gram ({M}x{M}) + Exact Cholesky...")
    t0 = time.time()
    G = (X_sparse.T @ X_sparse).toarray().astype(np.float32)
    G[np.diag_indices(M)] += reg_lambda
    
    P = gpu_cholesky_solve(G, rhs, device=device, return_tensor=return_tensor)
    del G
    return P


class _GramEigenCache:
    """
    Module-level cache for Gram matrix eigendecomposition.
    Keyed by (shape, nnz, data_checksum) to identify the same dataset.
    """
    _cache = {}  # key -> (V, eigvals)
    
    @classmethod
    def _key(cls, X_sparse):
        # Inclusion of indices and indptr ensures uniqueness across different data splits
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
        print(f"[gpu_accel] Eigen cache stored ({V.shape[0]}x{V.shape[0]}, {V.nbytes/1024**3:.1f}GB)")
    
    @classmethod
    def clear(cls):
        cls._cache.clear()



class _TruncatedSVDCache:
    """Cache truncated SVD of X for fast (G+λI)^{-1} approximation in HPO."""
    _cache = {}
    
    @classmethod
    def _key(cls, X_sparse):
        d = X_sparse.data
        idx = X_sparse.indices
        ptr = X_sparse.indptr
        return hash((X_sparse.shape, X_sparse.nnz,
                    d[:10].tobytes() if len(d) >= 10 else d.tobytes(),
                    idx[:10].tobytes() if len(idx) >= 10 else idx.tobytes(),
                    ptr[:10].tobytes() if len(ptr) >= 10 else ptr.tobytes()))
    
    @classmethod
    def get(cls, X_sparse):
        return cls._cache.get(cls._key(X_sparse))
    
    @classmethod
    def put(cls, X_sparse, V, sigma2):
        cls._cache[cls._key(X_sparse)] = (V, sigma2)
        print(f"[gpu_accel] SVD approx cache stored (k={V.shape[1]}, {V.nbytes/1024**3:.1f}GB)")
    
    @classmethod
    def clear(cls):
        cls._cache.clear()


# ============================================================
# SVD Cache Manager
# ============================================================

class SVDCacheManager:
    """SVD 결과를 캐싱하고 MPS 가속을 제공하는 관리자 클래스"""
    
    def __init__(self, cache_dir='data_cache', device='auto'):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.device = get_device(device)
        print(f"[SVD-Manager] Device: {self.device}")

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

    def get_svd(self, X_sparse, k=None, target_energy=None, dataset_name=None, force_recompute=False):
        """
        캐시에서 SVD 로드 또는 새로 계산.
        k가 주어지면 k개 추출, target_energy가 주어지면 누적 에너지가 해당 비율(예: 0.95)이 될 때까지 k를 자동 선택.
        
        Returns: u, s, v, total_energy
        """
        M, N = X_sparse.shape
        if M == 0 or N == 0:
            print("[SVD-Manager] Empty matrix detected. Returning empty tensors.")
            return torch.empty(M, 0), torch.empty(0), torch.empty(N, 0), 0.0

        if k is not None and k >= min(M, N):
            print(f"[SVD-Manager] k({k}) too large for {M}x{N}. Capping to {min(M,N)-1}")
            k = max(min(M, N) - 1, 1)
            
        # Determine Cache key suffix
        suffix = f"k{k}" if k is not None else f"e{target_energy}"
        
        # Cache check
        if dataset_name and not force_recompute:
            cache_path = os.path.join(self.cache_dir, f"svd_{dataset_name}_{suffix}.pt")
            if os.path.exists(cache_path):
                print(f"[SVD-Manager] Cache hit: {dataset_name} ({suffix})")
                cp = torch.load(cache_path, map_location='cpu')
                u, s, v = cp['u'], cp['s'], cp['v']
                total_energy = cp.get('total_energy')
                if total_energy is None:
                    total_energy = float(np.sum(X_sparse.data ** 2))
                return u, s, v, total_energy

        print(f"[SVD-Manager] Computing SVD ({suffix}, shape={X_sparse.shape})...")
        start = time.time()
        
        total_energy = float(np.sum(X_sparse.data ** 2))
        
        # Calculate full SVD or large SVD if target_energy is requested
        compute_k = k if k is not None else min(M, N) - 1
        
        min_dim = min(M, N)
        if self.device == 'mps' and min_dim >= 5000:
            u, s, v = self._mps_randomized_svd(X_sparse, compute_k)
        else:
            if self.device == 'mps':
                print(f"[SVD-Manager] Matrix small ({min_dim}<5000), using CPU SVD")
            u, s, v = self._cpu_svd(X_sparse, compute_k)
            
        print(f"[SVD-Manager] SVD done ({time.time()-start:.2f}s)")

        # Truncate based on target_energy if requested
        if target_energy is not None and k is None:
            s2 = s ** 2
            cum_energy = torch.cumsum(s2, dim=0) / total_energy
            k_target = (cum_energy >= target_energy).nonzero()
            
            if len(k_target) > 0:
                final_k = k_target[0].item() + 1
            else:
                final_k = len(s)
                
            print(f"[SVD-Manager] Auto-selected k={final_k} to reach {target_energy*100:.1f}% energy.")
            u = u[:, :final_k]
            s = s[:final_k]
            v = v[:, :final_k]
            
        # Cache save
        if dataset_name:
            cache_path = os.path.join(self.cache_dir, f"svd_{dataset_name}_{suffix}.pt")
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
            print(f"[MPS-SVD] Power iteration {i+1}/{n_iter}...")
            Z = self._sparse_mm_transposed_batched(X_sparse, Q, batch_size, device)
            Q_z = self._orthonormalize(Z, device)
            Y = self._sparse_mm_batched(X_sparse, Q_z, batch_size, device)
            Q = self._orthonormalize(Y, device)
            del Z, Q_z
        
        # Phase 2: Projection (B = Q^T @ A)
        print(f"[MPS-SVD] Projection...")
        B = self._sparse_mm_transposed_batched(X_sparse, Q, batch_size, device).t()
        
        # Phase 3: Small SVD via eigen trick
        print(f"[MPS-SVD] Small matrix SVD ({q}x{q})...")
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
