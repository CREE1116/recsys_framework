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

def gpu_cholesky_solve(G_np, rhs_np=None, device='auto', dataset_name=None, return_tensor=False):
    """
    Solve G @ X = rhs via Cholesky.
    Args:
        G_np: (M, M) numpy array or torch.Tensor on device
        rhs_np: (M, K) numpy array or torch.Tensor on device
        dataset_name: Optional string to enable persistent caching of the L factor.
        return_tensor: If True, returns torch.Tensor on device.
    """
    dev = get_device(device)
    
    # 1. Handle Tensor input (from other gpu_accel functions)
    if torch.is_tensor(G_np):
        G_t = G_np.float().to(dev)
        M = G_t.shape[0]
        G_np_for_hash = None # We'll skip hash if it's already a tensor and no dataset_name
    else:
        M = G_np.shape[0]
        G_np_for_hash = G_np

    # 2. Check Cache
    if dataset_name:
        L_cached = CholeskyCacheManager.get(G_np_for_hash if G_np_for_hash is not None else G_t, dataset_name, device=dev)
        if L_cached is not None:
            t0 = time.time()
            try:
                if rhs_np is None:
                    I = torch.eye(M, device=dev, dtype=torch.float32)
                    X_t = torch.cholesky_solve(I, L_cached)
                    del I
                else:
                    rhs_t = torch.from_numpy(rhs_np).float().to(dev) if isinstance(rhs_np, np.ndarray) else rhs_np.to(dev)
                    X_t = torch.cholesky_solve(rhs_t, L_cached)
                
                print(f"[gpu_accel] {dev.type.upper()} Cholesky Cache Hit ({M}x{M}): {time.time()-t0:.2f}s")
                return X_t if return_tensor else X_t.cpu().numpy()
            except Exception as e:
                print(f"[gpu_accel] {dev.type.upper()} cholesky_solve failed ({e}), fallback to CPU...")
                # L_cached is on device, but we need CPU L for scipy fallback
                # Actually _cpu_cholesky expects G_np. Let's just fall through to compute or CPU.
                # Since we already have the cache but cannot use it on this device, 
                # we'll proceed to the official fallback section below.
                pass

    # 3. Compute
    if dev.type in ('mps', 'cuda'):
        try:
            t0 = time.time()
            if not torch.is_tensor(G_np):
                G_t = torch.from_numpy(G_np).float().to(dev)
            
            L = torch.linalg.cholesky(G_t)
            
            # Save to Cache if dataset_name is provided
            if dataset_name:
                CholeskyCacheManager.put(G_np_for_hash if G_np_for_hash is not None else G_t, L, dataset_name)

            if rhs_np is None:
                # [MEMORY FIX] Solve identity in blocks to avoid M x M intermediate I on GPU
                # M=33k -> 4.26GB for identity matrix.
                X_t = torch.empty((M, M), device=dev, dtype=torch.float32)
                block_size = 5000
                for i in range(0, M, block_size):
                    end = min(i + block_size, M)
                    I_block = torch.zeros((M, end - i), device=dev, dtype=torch.float32)
                    I_block[i:end, :] = torch.eye(end - i, device=dev, dtype=torch.float32)
                    X_t[:, i:end] = torch.cholesky_solve(I_block, L)
                    del I_block
            else:
                rhs_t = torch.from_numpy(rhs_np).float().to(dev) if isinstance(rhs_np, np.ndarray) else rhs_np.to(dev)
                X_t = torch.cholesky_solve(rhs_t, L)
            del L
            if not torch.is_tensor(G_np): # G_t was created locally from NumPy
                del G_t
            
            if return_tensor:
                print(f"[gpu_accel] {dev.type.upper()} Cholesky Solve ({M}x{M}) [Tensor]: {time.time()-t0:.2f}s")
                return X_t
                
            result = X_t.cpu().numpy()
            print(f"[gpu_accel] {dev.type.upper()} Cholesky Solve ({M}x{M}) [NumPy]: {time.time()-t0:.2f}s")
            return result
        except RuntimeError as e:
            print(f"[gpu_accel] {dev.type.upper()} cholesky OOM ({e}), CPU fallback...")
    
    # Fallback to CPU if GPU failed or not available
    G_np_eval = G_np if isinstance(G_np, np.ndarray) else G_np.cpu().numpy()
    rhs_np_eval = rhs_np if (rhs_np is None or isinstance(rhs_np, np.ndarray)) else rhs_np.cpu().numpy()
    res_np = _cpu_cholesky(G_np_eval, rhs_np_eval)
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
    
def _build_gram(X_sparse, device):
    """
    Optimized Gram matrix (X^T X) build.
    - CUDA: bfloat16 matmul (ADA HW acceleration) -> float32.
    - Others: float32 matmul.
    """
    X_dense_cpu = torch.from_numpy(X_sparse.toarray())
    if device.type == 'cuda':
        # ADA architecture: bfloat16 is fully supported and fast
        X_bf = X_dense_cpu.bfloat16().to(device)
        G = (X_bf.t() @ X_bf).float()
        del X_bf
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        # MPS: bfloat16 might not be fully accelerated yet, use float32
        X_f = X_dense_cpu.float().to(device)
        G = (X_f.t() @ X_f)
        del X_f
    else:
        X_f = X_dense_cpu.float()
        G = X_f.t() @ X_f
        del X_f
    
    del X_dense_cpu
    return G

def gpu_gram_solve(X_sparse, reg_lambda, rhs=None, device='auto', dataset_name=None, return_tensor=False):
    """
    Compute (X^T X + λI)^-1 @ rhs.
    - Automatic caching (smart bfloat16 storage).
    - ADA Optimized bfloat16 path for matrix builds.
    """
    M = X_sparse.shape[1]
    EIGEN_THRESHOLD = 15000
    dev = get_device(device)

    # 1. Eigen Path (M <= EIGEN_THRESHOLD)
    if M <= EIGEN_THRESHOLD:
        print(f"[gpu_accel] Gram ({M}x{M}) eigendecomposition on {dev.type}...")
        t0 = time.time()
        
        # scipy float32 conversion to save memory
        G_np = (X_sparse.T @ X_sparse).astype(np.float32).toarray()
        
        if dev.type in ('cuda', 'mps'):
            try:
                G_t = torch.from_numpy(G_np).to(dev)
                eigvals, V = torch.linalg.eigh(G_t)
                del G_t
            except Exception as e:
                print(f"[gpu_accel] {dev.type.upper()} Eigen failed ({e}), fallback to Scipy CPU...")
                from scipy.linalg import eigh
                eigvals_np, V_np = eigh(G_np)
                V, eigvals = torch.from_numpy(V_np).to(dev), torch.from_numpy(eigvals_np).to(dev)
                del G_np
        else:
            from scipy.linalg import eigh
            eigvals_np, V_np = eigh(G_np)
            V, eigvals = torch.from_numpy(V_np).to(dev), torch.from_numpy(eigvals_np).to(dev)
            del G_np

        inv_eig = 1.0 / (eigvals + reg_lambda)
        if rhs is None:
            P = (V * inv_eig.unsqueeze(0)) @ V.t()
        else:
            rhs_t = torch.from_numpy(rhs).float().to(dev) if isinstance(rhs, np.ndarray) else rhs.to(dev)
            P = (V * inv_eig.unsqueeze(0)) @ (V.t() @ rhs_t)

        return P if return_tensor else P.cpu().numpy()

    # 2. Cholesky Path (M > EIGEN_THRESHOLD)
    if dev.type in ('cuda', 'mps'):
        try:
            # Re-enabling caching with bfloat16 optimization as per new ADA guide
            G_t = _GramMatrixCache.get(X_sparse, dataset_name, device=dev)
            if G_t is None:
                print(f"[gpu_accel] Gram ({M}x{M}) X^T X computing on {dev.type} (bfloat16 ADA path)...")
                t0 = time.time()
                # Use bfloat16 for the build
                G_t = _build_gram(X_sparse, dev)
                _GramMatrixCache.put(X_sparse, G_t, dataset_name)
                print(f"[gpu_accel] Gram X^T X cached (bfloat16 build, {time.time()-t0:.2f}s)")
            else:
                print(f"[gpu_accel] Gram ({M}x{M}) cache hit (bfloat16 optimized)")

            # G + λI (Modify in-place)
            G_t.diagonal().add_(reg_lambda)
            P = gpu_cholesky_solve(G_t, rhs, device=dev, dataset_name=None, return_tensor=return_tensor)
            # Revert to keep cached G clean
            G_t.diagonal().sub_(reg_lambda)
            return P
        except Exception as e:
            print(f"[gpu_accel] {dev.type.upper()} Gram+Cholesky prep failed ({e}), fallback to CPU...")

    # Fallback / CPU
    print(f"[gpu_accel] Gram ({M}x{M}) X^T X computing on CPU...")
    t0 = time.time()
    G_np = (X_sparse.T @ X_sparse).astype(np.float32).toarray()
    print(f"[gpu_accel] Gram X^T X computed ({time.time()-t0:.2f}s)")

    G_reg = G_np
    G_reg[np.diag_indices(M)] += reg_lambda
    P = gpu_cholesky_solve(G_reg, rhs, device=device, dataset_name=None, return_tensor=return_tensor)
    del G_reg
    return P


class GramEigenCacheManager(GlobalCacheManager):
    """
    Persistent cache for Gram matrix eigendecomposition.
    Keyed by (dataset_name, matrix_checksum).
    """
    _mem_cache = {}  # key -> (V, eigvals)
    _cache_dir = 'data_cache'
    
    @classmethod
    def _checksum(cls, X):
        if isinstance(X, csr_matrix):
            d = X.data
            idx = X.indices
            ptr = X.indptr
            checksum = hash((X.shape, X.nnz, 
                            d[:10].tobytes() if len(d) >= 10 else d.tobytes(),
                            idx[:10].tobytes() if len(idx) >= 10 else idx.tobytes(),
                            ptr[:10].tobytes() if len(ptr) >= 10 else ptr.tobytes()))
        else:
            # For dense or tensor
            X_np = X.cpu().numpy() if torch.is_tensor(X) else X
            checksum = hash((X_np.shape, X_np.flatten()[:100].tobytes()))
        return checksum
    
    @classmethod
    def get(cls, X, dataset_name=None, device='cpu'):
        if not dataset_name: return None
        checksum = cls._checksum(X)
        key = f"eigen_{dataset_name}_{checksum}"
        
        # 1. Memory cache
        if key in cls._mem_cache:
            V, e = cls._mem_cache[key]
            return V.to(device), e.to(device)
            
        # 2. Disk cache
        path = os.path.join(cls._cache_dir, f"{key}.pt")
        if os.path.exists(path):
            try:
                cp = torch.load(path, map_location='cpu')
                V, e = cp['V'], cp['eigvals']
                cls._mem_cache = {key: (V, e)}  # Keep only recent
                return V.to(device), e.to(device)
            except Exception: pass
        return None
    
    @classmethod
    def put(cls, X, V, eigvals, dataset_name=None):
        if not dataset_name: return
        os.makedirs(cls._cache_dir, exist_ok=True)
        checksum = cls._checksum(X)
        key = f"eigen_{dataset_name}_{checksum}"
        
        V_cpu, e_cpu = V.cpu(), eigvals.cpu()
        cls._mem_cache = {key: (V_cpu, e_cpu)}  # Keep only recent
        
        path = os.path.join(cls._cache_dir, f"{key}.pt")
        torch.save({'V': V_cpu, 'eigvals': e_cpu}, path)
        print(f"[gpu_accel] Eigen cached to disk: {os.path.basename(path)}")
    
    def summary(self):
        import glob
        files = glob.glob(os.path.join(self._cache_dir, "eigen_*.pt"))
        total_bytes = sum(os.path.getsize(f) for f in files) if files else 0
        return {"type": "GramEigen", "files": len(files), "size_mb": round(total_bytes / 1e6, 1)}

    def invalidate(self, key=None):
        import glob
        pattern = f"eigen_{key}*.pt" if key else "eigen_*.pt"
        for f in glob.glob(os.path.join(self._cache_dir, pattern)):
            os.remove(f)
        self._mem_cache.clear()

    @classmethod
    def clear(cls):
        """메모리 캐시만 비움 (디스크는 유지). HPO trial 사이 OOM 방지용."""
        cls._mem_cache.clear()

_GramEigenCache = GramEigenCacheManager


class GramMatrixCacheManager(GlobalCacheManager):
    """
    Persistent cache for Gram matrix G = X^T X.
    Uses bfloat16 to save 50% disk/memory on ADA architecture.
    """
    _mem_cache = {}   # key -> G (bfloat16 CPU tensor), 최대 1개
    _np_cache  = {}   # key -> G (numpy), 최대 1개
    _cache_dir = 'data_cache'

    @classmethod
    def _checksum(cls, X):
        if isinstance(X, csr_matrix):
            d, idx, ptr = X.data, X.indices, X.indptr
            # Raw matrix hash
            return hash((X.shape, X.nnz,
                         d[:200].tobytes()   if len(d)   >= 200 else d.tobytes(),
                         idx[:200].tobytes() if len(idx) >= 200 else idx.tobytes()))
        X_np = X.cpu().numpy() if torch.is_tensor(X) else X
        return hash((X_np.shape, X_np.flatten()[:200].tobytes()))

    @classmethod
    def _key(cls, X, dataset_name):
        return f"gram_{dataset_name}_{cls._checksum(X)}"

    @classmethod
    def get(cls, X, dataset_name=None, device='cpu'):
        if not dataset_name: return None
        key = cls._key(X, dataset_name)
        dev = torch.device(device)

        # 1. 메모리 캐시
        if key in cls._mem_cache:
            return cls._mem_cache[key].float().to(dev)

        # 2. 디스크 캐시
        path = os.path.join(cls._cache_dir, f"{key}.pt")
        if os.path.exists(path):
            try:
                # Load as stored (bfloat16) and cast to float on device
                G_bf = torch.load(path, map_location='cpu', weights_only=True)
                cls._mem_cache = {key: G_bf}   # 메모리는 최근 1개만 (bf16)
                print(f"[gpu_accel] Gram disk cache loaded: {os.path.basename(path)} (bfloat16)")
                return G_bf.float().to(dev)
            except Exception:
                pass
        return None

    @classmethod
    def put(cls, X, G, dataset_name=None):
        if not dataset_name: return
        os.makedirs(cls._cache_dir, exist_ok=True)
        key = cls._key(X, dataset_name)
        
        # Store as bfloat16 to save disk/memory (ADA Guide)
        G_bf = G.bfloat16().detach().cpu()
        cls._mem_cache = {key: G_bf}   # 메모리는 최근 1개만 유지
        
        path = os.path.join(cls._cache_dir, f"{key}.pt")
        torch.save(G_bf, path)
        print(f"[gpu_accel] Gram X^T X saved to disk: {os.path.basename(path)} (bfloat16)")

    @classmethod
    def get_numpy(cls, X, dataset_name=None):
        if not dataset_name: return None
        key = cls._key(X, dataset_name)

        # 1. 메모리 캐시
        if key in cls._np_cache:
            return cls._np_cache[key]

        # 2. 디스크 캐시 (tensor → numpy 변환)
        path = os.path.join(cls._cache_dir, f"{key}.pt")
        if os.path.exists(path):
            try:
                G_np = torch.load(path, map_location='cpu', weights_only=True).float().numpy()
                cls._np_cache = {key: G_np}
                print(f"[gpu_accel] Gram disk cache loaded (numpy): {os.path.basename(path)}")
                return G_np
            except Exception:
                pass
        return None

    @classmethod
    def put_numpy(cls, X, G_np, dataset_name=None):
        if not dataset_name: return
        os.makedirs(cls._cache_dir, exist_ok=True)
        key = cls._key(X, dataset_name)
        cls._np_cache = {key: G_np}
        path = os.path.join(cls._cache_dir, f"{key}.pt")
        # Save as bfloat16 even from numpy
        torch.save(torch.from_numpy(G_np).bfloat16(), path)
        print(f"[gpu_accel] Gram X^T X saved to disk: {os.path.basename(path)} (bfloat16)")

    def summary(self):
        import glob
        files = glob.glob(os.path.join(self._cache_dir, "gram_*.pt"))
        total_bytes = sum(os.path.getsize(f) for f in files) if files else 0
        return {"type": "GramMatrix", "files": len(files), "size_mb": round(total_bytes / 1e6, 1)}

    def invalidate(self, key=None):
        import glob
        pattern = f"gram_{key}*.pt" if key else "gram_*.pt"
        for f in glob.glob(os.path.join(self._cache_dir, pattern)):
            os.remove(f)
        self._mem_cache.clear()
        self._np_cache.clear()

    @classmethod
    def clear(cls):
        """메모리 캐시만 비움 (디스크는 유지). HPO trial 사이 OOM 방지용."""
        cls._mem_cache.clear()
        cls._np_cache.clear()

_GramMatrixCache = GramMatrixCacheManager


class CholeskyCacheManager(GlobalCacheManager):
    """
    Persistent cache for Cholesky L factors.
    Keyed by (dataset_name, matrix_checksum).
    """
    _mem_cache = {}
    _cache_dir = 'data_cache'

    @classmethod
    def _checksum(cls, X):
        # Similar to GramEigen but for dense matrices usually used in Cholesky
        X_np = X.cpu().numpy() if torch.is_tensor(X) else X
        return hash((X_np.shape, X_np.flatten()[:100].tobytes())) if X_np is not None else 0

    @classmethod
    def get(cls, X, dataset_name=None, device='cpu'):
        if not dataset_name: return None
        checksum = cls._checksum(X)
        key = f"chol_{dataset_name}_{checksum}"
        
        if key in cls._mem_cache:
            return cls._mem_cache[key].to(device)
            
        path = os.path.join(cls._cache_dir, f"{key}.pt")
        if os.path.exists(path):
            try:
                L = torch.load(path, map_location='cpu')
                cls._mem_cache = {key: L} # Keep only recent
                return L.to(device)
            except Exception: pass
        return None

    @classmethod
    def put(cls, X, L, dataset_name=None):
        if not dataset_name: return
        os.makedirs(cls._cache_dir, exist_ok=True)
        checksum = cls._checksum(X)
        key = f"chol_{dataset_name}_{checksum}"
        
        L_cpu = L.cpu()
        cls._mem_cache = {key: L_cpu}  # Keep only recent
        path = os.path.join(cls._cache_dir, f"{key}.pt")
        torch.save(L_cpu, path)
        print(f"[gpu_accel] Cholesky L cached to disk: {os.path.basename(path)}")

    @classmethod
    def clear(cls):
        """메모리 캐시만 비움."""
        cls._mem_cache.clear()

    def summary(self):
        import glob
        files = glob.glob(os.path.join(self._cache_dir, "chol_*.pt"))
        total_bytes = sum(os.path.getsize(f) for f in files) if files else 0
        return {"type": "Cholesky", "files": len(files), "size_mb": round(total_bytes / 1e6, 1)}

    def invalidate(self, key=None):
        import glob
        pattern = f"chol_{key}*.pt" if key else "chol_*.pt"
        for f in glob.glob(os.path.join(self._cache_dir, pattern)):
            os.remove(f)
        self._mem_cache.clear()




# ============================================================
# EVD Cache Manager (Eigen-decomposition based SVD)
# ============================================================

class EVDCacheManager(GlobalCacheManager):
    """
    Eigen-decomposition based Spectral Analysis Manager.
    Specifically designed for ASPIRE to avoid iterative SVD loops.
    """
    def __init__(self, cache_dir='data_cache', device='auto'):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.device = get_device(device)
        print(f"[EVD-Manager] Device: {self.device}")

    def summary(self):
        import glob
        files = glob.glob(os.path.join(self.cache_dir, "evd_*.pt"))
        total_size = sum(os.path.getsize(f) for f in files) if files else 0
        return {"type": "EVD", "files": len(files), "size_mb": round(total_size / 1e6, 1)}

    @classmethod
    def clear(cls):
        """메모리 내 보관 중인 데이터가 있다면 비움 (현재는 디스크 기반이나 인터페이스 유지를 위해 추가)"""
        pass

    def invalidate(self, key=None):
        import glob
        pattern = os.path.join(self.cache_dir, f"evd_{key or '*'}*.pt")
        for f in glob.glob(pattern):
            os.remove(f)
            print(f"[EVD-Manager] Invalidated: {os.path.basename(f)}")

    @staticmethod
    def _generate_matrix_id(X_sparse):
        if not hasattr(X_sparse, 'shape'): return "unknown"
        meta = (X_sparse.shape, X_sparse.nnz)
        d = X_sparse.data[:100].tobytes() if len(X_sparse.data) >= 100 else X_sparse.data.tobytes()
        idx = X_sparse.indices[:100].tobytes() if len(X_sparse.indices) >= 100 else X_sparse.indices.tobytes()
        import hashlib
        h = hashlib.md5()
        h.update(str(meta).encode())
        h.update(d)
        h.update(idx)
        return h.hexdigest()[:12]

    def get_evd(self, X_sparse, dataset_name=None, k=None, force_recompute=False):
        """
        Compute or load Eigen-decomposition based SVD.
        dataset_name: 캐시 키용 데이터셋 이름
        k: None일 경우 Full EVD 시도. 단, 행렬이 너무 크면 자동으로 최적 k로 Truncate.
        """
        matrix_id = self._generate_matrix_id(X_sparse)
        dataset_name = dataset_name or "unknown"
        
        # 1. Cache Search (Full or Large enough truncated)
        suffix = "full" if k is None else f"k{k}"
        if not force_recompute:
            import glob
            # Pattern: evd_{dataset_name}_{matrix_id}_*.pt
            pattern = os.path.join(self.cache_dir, f"evd_{dataset_name}_{matrix_id}_*.pt")
            cache_files = glob.glob(pattern)
            
            if cache_files:
                candidates = []
                for f in cache_files:
                    f_name = os.path.basename(f)
                    if "_full.pt" in f_name:
                        candidates.append((float('inf'), f))
                    elif "_k" in f_name:
                        try:
                            f_k = int(f_name.split("_k")[-1].replace(".pt", ""))
                            candidates.append((f_k, f))
                        except ValueError: continue
                
                candidates.sort(reverse=True) # Large k first
                
                best_file = None
                if k is None:
                    # Full requested -> only pick 'inf'
                    for ck, cp in candidates:
                        if ck == float('inf'):
                            best_file = cp; break
                else:
                    # Truncated requested -> pick smallest available k >= requested k
                    best_candidates = [c for c in candidates if c[0] >= k]
                    if best_candidates:
                        best_file = min(best_candidates, key=lambda x: x[0])[1]

                if best_file:
                    f_path = best_file
                    print(f"[EVD] Cache hit ({os.path.basename(f_path)})")
                    try:
                        cp = torch.load(f_path, map_location='cpu')
                    except Exception as e:
                        print(f"[EVD] Cache corrupted, deleting and re-computing: {e}")
                        os.remove(f_path)
                        cp = None
                    if cp is not None:
                        u, s, v = cp['u'], cp['s'], cp['v']
                        total_energy = cp.get('total_energy', float(np.sum(X_sparse.data**2)))
                        
                        if k is not None and u.shape[1] > k:
                             return u[:, :k].to(self.device), s[:k].to(self.device), v[:, :k].to(self.device), total_energy
                        return u.to(self.device), s.to(self.device), v.to(self.device), total_energy

        # 2. Compute
        u, s, v, k_final = self._compute_evd(X_sparse, k=k)
        
        # 3. Save Cache
        save_suffix = "full" if k is None and k_final == min(X_sparse.shape) else f"k{k_final}"
        cache_path = os.path.join(self.cache_dir, f"evd_{dataset_name}_{matrix_id}_{save_suffix}.pt")
        
        total_energy = float(np.sum(X_sparse.data**2))
        torch.save({'u': u.cpu(), 's': s.cpu(), 'v': v.cpu(), 'total_energy': total_energy}, cache_path)
        return u, s, v, total_energy

    @torch.no_grad()
    def _compute_evd(self, X_sparse, k=None):
        """
        Memory-efficient Full EVD path.
        Avoids creating full dense interaction matrix X.
        Uses Sparse-Dense MatVec logic for reconstruction to save memory.
        """
        M, N = X_sparse.shape
        t0 = time.time()
        
        # Explicit truncation ONLY if k is provided (no automatic fallback)
        if k is not None:
            print(f"[EVD-Manager] Path Truncated (k={k}, {M}x{N}) on {self.device}...")
            manager = SVDCacheManager(device=str(self.device))
            u, s, v = manager._cuda_randomized_svd(X_sparse, k) if self.device.type == 'cuda' else \
                      (manager._mps_randomized_svd(X_sparse, k) if self.device.type == 'mps' else \
                       manager._cpu_svd(X_sparse, k))
            return u.to(self.device), s.to(self.device), v.to(self.device), len(s)

        print(f"[EVD-Manager] Path Start Full ({M}x{N}) on {self.device}...")
        side = 'item' if N <= M else 'user'
        
        # 1. Compute Gram Matrix G = X^T X or X X^T efficiently
        # Using Sparse-Sparse MM (Scipy) is memory efficient
        print(f"[EVD-Manager] Computing Gram Matrix ({side} side) via Sparse MM...")
        if side == 'item':
            G_sparse = X_sparse.T @ X_sparse
        else:
            G_sparse = X_sparse @ X_sparse.T
            
        print(f"[EVD-Manager] Converting Gram Matrix to Dense (Size: {G_sparse.shape[0]}x{G_sparse.shape[1]})...")
        G_np = G_sparse.toarray().astype(np.float32)
        del G_sparse

        G = torch.from_numpy(G_np).to(self.device)
        del G_np

        # 2. Eigen-decomposition of Gram matrix
        print(f"[EVD-Manager] Solving EVD for {G.shape[0]}x{G.shape[1]} Gram Matrix...")
        try:
            # torch.linalg.eigh is generally faster than scipy.linalg.eigh on GPU
            eigvals, eigvecs = torch.linalg.eigh(G)
        except (RuntimeError, NotImplementedError) as e:
            print(f"[EVD-Manager] Solver failed on {self.device.type} ({e}), fallback to Scipy CPU...")
            G_cpu_np = G.cpu().numpy()
            from scipy.linalg import eigh as scipy_eigh
            eigvals_np, eigvecs_np = scipy_eigh(G_cpu_np)
            eigvals = torch.from_numpy(eigvals_np).to(self.device).float()
            eigvecs = torch.from_numpy(eigvecs_np).to(self.device).float()
        del G
        
        # Sort descending (Signal first)
        eigvals = torch.flip(eigvals, dims=[0])
        eigvecs = torch.flip(eigvecs, dims=[1])
        s = torch.sqrt(torch.clamp(eigvals, min=0.0))
        
        # 3. Reconstruct Singular Vectors
        # Avoid creating dense X. Use Sparse-Dense MM (MatVec style) in batches.
        print(f"[EVD-Manager] Reconstructing Singular Vectors via Sparse-Dense MM...")
        s_inv = torch.where(s > 1e-12, 1.0 / s, torch.zeros_like(s))
        
        # Force GC before heavy reconstruction
        import gc; gc.collect()

        if side == 'item':
            v_k = eigvecs
            # u_k = X @ v_k * s_inv
            u_k = SVDCacheManager._sparse_mm_batched(X_sparse, v_k, batch_size=2000, device=self.device)
            u_k = u_k * s_inv.unsqueeze(0)
        else:
            u_k = eigvecs
            # v_k = X^T @ u_k * s_inv
            v_k = SVDCacheManager._sparse_mm_transposed_batched(X_sparse, u_k, batch_size=2000, device=self.device)
            v_k = v_k * s_inv.unsqueeze(0)
            
        print(f"[EVD-Manager] Done in {time.time()-t0:.2f}s (Full Components: {len(s)})")
        return u_k, s, v_k, len(s)


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

    @classmethod
    def clear(cls):
        """메모리 비움 (인터페이스 유지용)"""
        pass

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
                else:
                    best_file = candidates[-1] # Pick largest available

                if best_file:
                    f_k, f_path = best_file
                    print(f"[SVD] Cache hit (k={f_k}, dataset={dataset_name})")
                    try:
                        cp = torch.load(f_path, map_location='cpu')
                    except Exception as e:
                        print(f"[SVD] Cache corrupted, deleting and re-computing: {e}")
                        os.remove(f_path)
                        cp = None
                    if cp is not None:
                        u, s, v = cp['u'], cp['s'], cp['v']
                        total_energy = cp.get('total_energy', float(np.sum(X_sparse.data**2)))
                        
                        if k is not None and f_k > k:
                            return u[:, :k].to(self.device), s[:k].to(self.device), v[:, :k].to(self.device), total_energy
                        return u.to(self.device), s.to(self.device), v.to(self.device), total_energy

        # 2. Compute
        if k is None:
            k = int(min(X_sparse.shape) * 0.1)
            print(f"[SVD] k not specified, using 10% of min_dim: {k}")

        if self.device.type == 'cuda':
            u, s, v = self._cuda_randomized_svd(X_sparse, k)
        elif self.device.type == 'mps':
            u, s, v = self._mps_randomized_svd(X_sparse, k)
        else:
            u, s, v = self._cpu_svd(X_sparse, k)

        # 3. Save Cache
        save_k = len(s)
        cache_path = os.path.join(self.cache_dir, f"svd_{dataset_name}_{matrix_id}_k{save_k}.pt")
        self._cleanup_old_cache(dataset_name, matrix_id, save_k)

        total_energy = float(np.sum(X_sparse.data**2))
        torch.save({'u': u, 's': s, 'v': v, 'total_energy': total_energy}, cache_path)
            
        return u.to(self.device), s.to(self.device), v.to(self.device), total_energy

    def _cleanup_old_cache(self, dataset_name, matrix_id, save_k):
        import glob
        pattern = os.path.join(self.cache_dir, f"svd_{dataset_name}_{matrix_id}_k*.pt")
        for f in glob.glob(pattern):
            try:
                # Be careful with parsing k
                parts = f.split("_k")
                if len(parts) < 2: continue
                f_k_str = parts[-1].replace(".pt", "")
                if not f_k_str.isdigit(): continue
                f_k = int(f_k_str)
                if f_k < save_k:
                    os.remove(f)
                    print(f"[SVD] Consolidating cache: removed smaller k={f_k}")
            except Exception: pass

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
    def _cuda_randomized_svd(self, X_sparse, k, n_iter=3, oversampling=20):
        """CUDA-accelerated Randomized SVD (ADA optimized).
        - Quality: n_iter=3, oversampling=20.
        - Performance: Avoid contiguous copy on B.
        """
        device = torch.device("cuda")
        M, N = X_sparse.shape
        q = min(k + oversampling, M, N)
        if q < 1: q = 1

        print(f"[CUDA-SVD] k={k}, q={q}, n_iter={n_iter}")

        # Build CUDA sparse CSR tensors
        if not isinstance(X_sparse, csr_matrix): X_sparse = X_sparse.tocsr()
        X_coo = X_sparse.tocoo()
        indices = torch.from_numpy(np.vstack((X_coo.row, X_coo.col))).long()
        values = torch.from_numpy(X_coo.data.copy()).float()
        X_t = torch.sparse_coo_tensor(indices, values, (M, N), device=device).coalesce().to_sparse_csr()
        Xt_t = torch.sparse_coo_tensor(torch.stack([indices[1], indices[0]]), values, (N, M), device=device).coalesce().to_sparse_csr()
        del X_coo, indices, values

        # Phase 1: Random sketch
        G = torch.randn(N, q, device=device, dtype=torch.float32)
        Y = torch.sparse.mm(X_t, G)
        Q, _ = torch.linalg.qr(Y)
        del G, Y

        # Phase 2: Power iterations
        for i in range(n_iter):
            Z = torch.sparse.mm(Xt_t, Q)
            Q_z, _ = torch.linalg.qr(Z)
            Y = torch.sparse.mm(X_t, Q_z)
            Q, _ = torch.linalg.qr(Y)
            del Z, Y, Q_z

        # Phase 3: Project into low-dimensional space
        # Optimized: B is (N, q), avoid .t() contiguous copy
        B = torch.sparse.mm(Xt_t, Q)

        # Phase 4: Small dense SVD
        # B = U_hat @ S @ Vht
        U_hat, S_vals, Vht = torch.linalg.svd(B, full_matrices=False)

        # Recover full-space singular vectors
        # X ~ Q @ B.t() = Q @ (Vht.t() @ S @ U_hat.t())
        U = torch.mm(Q, Vht.t())
        V = U_hat

        U, S_vals, V = U[:, :k].cpu(), S_vals[:k].cpu(), V[:, :k].cpu()
        print(f"[CUDA-SVD] Done! σ range: [{S_vals[-1]:.4f}, {S_vals[0]:.4f}]")
        return U, S_vals, V

    @torch.no_grad()
    def _mps_randomized_svd(self, X_sparse, k, n_iter=3, oversampling=20, batch_size=2000):
        """MPS-accelerated Randomized SVD (ADA/Apple optimization).
        - Quality: n_iter=3, oversampling=20.
        """
        device = torch.device("mps")
        M, N = X_sparse.shape
        q = min(k + oversampling, M, N)
        if q < 1: q = 1
        
        # Ensure CSR for batched slicing
        if not isinstance(X_sparse, csr_matrix): X_sparse = X_sparse.tocsr()

        print(f"[MPS-SVD] k={k}, q={q}, n_iter={n_iter}, batch={batch_size}")
        
        # Phase 1: Sketching + Power Iteration
        G = torch.randn(N, q, device=device, dtype=torch.float32)
        Y = self._sparse_mm_batched(X_sparse, G, batch_size, device)
        Q = self._orthonormalize(Y, device)
        del G, Y

        Xt_t = self.Xt_torch_csr.to(device)
        for i in range(n_iter):
            # Q is (M, q), Xt_t is (N, M) -> Z is (N, q)
            Z = torch.sparse.mm(Xt_t, Q)
            Q_z = self._orthonormalize(Z, device)
            
            # X_t @ Q_z -> Q
            Y = self._sparse_mm_batched(X_sparse, Q_z, batch_size, device)
            Q = self._orthonormalize(Y, device)
            del Z, Y, Q_z
        
        # Phase 2: Project
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
        if not isinstance(X_sparse, csr_matrix): X_sparse = X_sparse.tocsr()
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
        if not isinstance(X_sparse, csr_matrix): X_sparse = X_sparse.tocsr()
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
