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
                I = torch.eye(M, device=dev, dtype=torch.float32)
                X_t = torch.cholesky_solve(I, L)
                del I
            else:
                rhs_t = torch.from_numpy(rhs_np).float().to(dev) if isinstance(rhs_np, np.ndarray) else rhs_np.to(dev)
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
    
    return result


def gpu_gram_solve(X_sparse, reg_lambda, rhs=None, device='auto', dataset_name=None, return_tensor=False):
    """
    Compute (X^T X + λI)^-1 @ rhs.

    캐싱 전략:
    - M <= EIGEN_THRESHOLD: eigendecomposition 캐시 (_GramEigenCache). λ-agnostic.
    - M >  EIGEN_THRESHOLD: Gram matrix G=X^TX 캐시 (_GramMatrixCache) + Cholesky.
      G는 λ-agnostic하게 메모리 캐싱되며, Cholesky는 매번 새로 실행된다.
    """
    M = X_sparse.shape[1]
    EIGEN_THRESHOLD = 15000
    dev = get_device(device)

    # 1. Check eigen cache
    cache = _GramEigenCache.get(X_sparse, dataset_name, device=dev)
    if cache is not None:
        V, eigvals = cache
        t0 = time.time()

        inv_eig = 1.0 / (eigvals + reg_lambda)
        if rhs is None:
            P = (V * inv_eig.unsqueeze(0)) @ V.t()
        else:
            rhs_t = torch.from_numpy(rhs).float().to(dev) if isinstance(rhs, np.ndarray) else rhs.to(dev)
            P = (V * inv_eig.unsqueeze(0)) @ (V.t() @ rhs_t)

        print(f"[gpu_accel] Eigen solve [Tensor] Cache Hit on {dev}: {time.time()-t0:.2f}s")
        return P if return_tensor else P.cpu().numpy()

    # 2. Eigen Path
    if M <= EIGEN_THRESHOLD:
        print(f"[gpu_accel] Gram ({M}x{M}) eigendecomposition (first call) on {dev.type}...")
        t0 = time.time()

        if dev.type in ('cuda', 'mps'):
            try:
                if isinstance(X_sparse, csr_matrix):
                    X_torch_dense = torch.from_numpy(X_sparse.toarray()).float().to(dev)
                    G_t = torch.mm(X_torch_dense.t(), X_torch_dense)
                    del X_torch_dense
                else:
                    X_torch = torch.tensor(X_sparse, device=dev, dtype=torch.float32)
                    G_t = torch.mm(X_torch.t(), X_torch)

                eigvals_t, V_t = torch.linalg.eigh(G_t)
                del G_t

                _GramEigenCache.put(X_sparse, V_t, eigvals_t, dataset_name)

                print(f"[gpu_accel] {dev.type.upper()} torch.linalg.eigh done: {time.time()-t0:.2f}s")
                V, eigvals = V_t, eigvals_t
            except Exception as e:
                print(f"[gpu_accel] {dev.type.upper()} Eigen failed ({e}), fallback to Scipy CPU...")
                G = (X_sparse.T @ X_sparse).toarray().astype(np.float32)
                from scipy.linalg import eigh
                eigvals_np, V_np = eigh(G)
                V, eigvals = torch.from_numpy(V_np).to(dev), torch.from_numpy(eigvals_np).to(dev)
                _GramEigenCache.put(X_sparse, V, eigvals, dataset_name)
                del G
        else:
            G = (X_sparse.T @ X_sparse).toarray().astype(np.float32)
            from scipy.linalg import eigh
            eigvals_np, V_np = eigh(G)
            V, eigvals = torch.from_numpy(V_np).to(dev), torch.from_numpy(eigvals_np).to(dev)
            _GramEigenCache.put(X_sparse, V, eigvals, dataset_name)
            del G

        return gpu_gram_solve(X_sparse, reg_lambda, rhs, device, dataset_name, return_tensor)

    # 3. Cholesky Path (M > EIGEN_THRESHOLD)
    # G = X^T X를 λ-agnostic하게 캐싱.
    # λ가 바뀌어도 X^T X 재계산을 생략하고, G_cached + λI 후 Cholesky만 수행.
    if dev.type in ('cuda', 'mps'):
        try:
            G_t = _GramMatrixCache.get(X_sparse, dataset_name, device=dev)
            if G_t is None:
                print(f"[gpu_accel] Gram ({M}x{M}) X^T X computing on {dev.type}...")
                t0 = time.time()
                if isinstance(X_sparse, csr_matrix):
                    X_torch_dense = torch.from_numpy(X_sparse.toarray()).float().to(dev)
                    G_t = torch.mm(X_torch_dense.t(), X_torch_dense)
                    del X_torch_dense
                else:
                    X_t = torch.from_numpy(X_sparse).float().to(dev) if isinstance(X_sparse, np.ndarray) else X_sparse.to(dev)
                    G_t = torch.mm(X_t.t(), X_t)
                _GramMatrixCache.put(X_sparse, G_t, dataset_name)
                print(f"[gpu_accel] Gram X^T X cached ({time.time()-t0:.2f}s)")
            else:
                print(f"[gpu_accel] Gram ({M}x{M}) X^T X cache hit, Cholesky with λ={reg_lambda}")

            # G + λI (clone해서 캐시된 G_t는 λ 없이 보존)
            G_reg = G_t.clone()
            G_reg.diagonal().add_(reg_lambda)
            P = gpu_cholesky_solve(G_reg, rhs, device=device, dataset_name=None, return_tensor=return_tensor)
            del G_reg
            return P
        except Exception as e:
            print(f"[gpu_accel] {dev.type.upper()} Gram+Cholesky prep failed ({e}), fallback to CPU...")

    # Fallback / CPU
    G_np = _GramMatrixCache.get_numpy(X_sparse, dataset_name)
    if G_np is None:
        print(f"[gpu_accel] Gram ({M}x{M}) X^T X computing on CPU...")
        t0 = time.time()
        G_np = (X_sparse.T @ X_sparse).toarray().astype(np.float32)
        _GramMatrixCache.put_numpy(X_sparse, G_np, dataset_name)
        print(f"[gpu_accel] Gram X^T X cached ({time.time()-t0:.2f}s)")
    else:
        print(f"[gpu_accel] Gram ({M}x{M}) X^T X cache hit (CPU), Cholesky with λ={reg_lambda}")

    G_reg = G_np.copy()
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
                cls._mem_cache[key] = (V, e)
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
        cls._mem_cache[key] = (V_cpu, e_cpu)
        
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
    λ-agnostic Gram matrix G = X^T X 캐시 (디스크 + 메모리).
    λ가 바뀌어도 X^T X 재계산 없이 재사용 가능.

    - 디스크: gram_{dataset}_{checksum}.pt 로 영속 저장
    - 메모리: 가장 최근 1개만 보관 (M×M dense는 수백MB이므로 여러 개 쌓이면 OOM 위험)
    """
    _mem_cache = {}   # key -> G (CPU tensor), 최대 1개
    _np_cache  = {}   # key -> G (numpy), 최대 1개
    _cache_dir = 'data_cache'

    @classmethod
    def _checksum(cls, X):
        if isinstance(X, csr_matrix):
            d, idx, ptr = X.data, X.indices, X.indptr
            return hash((X.shape, X.nnz,
                         d[:10].tobytes()   if len(d)   >= 10 else d.tobytes(),
                         idx[:10].tobytes() if len(idx) >= 10 else idx.tobytes(),
                         ptr[:10].tobytes() if len(ptr) >= 10 else ptr.tobytes()))
        X_np = X.cpu().numpy() if torch.is_tensor(X) else X
        return hash((X_np.shape, X_np.flatten()[:100].tobytes()))

    @classmethod
    def _key(cls, X, dataset_name):
        return f"gram_{dataset_name}_{cls._checksum(X)}"

    @classmethod
    def get(cls, X, dataset_name=None, device='cpu'):
        if not dataset_name: return None
        key = cls._key(X, dataset_name)

        # 1. 메모리 캐시
        if key in cls._mem_cache:
            return cls._mem_cache[key].to(device)

        # 2. 디스크 캐시
        path = os.path.join(cls._cache_dir, f"{key}.pt")
        if os.path.exists(path):
            try:
                G = torch.load(path, map_location='cpu', weights_only=True)
                cls._mem_cache = {key: G}   # 메모리는 최근 1개만
                print(f"[gpu_accel] Gram disk cache loaded: {os.path.basename(path)}")
                return G.to(device)
            except Exception:
                pass
        return None

    @classmethod
    def put(cls, X, G, dataset_name=None):
        if not dataset_name: return
        os.makedirs(cls._cache_dir, exist_ok=True)
        key = cls._key(X, dataset_name)
        G_cpu = G.cpu()
        cls._mem_cache = {key: G_cpu}   # 메모리는 최근 1개만 유지
        path = os.path.join(cls._cache_dir, f"{key}.pt")
        torch.save(G_cpu, path)
        print(f"[gpu_accel] Gram X^T X saved to disk: {os.path.basename(path)}")

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
                G_np = torch.load(path, map_location='cpu', weights_only=True).numpy()
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
        torch.save(torch.from_numpy(G_np), path)
        print(f"[gpu_accel] Gram X^T X saved to disk: {os.path.basename(path)}")

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
                cls._mem_cache[key] = L
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
        cls._mem_cache[key] = L_cpu
        path = os.path.join(cls._cache_dir, f"{key}.pt")
        torch.save(L_cpu, path)
        print(f"[gpu_accel] Cholesky L cached to disk: {os.path.basename(path)}")

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
                    is_full_use = True
                    if k is not None and len(s) > k:
                        print(f"[SVD] Truncating: k={len(s)} -> k={k}")
                        u, s, v = u[:, :k], s[:k], v[:, :k]
                        is_full_use = False
                    
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
                                # Recompute loop로 진입하기 위해 여기서 return하지 않음.
                                print(f"[SVD] Cache k={len(s)} insufficient for target energy. Recomputing...")
                            else:
                                # 사용자가 k를 명시적으로 요청한 경우, 에너지가 부족하더라도 캐시된 k만큼은 반환.
                                return u, s, v, total_energy 
                    else:
                        # target_energy가 지정되지 않은 경우, 찾은 캐시를 즉시 반환.
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
            
            
        # 3. Save Cache & Cleanup smaller k
        save_k = len(s)
        cache_path = os.path.join(self.cache_dir, f"svd_{dataset_name}_{matrix_id}_k{save_k}.pt")
        
        # Cleanup smaller k files for this matrix/dataset
        import glob
        pattern = os.path.join(self.cache_dir, f"svd_{dataset_name}_{matrix_id}_k*.pt")
        for f in glob.glob(pattern):
            try:
                f_k = int(f.split("_k")[-1].replace(".pt", ""))
                if f_k < save_k:
                    os.remove(f)
                    print(f"[SVD] Consolidating cache: removed smaller k={f_k}")
            except Exception: pass

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
