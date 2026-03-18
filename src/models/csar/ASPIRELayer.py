from __future__ import annotations

import os
import json
import time
import glob as _glob

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.utils.gpu_accel import SVDCacheManager, EVDCacheManager
from src.models.csar.aspire_visualizer import ASPIREVisualizer
from src.utils.cache_manager import GlobalCacheManager
from src.models.csar import beta_estimators

# ==============================================================================
# 캐시 매니저
# ==============================================================================

class ASPIREBetaCacheManager(GlobalCacheManager):
    """β 영속 캐시. dataset_name 단위."""
    _mem_cache: dict = {}
    _cache_dir: str  = "data_cache"

    @classmethod
    def _get_path(cls, dataset_name):
        if not dataset_name: return None
        os.makedirs(cls._cache_dir, exist_ok=True)
        return os.path.join(cls._cache_dir, f"aspire_beta_{dataset_name}.json")

    @classmethod
    def get(cls, dataset_name):
        if not dataset_name: return None
        if dataset_name in cls._mem_cache:
            return cls._mem_cache[dataset_name]
        path = cls._get_path(dataset_name)
        if path and os.path.exists(path):
            try:
                with open(path, encoding='utf-8') as f:
                    data = json.load(f)
                cls._mem_cache[dataset_name] = data
                return data
            except Exception:
                return None
        return None

    @classmethod
    def put(cls, dataset_name, val):
        if not dataset_name: return
        cls._mem_cache[dataset_name] = val
        path = cls._get_path(dataset_name)
        if path:
            try:
                payload = val if isinstance(val, dict) else {"beta": val}
                payload["timestamp"] = time.time()
                with open(path, "w", encoding='utf-8') as f:
                    json.dump(payload, f)
            except Exception as e:
                print(f"[ASPIREBetaCache] save failed: {e}")

    def summary(self):
        files = _glob.glob(os.path.join(self._cache_dir, "aspire_beta_*.json"))
        return {"type": "ASPIRE_Beta", "entries": len(self._mem_cache), "files": len(files)}

    def invalidate(self, key=None):
        if key:
            self._mem_cache.pop(key, None)
            path = self._get_path(key)
            if path and os.path.exists(path): os.remove(path)
        else:
            self._mem_cache.clear()
            for f in _glob.glob(os.path.join(self._cache_dir, "aspire_beta_*.json")):
                os.remove(f)

_BetaCache = ASPIREBetaCacheManager

class GramMatrixCacheManager(GlobalCacheManager):
    """Gram 행렬 (XᵀX) 영속 캐시."""
    _mem_cache: dict = {}
    _cache_dir: str  = "data_cache"

    @classmethod
    def _get_path(cls, dataset_name):
        if not dataset_name: return None
        os.makedirs(cls._cache_dir, exist_ok=True)
        return os.path.join(cls._cache_dir, f"gram_{dataset_name}.pt")

    @classmethod
    def get(cls, dataset_name, device="cpu"):
        if not dataset_name: return None
        if dataset_name in cls._mem_cache:
            return cls._mem_cache[dataset_name].to(device)
        path = cls._get_path(dataset_name)
        if path and os.path.exists(path):
            try:
                val = torch.load(path, map_location=device)
                cls._mem_cache[dataset_name] = val.cpu()
                return val
            except Exception:
                return None
        return None

    @classmethod
    def put(cls, dataset_name, val: torch.Tensor):
        if not dataset_name: return
        cls._mem_cache[dataset_name] = val.cpu()
        path = cls._get_path(dataset_name)
        if path:
            try:
                torch.save(val.cpu(), path)
            except Exception as e:
                print(f"[GramCache] save failed: {e}")

    def summary(self):
        files = _glob.glob(os.path.join(self._cache_dir, "gram_*.pt"))
        return {"type": "Gram",
                "cached_datasets": list(self._mem_cache.keys()),
                "files": len(files)}

    def invalidate(self, key=None):
        if key:
            self._mem_cache.pop(key, None)
            path = self._get_path(key)
            if path and os.path.exists(path): os.remove(path)
        else:
            self._mem_cache.clear()
            for f in _glob.glob(os.path.join(self._cache_dir, "gram_*.pt")):
                os.remove(f)

_GramCache = GramMatrixCacheManager

# ==============================================================================
# AspireEngine
# ==============================================================================

class AspireEngine:
    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        if torch.is_tensor(x):
            return x.detach().cpu().numpy().astype(float)
        return np.asarray(x, dtype=float)

    @staticmethod
    def compute_spp(V, item_frequencies) -> np.ndarray:
        V_np    = AspireEngine._to_numpy(V)
        n_i     = AspireEngine._to_numpy(item_frequencies).flatten()
        n_max   = n_i.max() + 1e-12
        p_i     = n_i / n_max
        p_tilde = (V_np ** 2).T @ p_i
        return p_tilde

    @staticmethod
    def estimate_beta(
        singular_values: torch.Tensor,
        p_tilde: np.ndarray,
        trim: float = 0.0,
        verbose: bool = True,
        dataset_name: str = "",
        estimator_type: str = "lad",
        item_freq: np.ndarray = None,
        n_items: int = None,
        n_users: int = None,
        q: float | str = 0.5,
        **kwargs
    ) -> tuple:
        s  = np.sort(np.abs(AspireEngine._to_numpy(singular_values)))[::-1]
        pt = np.asarray(p_tilde, dtype=float)
        estimator_type = estimator_type.lower() if isinstance(estimator_type, str) else "ols"

        if estimator_type == "ols":
            beta, r2 = beta_estimators.beta_ols(s, pt)
            diag = {}
        elif estimator_type == "lad":
            beta, r2 = beta_estimators.beta_lad(s, pt)
            diag = {}
        elif estimator_type == "vector_opt":
            beta, r2, diag = beta_estimators.estimate_vector_beta(s, pt)
        elif estimator_type == "smooth_vector":
            beta, r2, diag = beta_estimators.smooth_estimate_vector_opt(s, pt)
        elif estimator_type == "iso_detrend":
            beta, r2 = beta_estimators.estimate_beta_with_detrending(s)
            diag = {}
        elif estimator_type == "iso_no_detrend":
            beta, r2 = beta_estimators.estimate_beta_no_detrending(s)
            diag = {}
        elif estimator_type == "decoupling":
            beta, r2 = beta_estimators.estimate_beta_decoupling(s, pt)
            diag = {}
        else:
            beta, r2 = beta_estimators.beta_ols(s, pt)
            diag = {}

        if verbose:
            if isinstance(beta, np.ndarray):
                print(f"[ASPIRE] {dataset_name} ({estimator_type}): β(mean)={np.mean(beta):.4f}, R²={r2:.4f}")
            else:
                print(f"[ASPIRE] {dataset_name} ({estimator_type}): β={beta:.4f}, R²={r2:.4f}")
        
        # Ensure beta goes out as vector if requested or scalar
        out_beta = beta if isinstance(beta, np.ndarray) else float(beta)
        return out_beta, float(r2), diag

    @staticmethod
    def apply_filter(s, alpha: float, beta):
        if torch.is_tensor(s):
            if isinstance(beta, np.ndarray):
                beta_t = torch.from_numpy(beta).float().to(s.device)
            elif torch.is_tensor(beta):
                beta_t = beta.float().to(s.device)
            else:
                beta_t = float(beta)
            exponent = 2.0 / (1.0 + beta_t)
            sp = torch.pow(torch.clamp(s.float(), min=1e-9), exponent)
            return (sp / (sp + float(alpha))).float()
        else:
            if torch.is_tensor(beta):
                beta_np = beta.detach().cpu().numpy()
            elif isinstance(beta, np.ndarray):
                beta_np = beta
            else:
                beta_np = float(beta)
            exponent = 2.0 / (1.0 + beta_np)
            sp = np.power(np.clip(s, 1e-9, None), exponent)
            return (sp / (sp + float(alpha))).astype(np.float32)

# ==============================================================================
# ASPIRELayer
# ==============================================================================

class ASPIRELayer(nn.Module):
    def __init__(
        self,
        k: int = 200,
        alpha: float = 500.0,
        beta: float | str = "auto",
        estimator_type: str = "lad",
        symmetric_norm: bool = False,
        **kwargs
    ):
        super().__init__()
        self.k = int(k)
        self.alpha = float(alpha)
        self.beta_config = beta
        self.estimator_type = estimator_type
        self.symmetric_norm = symmetric_norm
        self.q = kwargs.get("q", 0.5)

        self.beta = 0.5
        self.r_squared = 0.0

        self.register_buffer("singular_values", torch.empty(0))
        self.register_buffer("V_raw",           torch.empty(0, 0))
        self.register_buffer("filter_diag",     torch.empty(0))
        self.register_buffer("user_norm_weights", torch.empty(0))
        self.register_buffer("item_norm_weights", torch.empty(0))

    @property
    def V_k(self) -> torch.Tensor:
        return self.V_raw

    @torch.no_grad()
    def build(self, X_sparse, dataset_name: str | None = None):
        dev = next((p.device for p in self.parameters()), torch.device("cpu"))
        manager = EVDCacheManager(device=dev)
        item_pops_raw = np.array(X_sparse.sum(axis=0)).flatten().astype(float)
        M, N = X_sparse.shape

        if self.symmetric_norm:
            import scipy.sparse as sp
            user_sums = np.array(X_sparse.sum(axis=1)).flatten()
            item_sums = np.array(X_sparse.sum(axis=0)).flatten()
            def get_inv_sqrt(sums):
                inv_sqrt = np.zeros_like(sums)
                mask = sums > 0
                inv_sqrt[mask] = np.power(sums[mask], -0.5)
                return inv_sqrt
            w_u = get_inv_sqrt(user_sums)
            w_i = get_inv_sqrt(item_sums)
            self.register_buffer("user_norm_weights", torch.from_numpy(w_u).float().to(dev))
            self.register_buffer("item_norm_weights", torch.from_numpy(w_i).float().to(dev))
            X_target = sp.diags(w_u) @ X_sparse @ sp.diags(w_i)
            svd_dataset_name = f"{dataset_name}_norm" if dataset_name else None
            _, s, v, _ = manager.get_evd(X_target, dataset_name=svd_dataset_name)
            item_pops = np.array(X_target.sum(axis=0)).flatten().astype(float)
        else:
            _, s, v, _ = manager.get_evd(X_sparse, dataset_name=dataset_name)
            item_pops = item_pops_raw

        self.k = len(s)
        self.register_buffer("singular_values", s.to(dev))
        self.register_buffer("V_raw", v.to(dev))

        p_tilde = AspireEngine.compute_spp(self.V_raw.cpu().numpy(), item_pops)
        res = AspireEngine.estimate_beta(
            self.singular_values, p_tilde,
            verbose=True, dataset_name=dataset_name or "?",
            estimator_type=self.estimator_type,
            item_freq=item_pops, n_items=N, n_users=M,
            q=self.q
        )
        self.beta, self.r_squared = res[:2]
        
        applied_beta = float(self.beta_config) if not isinstance(self.beta_config, str) else self.beta
        self.beta = applied_beta  
        h = AspireEngine.apply_filter(self.singular_values, self.alpha, self.beta)
        self.register_buffer("filter_diag", h)

        beta_print = float(np.mean(AspireEngine._to_numpy(self.beta))) if isinstance(self.beta, (np.ndarray, torch.Tensor)) else float(self.beta)
        print(f"[ASPIRELayer] build complete | k={len(s)}  β(mean)={beta_print:.4f}  method={self.estimator_type}")

    @torch.no_grad()
    def forward(self, X_batch: torch.Tensor, user_ids=None) -> torch.Tensor:
        X = X_batch
        if self.symmetric_norm:
            if user_ids is not None:
                u_w = self.user_norm_weights[user_ids].view(-1, 1)
                X = X * u_w
            X = X * self.item_norm_weights.view(1, -1)
        XV = torch.mm(X, self.V_raw)
        scores = torch.mm(XV * self.filter_diag, self.V_raw.t())
        if self.symmetric_norm:
            scores = scores * self.item_norm_weights.view(1, -1)
        return scores

    @torch.no_grad()
    def visualize_matrices(self, X_sparse=None, save_dir=None, lightweight=False):
        ASPIREVisualizer.visualize_aspire_spectral(
            self.singular_values, self.filter_diag, self.alpha,
            beta=self.beta, X_sparse=X_sparse, save_dir=save_dir, file_prefix="aspire"
        )

# ==============================================================================
# ChebyASPIRELayer
# ==============================================================================

class ChebyASPIRELayer(nn.Module):
    def __init__(
        self,
        alpha: float = 500.0,
        degree: int = 20,
        beta: float | str = "auto",
        lambda_max_estimate: float | str = "auto",
        estimator_type: str = "lad",
        symmetric_norm: bool = False,
        **kwargs
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.degree = int(degree)
        self.beta_config = beta
        self.lambda_max_estimate = lambda_max_estimate
        self.estimator_type = estimator_type
        self.symmetric_norm = symmetric_norm

        self.beta = 0.5
        self.r_squared = 0.0

        self.register_buffer("cheby_coeffs", torch.empty(0))
        self.register_buffer("t_mid",        torch.tensor(0.0))
        self.register_buffer("t_half",       torch.tensor(0.0))
        self.register_buffer("item_weights", torch.empty(0))
        self.register_buffer("user_norm_weights", torch.empty(0))
        self.register_buffer("item_norm_weights", torch.empty(0))

        self.X_torch_csr = None
        self.Xt_torch_csr = None
        self.sparse_device = None

    def _aspire_filter(self, lam: np.ndarray) -> np.ndarray:
        exponent = 1.0 / (1.0 + self.beta)
        lam_pow = np.power(np.maximum(lam, 1e-12), exponent)
        return lam_pow / (lam_pow + self.alpha)

    @torch.no_grad()
    def _estimate_lambda_max(self, X_csr, Xt_csr) -> float:
        # Sparse Matrix Multiplication might hang on MPS, force CPU
        X_cpu = X_csr.to("cpu")
        Xt_cpu = Xt_csr.to("cpu")
        v = torch.randn(X_cpu.shape[1], 1, device="cpu")
        v = v / (v.norm() + 1e-12)
        
        last_lambda = 0.0
        lambda_est = 0.0
        for _ in range(30):
            v_next = torch.sparse.mm(Xt_cpu, torch.sparse.mm(X_cpu, v))
            lambda_est = v_next.norm().item()
            if lambda_est < 1e-12: break
            v = v_next / lambda_est
            if abs(lambda_est - last_lambda) / (lambda_est + 1e-12) < 1e-4:
                break
            last_lambda = lambda_est
            
        return lambda_est * 1.01 # 1% safety margin

    def _compute_chebyshev_coeffs(self, lam_min, lam_max, K) -> np.ndarray:
        j = np.arange(K + 1)
        theta = np.pi * (j + 0.5) / (K + 1)
        mid, half = (lam_max + lam_min) / 2.0, (lam_max - lam_min) / 2.0
        lam_nodes = mid + half * np.cos(theta)
        f_nodes = self._aspire_filter(lam_nodes)
        coeffs = np.zeros(K + 1)
        for k in range(K + 1):
            coeffs[k] = (2.0 / (K + 1)) * np.sum(f_nodes * np.cos(k * theta))
        coeffs[0] /= 2.0
        return coeffs

    @torch.no_grad()
    def build(self, X_sparse, dataset_name: str | None = None):
        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.sparse_device = torch.device("cpu" if "mps" in dev else dev)
        M, N = X_sparse.shape

        if self.symmetric_norm:
            import scipy.sparse as sp
            user_sums = np.array(X_sparse.sum(axis=1)).flatten()
            item_sums = np.array(X_sparse.sum(axis=0)).flatten()
            def get_inv_sqrt(sums):
                inv_sqrt = np.zeros_like(sums)
                mask = sums > 0
                inv_sqrt[mask] = np.power(sums[mask], -0.5)
                return inv_sqrt
            w_u, w_i = get_inv_sqrt(user_sums), get_inv_sqrt(item_sums)
            self.register_buffer("user_norm_weights", torch.from_numpy(w_u).float().to(dev))
            self.register_buffer("item_norm_weights", torch.from_numpy(w_i).float().to(dev))
            X_sparse = sp.diags(w_u) @ X_sparse @ sp.diags(w_i)

        X_coo = X_sparse.tocoo()
        indices = torch.from_numpy(np.vstack((X_coo.row, X_coo.col))).long()
        values = torch.from_numpy(X_coo.data).float()
        self.X_torch_csr = torch.sparse_coo_tensor(indices, values, X_coo.shape, device=self.sparse_device).coalesce().to_sparse_csr()
        self.Xt_torch_csr = torch.sparse_coo_tensor(torch.stack([indices[1], indices[0]]), values, (X_coo.shape[1], X_coo.shape[0]), device=self.sparse_device).coalesce().to_sparse_csr()

        lambda_max = self._estimate_lambda_max(self.X_torch_csr, self.Xt_torch_csr) if self.lambda_max_estimate == "auto" else float(self.lambda_max_estimate)

        actual_name = f"{dataset_name}_norm" if (dataset_name and self.symmetric_norm) else dataset_name
        cache_key = f"{actual_name}_aspire_v26" if actual_name else None
        cached = _BetaCache.get(cache_key) if cache_key else None

        if isinstance(self.beta_config, str):
            if cached is not None:
                cached_b = cached["beta"]
                self.beta = float(np.mean(cached_b)) if isinstance(cached_b, (list, np.ndarray)) else float(cached_b)
            else:
                curr_item_pops = np.array(X_sparse.sum(axis=0)).flatten()
                
                # SVD-free 전용 에스티메이터 선택
                est_type = self.estimator_type.lower()
                if est_type == "iso_pop_detrend":
                    self.beta, _ = beta_estimators.estimate_beta_with_detrending(curr_item_pops, is_svd=False)
                elif est_type == "iso_pop_no_detrend":
                    self.beta, _ = beta_estimators.estimate_beta_no_detrending(curr_item_pops, is_svd=False)
                else: # Default: max_median
                    self.beta, _ = beta_estimators.estimate_beta_max_median(curr_item_pops)
                
                if cache_key:
                    _BetaCache.put(cache_key, {"beta": self.beta})
        else:
            self.beta = float(self.beta_config)

        coeffs = self._compute_chebyshev_coeffs(0.0, lambda_max, self.degree)
        self.register_buffer("cheby_coeffs", torch.from_numpy(coeffs).float().to(dev))
        self.register_buffer("t_mid", torch.tensor(lambda_max / 2.0, device=dev))
        self.register_buffer("t_half", torch.tensor(lambda_max / 2.0, device=dev))

        n = X_sparse.shape[1]
        L = _GramCache.get(dataset_name, device=dev)
        if L is not None and L.shape[0] != n:
            L = None # Shape mismatch, ignore cache
            
        if L is None and n <= 15000:
            X_dense = self.X_torch_csr.to_dense().to(dev)
            L = torch.mm(X_dense.t(), X_dense)
            if dataset_name: _GramCache.put(dataset_name, L)
        
        if L is not None:
            self.item_weights = self._dense_chebyshev(L, coeffs, n, dev)
        else:
            self.item_weights = torch.empty(0)

        print(f"[ChebyASPIRELayer] build complete | β={self.beta:.4f}  λ_max={lambda_max:.2f}")

    def _dense_chebyshev(self, L, coeffs, n, dev) -> torch.Tensor:
        T_prev = torch.eye(n, device=dev)
        T_curr = (L - self.t_mid * T_prev) / self.t_half
        W = float(coeffs[0]) * T_prev + float(coeffs[1]) * T_curr
        for k in range(2, self.degree + 1):
            T_next = (2.0 * (torch.mm(L, T_curr) - self.t_mid * T_curr) / self.t_half) - T_prev
            W = W + float(coeffs[k]) * T_next
            T_prev, T_curr = T_curr, T_next
        return W

    @torch.no_grad()
    def forward(self, X_batch: torch.Tensor, user_ids=None) -> torch.Tensor:
        X = X_batch
        if self.symmetric_norm:
            if user_ids is not None:
                X = X * self.user_norm_weights[user_ids].view(-1, 1)
            X = X * self.item_norm_weights.view(1, -1)
        
        if self.item_weights.numel() > 0:
            scores = torch.mm(X, self.item_weights)
        else:
            scores = self._spmv_forward(X)
            
        if self.symmetric_norm:
            scores = scores * self.item_norm_weights.view(1, -1)
        return scores

    def _spmv_forward(self, X):
        X_t = X.t().to(self.sparse_device)
        T_prev = X_t
        with torch.no_grad():
            T_curr = (torch.sparse.mm(self.Xt_torch_csr, torch.sparse.mm(self.X_torch_csr, T_prev)) - self.t_mid * T_prev) / self.t_half
            W = float(self.cheby_coeffs[0]) * T_prev + float(self.cheby_coeffs[1]) * T_curr
            for k in range(2, self.degree + 1):
                T_next = (2.0 * (torch.sparse.mm(self.Xt_torch_csr, torch.sparse.mm(self.X_torch_csr, T_curr)) - self.t_mid * T_curr) / self.t_half) - T_prev
                W = W + float(self.cheby_coeffs[k]) * T_next
                T_prev, T_curr = T_curr, T_next
        return W.t().to(X.device)

    @torch.no_grad()
    def visualize_matrices(self, X_sparse=None, save_dir=None, lightweight=False):
        sigmas = np.linspace(1.0, 1e-3, 500)
        filter_diag = self._aspire_filter(sigmas**2)
        ASPIREVisualizer.visualize_aspire_spectral(
            torch.from_numpy(sigmas).float(), torch.from_numpy(filter_diag).float(),
            self.alpha, beta=self.beta, save_dir=save_dir, file_prefix="cheby_aspire"
        )
