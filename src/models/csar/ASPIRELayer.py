from __future__ import annotations

import os
import json
import time
import glob as _glob

import numpy as np
import torch
import torch.nn as nn

from src.utils.gpu_accel import SVDCacheManager, EVDCacheManager, GramMatrixCacheManager
from src.models.csar.aspire_visualizer import ASPIREVisualizer
from src.utils.cache_manager import GlobalCacheManager

from src.utils.cache_manager import GlobalCacheManager


# ==============================================================================
# Cheby Setup Cache (lambda_max)
# ==============================================================================

class ChebySetupCacheManager(GlobalCacheManager):
    """
    Persistent cache for dataset-specific parameters (lambda_max).
    These are independent of alpha/gamma hyperparameters.
    """
    _mem_cache: dict = {}
    _cache_dir: str  = "data_cache"

    @classmethod
    def get_lambda(cls, dataset_name, matrix_id):
        if not dataset_name or not matrix_id: return None
        key = f"cheby_lam_{dataset_name}_{matrix_id}"
        if key in cls._mem_cache: return cls._mem_cache[key]
        
        path = os.path.join(cls._cache_dir, f"{key}.json")
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)['lambda_max']
            except Exception: pass
        return None

    @classmethod
    def put_lambda(cls, dataset_name, matrix_id, lambda_max):
        if not dataset_name or not matrix_id: return
        os.makedirs(cls._cache_dir, exist_ok=True)
        key = f"cheby_lam_{dataset_name}_{matrix_id}"
        cls._mem_cache[key] = lambda_max
        path = os.path.join(cls._cache_dir, f"{key}.json")
        try:
            with open(path, 'w') as f:
                json.dump({'lambda_max': float(lambda_max)}, f)
        except Exception: pass

_ChebyCache = ChebySetupCacheManager


# ==============================================================================
# Cheby Basis Matrix Cache (T_k(L) X^T)
# ==============================================================================

class ChebyBasisCacheManager(GlobalCacheManager):
    """
    Persistent cache for Chebyshev basis matrices: Z_k = T_k(L) X^T.
    These are independent of alpha/gamma and depend only on dataset and degree.
    """
    _cache_dir = "data_cache"
    
    @classmethod
    def get_path(cls, dataset_name, matrix_id, k):
        return os.path.join(cls._cache_dir, f"cheby_basis_{dataset_name}_{matrix_id}_k{k}.pt")

    @classmethod
    def exists(cls, dataset_name, matrix_id, degree):
        if not dataset_name or not matrix_id: return False
        # Check if all k up to degree exist
        for k in range(degree + 1):
            if not os.path.exists(cls.get_path(dataset_name, matrix_id, k)):
                return False
        return True

    @classmethod
    def load(cls, dataset_name, matrix_id, degree, device="cpu"):
        bases = []
        for k in range(degree + 1):
            path = cls.get_path(dataset_name, matrix_id, k)
            bases.append(torch.load(path, map_location=device, weights_only=True))
        return bases

    @classmethod
    def save(cls, dataset_name, matrix_id, k, val: torch.Tensor):
        if not dataset_name or not matrix_id: return
        os.makedirs(cls._cache_dir, exist_ok=True)
        path = cls.get_path(dataset_name, matrix_id, k)
        torch.save(val.cpu(), path)


_BasisCache = ChebyBasisCacheManager


_GramCache = GramMatrixCacheManager

# ==============================================================================
# AspireFilter Utility
# ==============================================================================

class AspireFilter:
    """
    ASPIRE 필터링 유틸리티.
    τ (tau) 재매개변수화 기반 Scale-invariant Gamma 필터.

    h(σ̃_k; γ, τ) = σ̃_k^γ / (σ̃_k^γ + τ^γ)

    - γ (gamma): 필터 경사도/형상. 커질수록 급격한 고역 억제.
    - τ (tau):   컷오프 임계값. σ̃_k = τ 에서 h = 0.5 (γ와 독립).

    완전 디커플링:
      dh/d(τ)|_{γ 고정} → 위치만 변경
      dh/d(γ)|_{τ 고정} → 형상만 변경
    """
    @staticmethod
    def apply_filter(vals: torch.Tensor, gamma: float = 1.0, alpha: float = 1.0, 
                    mode: str = 'gamma_only', is_gram: bool = False) -> tuple[torch.Tensor, float, float]:
        """
        Returns:
            - h (Tensor): Filter coefficients
            - alpha (float): The alpha parameter
            - alpha_abs (float): The effective lambda used in the denominator
        """
        s = torch.clamp(vals.float(), min=1e-12)
        exp = float(gamma) if not is_gram else float(gamma) / 2.0
        s_gamma = torch.pow(s, exp)
        
        if mode == 'gamma_only':
            # [Gamma-only] h = s^g / (s^g + s_max^g)
            # This ensures h(s_max) = 0.5.
            s_max_gamma = s_gamma.max().item()
            effective_lambda = s_max_gamma
            alpha_val = 1.0
        else:
            # [Standard/Tikhonov] uses provided alpha
            effective_lambda = float(alpha)
            alpha_val = float(alpha)

        h = s_gamma / (s_gamma + effective_lambda + 1e-10)
        
        return h.float(), float(alpha_val), float(effective_lambda)

    @staticmethod
    def compute_rho(singular_values: torch.Tensor) -> float:
        """
        SPP 진단 (rho): log(s_tilde) vs log(rank) 의 Power-law 적합성 (R^2).
        """
        s_vals = singular_values.detach().cpu().numpy()
        s_tilde = s_vals / (s_vals[0] + 1e-10)
        log_s = np.log(s_tilde + 1e-10)
        
        K = len(s_tilde)
        if K < 5: return 0.0
        
        log_rank = np.log(np.arange(1, K + 1))
        
        # OLS R^2
        cov = np.cov(log_rank, log_s)
        if cov[0, 0] < 1e-10: return 0.0
        
        beta_hat = cov[0, 1] / cov[0, 0]
        residuals = log_s - (beta_hat * log_rank + (np.mean(log_s) - beta_hat * np.mean(log_rank)))
        ss_res = np.var(residuals)
        ss_tot = np.var(log_s)
        
        rho = 1 - ss_res / (ss_tot + 1e-10)
        return float(np.clip(rho, 0, 1))

# ==============================================================================
# ASPIRELayer (Standard SVD-based)
# ==============================================================================

class ASPIRELayer(nn.Module):
    def __init__(self, k: int = 200, gamma: float = 1.0, alpha: float = 1.0, filter_mode: str = "gamma_only", **kwargs):
        super().__init__()
        self.k = k
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.target_energy = kwargs.get("target_energy", 0.9)
        self.filter_mode = filter_mode

        self.register_buffer("singular_values", torch.empty(0))
        self.register_buffer("V_raw",           torch.empty(0, 0))
        self.register_buffer("filter_diag",     torch.empty(0))
        self.alpha_abs = 0.0
        self.rho = 0.0

    @property
    def V_k(self) -> torch.Tensor:
        return self.V_raw

    @torch.no_grad()
    def build(self, X_sparse, dataset_name: str | None = None, device=None, verbose: bool = True):
        # Determine device: priority = explicit > buffers > cpu
        if device is not None:
            dev = torch.device(device)
        else:
            try:
                dev = next(self.buffers()).device
            except StopIteration:
                dev = torch.device("cpu")
        
        # 1. SVD (Shared Cache Support)
        manager = SVDCacheManager(device=dev)
        # [Optimization] Pass k=self.k to enable early truncation in cache manager
        _, s, v, _ = manager.get_svd(X_sparse, k=self.k, dataset_name=dataset_name)

        # 2. Energy-based or K-based Truncation
        if self.k is None and self.target_energy is not None:
            cumsum_ev = torch.cumsum(s, dim=0)
            k_energy = torch.where(cumsum_ev / (cumsum_ev[-1] + 1e-12) >= self.target_energy)[0]
            if len(k_energy) > 0:
                self.k = int(k_energy[0]) + 1
                s, v = s[:self.k], v[:, :self.k]
        
        self.k = len(s)
        self.register_buffer("singular_values", s.to(dev))
        self.register_buffer("V_raw", v.to(dev))

        # 3. Filtering
        h, self.alpha, self.alpha_abs = AspireFilter.apply_filter(
            self.singular_values, gamma=self.gamma, alpha=self.alpha, mode=self.filter_mode
        )
        self.register_buffer("filter_diag", h)
        self.rho = AspireFilter.compute_rho(self.singular_values)

        if verbose:
            print(f"[ASPIRELayer] Built | γ={self.gamma:.2f} α={self.alpha:.2f} ρ={self.rho:.4f}")

    @torch.no_grad()
    def forward(self, X_batch: torch.Tensor) -> torch.Tensor:
        XV = torch.mm(X_batch, self.V_raw)
        return torch.mm(XV * self.filter_diag, self.V_raw.t())

    @torch.no_grad()
    def visualize_matrices(self, X_sparse=None, save_dir=None, lightweight=False):
        ASPIREVisualizer.visualize_aspire_spectral(
            self.singular_values, self.filter_diag, alpha=self.alpha,
            gamma=self.gamma, alpha_abs=self.alpha_abs, 
            X_sparse=X_sparse, save_dir=save_dir, file_prefix="aspire"
        )

# ==============================================================================
# ChebyASPIRELayer (SVD-free / Gram-based)
# ==============================================================================

class ChebyASPIRELayer(nn.Module):
    def __init__(self, degree: int = 20, gamma: float = 1.0, 
                 alpha: float = 1.0, filter_mode: str = "gamma_only",
                 lambda_max_estimate: float | str = "auto", **kwargs):
        super().__init__()
        self.degree = int(degree)
        self.gamma = float(gamma)
        self.lambda_max_estimate = lambda_max_estimate
        self.filter_mode = filter_mode
        self.alpha = float(alpha)

        self.register_buffer("cheby_coeffs", torch.empty(0))
        self.register_buffer("t_mid",        torch.tensor(0.0))
        self.register_buffer("t_half",       torch.tensor(0.0))
        self.register_buffer("item_weights", torch.empty(0))
        self.alpha_abs = 0.0
        self.rho = 0.0

        self.X_torch_csr = None
        self.Xt_torch_csr = None
        self.sparse_device = None
        self.scores_cache = None  # Precomputed scores for all users
        self.matrix_id = None

    @torch.no_grad()
    def _estimate_lambda_max(self, X_csr, Xt_csr) -> float:
        """
        Power iteration to estimate the largest eigenvalue of Gram matrix (X^T X).
        Use GPU if CUDA is available, otherwise CPU (MPS sparse support is flaky).
        """
        device = X_csr.device
        # Use GPU for CUDA, but fallback to CPU for MPS due to sparse issues
        calc_device = device if device.type == 'cuda' else torch.device("cpu")
        
        X_local = X_csr.to(calc_device)
        Xt_local = Xt_csr.to(calc_device)
        
        v = torch.randn(X_local.shape[1], 1, device=calc_device)
        v /= (v.norm() + 1e-12)
        
        last_lambda = 0.0
        # 30 iterations is usually enough for spectral radius
        for _ in range(30):
            # L @ v = Xt @ (X @ v)
            v_next = torch.sparse.mm(Xt_local, torch.sparse.mm(X_local, v))
            lambda_est = v_next.norm().item()
            if lambda_est < 1e-12: break
            v = v_next / (lambda_est + 1e-12)
            
            if abs(lambda_est - last_lambda) / (lambda_est + 1e-12) < 1e-4: 
                break
            last_lambda = lambda_est
            
        return last_lambda * 1.01 

    def _compute_chebyshev_coeffs(self, lam_max, K) -> np.ndarray:
        j = np.arange(K + 1)
        theta = np.pi * (j + 0.5) / (K + 1)
        mid = half = lam_max / 2.0
        lam_nodes = mid + half * np.cos(theta)

        # Apply finalized filter
        lam_torch = torch.from_numpy(lam_nodes).float()
        h_nodes, self.alpha, self.alpha_abs = AspireFilter.apply_filter(
            lam_torch, gamma=self.gamma, alpha=self.alpha, mode=self.filter_mode, is_gram=True
        )

        f_nodes = h_nodes.numpy()
        coeffs = np.zeros(K + 1)
        for k in range(K + 1):
            coeffs[k] = (2.0 / (K + 1)) * np.sum(f_nodes * np.cos(k * theta))
        coeffs[0] /= 2.0
        return coeffs

    @torch.no_grad()
    def build(self, X_sparse, dataset_name: str | None = None, verbose: bool = True):
        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.sparse_device = torch.device("cpu" if "mps" in dev else dev)
        
        X_coo = X_sparse.tocoo()
        indices = torch.from_numpy(np.vstack((X_coo.row, X_coo.col))).long()
        values = torch.from_numpy(X_coo.data).float()
        self.X_torch_csr = torch.sparse_coo_tensor(indices, values, X_coo.shape, device=self.sparse_device).coalesce().to_sparse_csr()
        self.Xt_torch_csr = torch.sparse_coo_tensor(torch.stack([indices[1], indices[0]]), values, (X_coo.shape[1], X_coo.shape[0]), device=self.sparse_device).coalesce().to_sparse_csr()

        # --- Cache Logic for Lambda Max (Parameter Independent) ---
        self.matrix_id = EVDCacheManager._generate_matrix_id(X_sparse)
        lambda_max = _ChebyCache.get_lambda(dataset_name, self.matrix_id)
        
        if lambda_max is None:
            lambda_max = self._estimate_lambda_max(self.X_torch_csr, self.Xt_torch_csr) if self.lambda_max_estimate == "auto" else float(self.lambda_max_estimate)
            _ChebyCache.put_lambda(dataset_name, self.matrix_id, lambda_max)
        elif verbose:
            print(f"[ChebyASPIRE] Lambda Cache Hit | λ_max={lambda_max:.2f}")

        coeffs = self._compute_chebyshev_coeffs(lambda_max, self.degree)

        self.register_buffer("cheby_coeffs", torch.from_numpy(coeffs).float().to(dev))
        self.register_buffer("t_mid", torch.tensor(lambda_max / 2.0, device=dev))
        self.register_buffer("t_half", torch.tensor(lambda_max / 2.0, device=dev))

        n = X_sparse.shape[1]
        L = None
        
        # ML-20M (18k items), Yelp 10-core (45k items)
        # We increase threshold to 50,000 enabled by bfloat16 savings.
        if n <= 50000:
            try:
                print(f"[ChebyASPIRE] Computing dense Gram matrix G = X^T X (Items={n}) in bfloat16...")
                # Optimized Gram build: bfloat16 for memory efficiency (ADA GPU)
                from src.utils.gpu_accel import _build_gram
                L = _build_gram(X_sparse, dev)
            except Exception as e:
                print(f"[ChebyASPIRE] Dense Gram build failed: {e}")
                L = None
        
        if verbose:
            print(f"[ChebyASPIRE] Build done (Items: {n}, Threshold: 50,000)")
        
        # Kernel fusion for CUDA (ADA)
        if dev.type == 'cuda':
            try:
                self._dense_chebyshev = torch.compile(self._dense_chebyshev, mode='reduce-overhead')
            except Exception as e:
                print(f"[ChebyASPIRE] torch.compile skipped: {e}")
            print(f"[ChebyASPIRELayer] Built | γ={self.gamma:.2f} α={self.alpha:.2f} (abs={self.alpha_abs:.2f})")

    def _dense_chebyshev(self, L, coeffs, n, dev) -> torch.Tensor:
        T_prev = torch.eye(n, device=dev)
        T_curr = (L - self.t_mid * T_prev) / self.t_half
        W = float(coeffs[0]) * T_prev + float(coeffs[1]) * T_curr
        for k in range(2, self.degree + 1):
            T_next = (2.0 * (torch.mm(L, T_curr) - self.t_mid * T_curr) / self.t_half) - T_prev
            W += float(coeffs[k]) * T_next
            T_prev, T_curr = T_curr, T_next
        return W

    @torch.no_grad()
    def forward(self, X_batch: torch.Tensor) -> torch.Tensor:
        if self.item_weights.numel() > 0:
            return torch.mm(X_batch, self.item_weights)
        return self._spmv_forward(X_batch)

    def _spmv_forward(self, X):
        # Optimization: In-place addition and pre-computed constants
        X_t = X.t().to(self.sparse_device)
        T_prev = X_t
        mid_val = self.t_mid.to(self.sparse_device)
        half_val = self.t_half.to(self.sparse_device)
        inv_half = 1.0 / half_val
        
        with torch.no_grad():
            T_curr = (torch.sparse.mm(self.Xt_torch_csr, torch.sparse.mm(self.X_torch_csr, T_prev)) - mid_val * T_prev) * inv_half
            W = float(self.cheby_coeffs[0]) * T_prev + float(self.cheby_coeffs[1]) * T_curr
            
            coeff_inv_half = 2.0 * inv_half
            
            for k in range(2, self.degree + 1):
                # T_next = (2.0 * (L @ T_curr - mid * T_curr) / half) - T_prev
                temp = torch.sparse.mm(self.Xt_torch_csr, torch.sparse.mm(self.X_torch_csr, T_curr))
                T_next = coeff_inv_half * (temp - mid_val * T_curr) - T_prev
                W.add_(T_next, alpha=float(self.cheby_coeffs[k]))
                T_prev, T_curr = T_curr, T_next
        return W.t().to(X.device)

    @torch.no_grad()
    def precompute(self, X_all_sparse, dataset_name=None, matrix_id=None, device=None):
        """
        Memory-efficient precomputation.
        1. If N is small, compute item_weights (N x N) and use them.
        2. If scores_cache is needed, compute it in batches to avoid M x M intermediates.
        """
        print(f"[ChebyASPIRE] Precomputing (Items: {X_all_sparse.shape[1]}, Users: {X_all_sparse.shape[0]})...")
        t0 = time.time()
        
        N_users, N_items = X_all_sparse.shape
        # If we don't have item_weights yet, try to build them first (much smaller than scores_cache)
        if self.item_weights.numel() == 0 and N_items <= 25000:
            print(f"[ChebyASPIRE] Building item-wise filter (weights)...")
            I_n = torch.eye(N_items, device=device or self.sparse_device)
            self.item_weights = self._spmv_forward(I_n).to(device or self.sparse_device)
            del I_n

        # [ADA OPTIMIZATION] Option A: skip scores_cache if item_weights exist 
        # (77k x 45k x 4 = 14GB CPU RAM risk)
        if self.item_weights.numel() > 0:
            print(f"[ChebyASPIRE] scores_cache overflow risk (Items={N_items}), using on-the-fly batch inference.")
            self.scores_cache = None
            return
        else:
            # Absolute fallback: Batch-wise iterative expansion (Very slow but safe)
            print(f"[ChebyASPIRE] Fallback: Batch-wise iterative expansion...")
            batch_size = 5000
            all_scores = torch.zeros((N_users, N_items), device="cpu")
            for i in range(0, N_users, batch_size):
                end = min(i + batch_size, N_users)
                X_batch = torch.from_numpy(X_all_sparse[i:end].toarray()).float().to(self.sparse_device)
                all_scores[i:end] = self._spmv_forward(X_batch).cpu()
                del X_batch
            self.scores_cache = all_scores

        print(f"[ChebyASPIRE] Precomputation done: {time.time()-t0:.2f}s")

    @torch.no_grad()
    def visualize_matrices(self, X_sparse=None, save_dir=None, lightweight=False):
        # Normalize range for filter visualization [0, 1]
        sigmas = np.linspace(1.0, 1e-3, 500)
        lam_nodes = torch.from_numpy(sigmas**2).float()
        h_vals, eff_alpha, _ = AspireFilter.apply_filter(
            lam_nodes, gamma=self.gamma, is_gram=True
        )
        ASPIREVisualizer.visualize_aspire_spectral(
            torch.from_numpy(sigmas).float(), h_vals,
            self.alpha, gamma=self.gamma, 
            alpha_abs=self.alpha_abs, effective_alpha=eff_alpha, 
            save_dir=save_dir, file_prefix="cheby_aspire"
        )
