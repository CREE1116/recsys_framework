from __future__ import annotations

import os
import json
import time
import glob as _glob

import numpy as np
import torch
import torch.nn as nn

from src.utils.gpu_accel import SVDCacheManager, EVDCacheManager
from src.models.csar.aspire_visualizer import ASPIREVisualizer
from src.utils.cache_manager import GlobalCacheManager

# ==============================================================================
# Gram Matrix Cache (XᵀX)
# ==============================================================================

class GramMatrixCacheManager(GlobalCacheManager):
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
                val = torch.load(path, map_location=device, weights_only=True)
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
    def apply_filter(vals: torch.Tensor, gamma: float = 1.0, is_gram: bool = False) -> tuple[torch.Tensor, float, float]:
        """
        Returns:
            - h (Tensor): Filter coefficients
            - alpha (float): The alpha parameter (fixed to 1.0)
            - alpha_abs (float): The effective lambda used in the denominator (s_max^gamma)
        """
        s = torch.clamp(vals.float(), min=1e-12)
        exp = float(gamma) if not is_gram else float(gamma) / 2.0
        s_gamma = torch.pow(s, exp)
        
        # [Gamma-only] h = s^g / (s^g + s_max^g)
        # This ensures h(s_max) = 0.5.
        s_max_gamma = s_gamma.max().item()
        effective_lambda = s_max_gamma
        h = s_gamma / (s_gamma + effective_lambda + 1e-10)
        
        return h.float(), 1.0, float(effective_lambda)

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
    def __init__(self, k: int = 200, gamma: float = 1.0, **kwargs):
        super().__init__()
        self.k = k
        self.gamma = float(gamma)
        self.alpha = 1.0  # Fixed in gamma_only mode
        self.target_energy = kwargs.get("target_energy", 0.9)
        self.filter_mode = "gamma_only"

        self.register_buffer("singular_values", torch.empty(0))
        self.register_buffer("V_raw",           torch.empty(0, 0))
        self.register_buffer("filter_diag",     torch.empty(0))
        self.alpha_abs = 0.0
        self.rho = 0.0

    @property
    def V_k(self) -> torch.Tensor:
        return self.V_raw

    @torch.no_grad()
    def build(self, X_sparse, dataset_name: str | None = None, verbose: bool = True):
        dev = next((p.device for p in self.parameters()), torch.device("cpu"))
        
        # 1. SVD
        manager = EVDCacheManager(device=dev)
        _, s, v, _ = manager.get_evd(X_sparse, dataset_name=dataset_name)

        # 2. Energy-based Truncation
        if self.k is None and self.target_energy is not None:
            cumsum_ev = torch.cumsum(s**2, dim=0)
            k_energy = torch.where(cumsum_ev / (cumsum_ev[-1] + 1e-12) >= self.target_energy)[0]
            if len(k_energy) > 0:
                self.k = int(k_energy[0]) + 1
                s, v = s[:self.k], v[:, :self.k]
        
        self.k = len(s)
        self.register_buffer("singular_values", s.to(dev))
        self.register_buffer("V_raw", v.to(dev))

        # 3. Filtering
        h, self.alpha, self.alpha_abs = AspireFilter.apply_filter(
            self.singular_values, gamma=self.gamma
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
                 lambda_max_estimate: float | str = "auto", **kwargs):
        super().__init__()
        self.degree = int(degree)
        self.gamma = float(gamma)
        self.lambda_max_estimate = lambda_max_estimate
        self.filter_mode = "gamma_only"
        self.alpha = 1.0

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
            lam_torch, gamma=self.gamma, is_gram=True
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

        lambda_max = self._estimate_lambda_max(self.X_torch_csr, self.Xt_torch_csr) if self.lambda_max_estimate == "auto" else float(self.lambda_max_estimate)
        coeffs = self._compute_chebyshev_coeffs(lambda_max, self.degree)

        self.register_buffer("cheby_coeffs", torch.from_numpy(coeffs).float().to(dev))
        self.register_buffer("t_mid", torch.tensor(lambda_max / 2.0, device=dev))
        self.register_buffer("t_half", torch.tensor(lambda_max / 2.0, device=dev))

        n = X_sparse.shape[1]
        L = _GramCache.get(dataset_name, device=dev)
        if L is None and n <= 15000:
            L = torch.mm(self.X_torch_csr.to_dense().to(dev).t(), self.X_torch_csr.to_dense().to(dev))
            if dataset_name: _GramCache.put(dataset_name, L)
        
        self.item_weights = self._dense_chebyshev(L, coeffs, n, dev) if L is not None else torch.empty(0)
        self.scores_cache = None # Reset cache on rebuild

        if verbose:
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
        with torch.no_grad():
            T_curr = (torch.sparse.mm(self.Xt_torch_csr, torch.sparse.mm(self.X_torch_csr, T_prev)) - self.t_mid * T_prev) / self.t_half
            W = float(self.cheby_coeffs[0]) * T_prev + float(self.cheby_coeffs[1]) * T_curr
            
            inv_half = 2.0 / self.t_half
            mid_val = self.t_mid
            
            for k in range(2, self.degree + 1):
                # T_next = (2.0 * (L @ T_curr - mid * T_curr) / half) - T_prev
                temp = torch.sparse.mm(self.Xt_torch_csr, torch.sparse.mm(self.X_torch_csr, T_curr))
                T_next = inv_half * (temp - mid_val * T_curr) - T_prev
                W.add_(T_next, alpha=float(self.cheby_coeffs[k]))
                T_prev, T_curr = T_curr, T_next
        return W.t().to(X.device)

    @torch.no_grad()
    def precompute(self, X_all_sparse, device=None):
        """
        Precompute scores for all users in X_all_sparse.
        Useful for speeding up evaluation on large datasets.
        """
        print(f"[ChebyASPIRE] Precomputing scores for all {X_all_sparse.shape[0]} users...")
        t0 = time.time()
        
        # We need to process in batches if X_all is too large to handle at once in _spmv_forward
        N = X_all_sparse.shape[0]
        batch_size = 5000 # Adjustable
        all_scores = []
        
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            batch_indices = np.arange(i, end)
            X_batch_dense = torch.from_numpy(X_all_sparse[batch_indices].toarray()).float().to(device or self.sparse_device)
            all_scores.append(self.forward(X_batch_dense).cpu())
            
        self.scores_cache = torch.cat(all_scores, dim=0)
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
