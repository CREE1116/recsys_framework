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
    def apply_filter(vals: torch.Tensor, tau: float, gamma: float, is_gram: bool = False) -> tuple[torch.Tensor, float]:
        """
        vals: Singular values (is_gram=False) or Eigenvalues (is_gram=True).
        h(σ̃) = σ̃^γ / (σ̃^γ + τ^γ)
        cut-off: σ̃ = τ → h = 0.5, regardless of gamma.
        """
        # 1. Normalize by max value (σ̃)
        v_max = vals.max().item() + 1e-12
        s_tilde = vals / v_max

        # 2. Compute exponent (eigenvalue = sigma^2, so use gamma/2)
        exp = float(gamma) if not is_gram else float(gamma) / 2.0
        s_gamma = torch.pow(torch.clamp(s_tilde.float(), min=1e-12), exp)

        # 3. tau^gamma as the cut-off level (completely decoupled from gamma shape)
        tau_gamma = float(tau) ** exp

        # 4. Filter
        h = s_gamma / (s_gamma + tau_gamma + 1e-10)
        return h.float(), tau_gamma

    @staticmethod
    def estimate_tau(singular_values: torch.Tensor, X_sparse=None, method: str = "mp") -> float:
        """
        데이터 기반 τ 자동 추정 (HPO 불필요).

        methods:
          'mp'           : Marchenko-Pastur noise edge
                           τ* = sqrt(nnz) / sigma_max
                           이론적으로 SNR=1이 되는 지점.
                           Sparse할수록 τ 작아짐(더 강한 복원).
          'spectral_gap' : log-scale 2차 미분이 최대인 지점(변곡점).
          'median'       : 정규화 특이값의 중간값.
        """
        s = singular_values.detach().cpu().numpy()
        sigma_max = s[0] + 1e-12
        s_tilde = s / sigma_max

        if method == "mp" and X_sparse is not None:
            # Marchenko-Pastur noise edge:
            # σ_noise ≈ sqrt(density) * σ_max  →  τ* = σ_noise / σ_max = sqrt(density)
            n_users, n_items = X_sparse.shape
            density = X_sparse.nnz / (n_users * n_items)
            tau = float(np.clip(np.sqrt(density), 0.01, 0.95))

        elif method == "spectral_gap":
            log_s = np.log(s_tilde + 1e-10)
            if len(log_s) < 5:
                return 0.3
            d2 = np.abs(np.diff(np.diff(log_s)))
            gap_idx = int(np.argmax(d2)) + 1
            tau = float(np.clip(s_tilde[gap_idx], 0.01, 0.99))

        else:  # 'median' fallback
            tau = float(np.clip(np.median(s_tilde), 0.01, 0.99))

        return tau

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
    def __init__(self, k: int = 200, tau: float = 0.3, gamma: float = 1.0, **kwargs):
        super().__init__()
        self.k = k
        self.tau = tau if isinstance(tau, str) else float(tau)  # 'auto' 허용
        self.gamma = float(gamma)
        self.target_energy = kwargs.get("target_energy", 0.9)

        self.register_buffer("singular_values", torch.empty(0))
        self.register_buffer("V_raw",           torch.empty(0, 0))
        self.register_buffer("filter_diag",     torch.empty(0))

        self.tau_gamma = 0.0  # τ^γ (effective cut-off level)
        self.rho = 0.0

    @property
    def V_k(self) -> torch.Tensor:
        return self.V_raw

    @torch.no_grad()
    def build(self, X_sparse, dataset_name: str | None = None, verbose: bool = True):
        dev = next((p.device for p in self.parameters()), torch.device("cpu"))
        manager = EVDCacheManager(device=dev)
        
        # 1. SVD
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

        # 3. Auto-estimate tau if requested
        if isinstance(self.tau, str) and self.tau == "auto":
            self.tau = AspireFilter.estimate_tau(self.singular_values, X_sparse=X_sparse, method="mp")

        # 4. Filtering with tau-reparameterization
        h, tau_g = AspireFilter.apply_filter(self.singular_values, self.tau, self.gamma)
        self.register_buffer("filter_diag", h)
        self.tau_gamma = tau_g
        self.rho = AspireFilter.compute_rho(self.singular_values)

        if verbose:
            print(f"[ASPIRELayer] Built | γ={self.gamma:.2f} τ={self.tau:.4f} (τ^γ={self.tau_gamma:.4f}) ρ={self.rho:.4f} k={self.k}")

    @torch.no_grad()
    def forward(self, X_batch: torch.Tensor) -> torch.Tensor:
        XV = torch.mm(X_batch, self.V_raw)
        return torch.mm(XV * self.filter_diag, self.V_raw.t())

    @torch.no_grad()
    def visualize_matrices(self, X_sparse=None, save_dir=None, lightweight=False):
        ASPIREVisualizer.visualize_aspire_spectral(
            self.singular_values, self.filter_diag, self.tau,
            gamma=self.gamma, effective_alpha=self.tau_gamma,
            X_sparse=X_sparse, save_dir=save_dir, file_prefix="aspire"
        )

# ==============================================================================
# ChebyASPIRELayer (SVD-free / Gram-based)
# ==============================================================================

class ChebyASPIRELayer(nn.Module):
    def __init__(self, tau: float = 0.3, degree: int = 20, gamma: float = 1.0, 
                 lambda_max_estimate: float | str = "auto", **kwargs):
        super().__init__()
        self.tau = tau if isinstance(tau, str) else float(tau)  # 'auto' 허용
        self.degree = int(degree)
        self.gamma = float(gamma)
        self.lambda_max_estimate = lambda_max_estimate

        self.register_buffer("cheby_coeffs", torch.empty(0))
        self.register_buffer("t_mid",        torch.tensor(0.0))
        self.register_buffer("t_half",       torch.tensor(0.0))
        self.register_buffer("item_weights", torch.empty(0))

        self.tau_gamma = 0.0  # τ^γ (effective cut-off level)
        self.rho = 0.0

        self.X_torch_csr = None
        self.Xt_torch_csr = None
        self.sparse_device = None

    @torch.no_grad()
    def _estimate_lambda_max(self, X_csr, Xt_csr) -> float:
        X_cpu, Xt_cpu = X_csr.to("cpu"), Xt_csr.to("cpu")
        v = torch.randn(X_cpu.shape[1], 1, device="cpu")
        v /= (v.norm() + 1e-12)
        
        last_lambda = 0.0
        for _ in range(30):
            v_next = torch.sparse.mm(Xt_cpu, torch.sparse.mm(X_cpu, v))
            lambda_est = v_next.norm().item()
            if lambda_est < 1e-12: break
            v = v_next / lambda_est
            if abs(lambda_est - last_lambda) / (lambda_est + 1e-12) < 1e-4: break
            last_lambda = lambda_est
        return last_lambda * 1.01 

    def _compute_chebyshev_coeffs(self, lam_max, K) -> np.ndarray:
        j = np.arange(K + 1)
        theta = np.pi * (j + 0.5) / (K + 1)
        mid = half = lam_max / 2.0
        lam_nodes = mid + half * np.cos(theta)

        # Apply tau-reparameterized filter
        lam_torch = torch.from_numpy(lam_nodes).float()
        h_nodes, tau_g = AspireFilter.apply_filter(lam_torch, self.tau, self.gamma, is_gram=True)
        self.tau_gamma = tau_g

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

        if verbose:
            print(f"[ChebyASPIRELayer] Built | γ={self.gamma:.2f} τ={self.tau:.4f} (τ^γ={self.tau_gamma:.4f}) λ_max={lambda_max:.2f}")

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
        X_t = X.t().to(self.sparse_device)
        T_prev = X_t
        with torch.no_grad():
            T_curr = (torch.sparse.mm(self.Xt_torch_csr, torch.sparse.mm(self.X_torch_csr, T_prev)) - self.t_mid * T_prev) / self.t_half
            W = float(self.cheby_coeffs[0]) * T_prev + float(self.cheby_coeffs[1]) * T_curr
            for k in range(2, self.degree + 1):
                T_next = (2.0 * (torch.sparse.mm(self.Xt_torch_csr, torch.sparse.mm(self.X_torch_csr, T_curr)) - self.t_mid * T_curr) / self.t_half) - T_prev
                W += float(self.cheby_coeffs[k]) * T_next
                T_prev, T_curr = T_curr, T_next
        return W.t().to(X.device)

    @torch.no_grad()
    def visualize_matrices(self, X_sparse=None, save_dir=None, lightweight=False):
        sigmas = np.linspace(1.0, 1e-3, 500)
        h_vals, tau_g = AspireFilter.apply_filter(torch.from_numpy(sigmas**2).float(), self.tau, self.gamma, is_gram=True)
        ASPIREVisualizer.visualize_aspire_spectral(
            torch.from_numpy(sigmas).float(), h_vals,
            self.tau, gamma=self.gamma, effective_alpha=tau_g,
            save_dir=save_dir, file_prefix="cheby_aspire"
        )
