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
    ASPIRE 싱글 파라미터(1-Parameter) 필터링 유틸리티.
    h(σ_k; γ) = σ_k^γ / (σ_k^γ + τ^γ)

    - γ (gamma): Global Compression 파라미터.
    - sigma_ref: gamma_only 모드에서 정규화 기준값 선택
        * 'sigma1'      (기본): 최대 특이값 σ_1 사용 (원본 ASPIRE)
        * 'sigma_median': 특이값 중앙값 σ_median 사용
        * 'sigma_mean'  : 특이값 평균값 σ_mean 사용
        * 'sigma_k'     : 마지막 특이값 σ_k 사용
    - α (alpha): Ranking Invariance 정리에 의해, 선형 추천 모델에서 스케일 계수 α는
      최종 추천 리스트의 순위에 어떠한 수학적 영향도 미치지 못함. (gamma_only 모드 권장)
    """
    @staticmethod
    def apply_filter(vals: torch.Tensor, gamma: float = 1.0, alpha: float = 1.0,
                    mode: str = 'gamma_only', is_gram: bool = False,
                    sigma_ref: str = 'sigma1', custom_lambda: float = None) -> tuple[torch.Tensor, float, float]:
        s = torch.clamp(vals.float(), min=1e-12)
        exp = float(gamma) if not is_gram else float(gamma) / 2.0
        s_gamma = torch.pow(s, exp)
        
        if mode == 'gamma_only':
            # --- Normalization reference selection ---
            # 의미론: "해당 σ 참조값에서 h(σ_ref) = 0.5" 가 되도록 τ = σ_ref^γ 로 설정.
            # 외부에서 동적으로 산출된 람다(custom_lambda = τ^γ)가 있다면 이를 우선 적용.
            if custom_lambda is not None:
                effective_lambda = float(custom_lambda)
            elif sigma_ref == 'sigma_median':
                # τ = median(σ)^γ  →  h(σ_median) = 0.5
                effective_lambda = float(torch.median(s).item()) ** exp
            elif sigma_ref == 'sigma_mean':
                # τ = mean(σ)^γ   →  h(σ_mean) = 0.5
                # (주의: mean(σ^γ) ≠ mean(σ)^γ — Jensen 부등식. 후자가 의미론적으로 정확)
                effective_lambda = float(s.mean().item()) ** exp
            elif sigma_ref == 'sigma_k':
                # τ = σ_k^γ  →  h(σ_k) = 0.5 at the truncation boundary.
                # 가장 작은 retention 특이값(σ_k = s[-1])을 기준으로 삼아
                # 모든 retained 성분에서 h ≥ 0.5 를 보장.
                # k가 크면 σ_k ↓ → τ ↓ → 전체 필터 완화; k가 작으면 반대.
                # → k와 γ가 자연스럽게 coupling 되는 효과.
                effective_lambda = s_gamma[-1].item()  # s_gamma[-1] = σ_k^exp
            else:  # 'sigma1' (default): τ = σ₁^γ  →  h(σ₁) = 0.5
                effective_lambda = s_gamma.max().item()  # max(σ^γ) == (max σ)^γ, same by monotonicity
            alpha_val = 1.0
        else:
            effective_lambda = float(alpha)
            alpha_val = float(alpha)

        h = s_gamma / (s_gamma + effective_lambda + 1e-10)
        
        return h.float(), float(alpha_val), float(effective_lambda)

    @staticmethod
    def compute_rho(singular_values: torch.Tensor) -> float:
        """
        SPP 진단 (rho): log(s_tilde) vs log(rank) 의 Power-law 적합도 (R^2).
        """
        s_vals = singular_values.detach().cpu().numpy()
        s_tilde = s_vals / (s_vals[0] + 1e-10)
        log_s = np.log(s_tilde + 1e-10)
        
        K = len(s_tilde)
        if K < 5: return 0.0
        
        log_rank = np.log(np.arange(1, K + 1))
        
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
    def __init__(self, k: int = 200, gamma: float = 1.0, alpha: float = 1.0,
                 filter_mode: str = "gamma_only", sigma_ref: str = 'sigma1', **kwargs):
        super().__init__()
        self.k = k
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.target_energy = kwargs.get("target_energy", 0.9)
        self.filter_mode = filter_mode
        # sigma_ref: normalization reference in gamma_only mode
        # 'sigma1' (default) | 'sigma_median' | 'sigma_mean'
        self.sigma_ref = sigma_ref

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
        if device is not None:
            dev = torch.device(device)
        else:
            try:
                dev = next(self.buffers()).device
            except StopIteration:
                dev = torch.device("cpu")
        
        manager = SVDCacheManager(device=dev)
        _, s, v, _ = manager.get_svd(X_sparse, k=self.k, dataset_name=dataset_name)

        if self.k is None and self.target_energy is not None:
            cumsum_ev = torch.cumsum(s, dim=0)
            k_energy = torch.where(cumsum_ev / (cumsum_ev[-1] + 1e-12) >= self.target_energy)[0]
            if len(k_energy) > 0:
                self.k = int(k_energy[0]) + 1
                s, v = s[:self.k], v[:, :self.k]
        
        self.k = len(s)
        self.register_buffer("singular_values", s.to(dev))
        self.register_buffer("V_raw", v.to(dev))

        custom_lam = None
        if self.sigma_ref in ['sigma_global', 'sigma_tail']:
            total_energy = X_sparse.nnz
            max_rank = min(X_sparse.shape)
            
            if self.sigma_ref == 'sigma_global':
                # λ = M / min(U,I)  (에너지 평균)
                custom_lam = total_energy / max_rank
            else:  # 'sigma_tail'
                # λ = (M - sum(s_retained^2)) / (min(U,I) - K)  (꼬리 노이즈 에너지 평균)
                explained_energy = (self.singular_values ** 2).sum().item()
                if max_rank > self.k:
                    custom_lam = max(0.0, total_energy - explained_energy) / (max_rank - self.k)
                else:
                    custom_lam = total_energy / max_rank

        h, self.alpha, self.alpha_abs = AspireFilter.apply_filter(
            self.singular_values, gamma=self.gamma, alpha=self.alpha,
            mode=self.filter_mode, sigma_ref=self.sigma_ref, custom_lambda=custom_lam
        )
        self.register_buffer("filter_diag", h)
        self.rho = AspireFilter.compute_rho(self.singular_values)

        if verbose:
            ref_desc = {
                'sigma1': 'σ₁', 'sigma_median': 'σ_median',
                'sigma_mean': 'σ_mean', 'sigma_k': 'σ_k',
                'sigma_global': 'E_avg', 'sigma_tail': 'E_tail'
            }.get(self.sigma_ref, self.sigma_ref)
            
            print(f"[ASPIRELayer] Built | γ={self.gamma:.2f} α={self.alpha:.2f} ρ={self.rho:.4f}")
            print(f"             | ref={ref_desc}, λ(τᵞ)={self.alpha_abs:.4f}")

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
        # sigma_ref: normalization reference in gamma_only mode (inherited from ASPIRE config)
        self.sigma_ref = kwargs.get('sigma_ref', 'sigma1')

        self.register_buffer("cheby_coeffs", torch.empty(0))
        self.register_buffer("t_mid",        torch.tensor(0.0))
        self.register_buffer("t_half",       torch.tensor(0.0))
        self.register_buffer("item_weights", torch.empty(0))
        self.alpha_abs = 0.0
        self.rho = 0.0

        self.X_torch_csr = None
        self.Xt_torch_csr = None
        self.sparse_device = None
        self.scores_cache = None
        self.matrix_id = None

    @torch.no_grad()
    def _estimate_lambda_max(self, X_csr, Xt_csr) -> float:
        device = X_csr.device
        calc_device = device if device.type == 'cuda' else torch.device("cpu")
        
        X_local = X_csr.to(calc_device)
        Xt_local = Xt_csr.to(calc_device)
        
        v = torch.randn(X_local.shape[1], 1, device=calc_device)
        v /= (v.norm() + 1e-12)
        
        last_lambda = 0.0
        for _ in range(30):
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

        lam_torch = torch.from_numpy(lam_nodes).float()
        # sigma_ref는 Chebyshev 노드들(라무다 샘플링 포인트) 기준으로 적용—
        # is_gram=True 이므로 lam_torch^(gamma/2) = sigma_k^gamma 스케일에서의 median/mean헤다.
        h_nodes, self.alpha, self.alpha_abs = AspireFilter.apply_filter(
            lam_torch, gamma=self.gamma, alpha=self.alpha,
            mode=self.filter_mode, is_gram=True, sigma_ref=self.sigma_ref,
            custom_lambda=getattr(self, '_custom_lam', None)
        )

        f_nodes = h_nodes.numpy()
        coeffs = np.zeros(K + 1)
        for k in range(K + 1):
            coeffs[k] = (2.0 / (K + 1)) * np.sum(f_nodes * np.cos(k * theta))
        coeffs[0] /= 2.0
        return coeffs

    @torch.no_grad()
    def build(self, X_sparse, dataset_name: str | None = None, device=None, verbose: bool = True):
        if device is not None:
            dev = torch.device(device)
        else:
            dev = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        
        self.sparse_device = torch.device("cpu" if "mps" in dev.type else dev)
        
        X_coo = X_sparse.tocoo()
        indices = torch.from_numpy(np.vstack((X_coo.row, X_coo.col))).long()
        values = torch.from_numpy(X_coo.data).float()
        self.X_torch_csr = torch.sparse_coo_tensor(indices, values, X_coo.shape, device=self.sparse_device).coalesce().to_sparse_csr()
        self.Xt_torch_csr = torch.sparse_coo_tensor(torch.stack([indices[1], indices[0]]), values, (X_coo.shape[1], X_coo.shape[0]), device=self.sparse_device).coalesce().to_sparse_csr()

        self.matrix_id = EVDCacheManager._generate_matrix_id(X_sparse)
        
        self._custom_lam = None
        if self.sigma_ref in ['sigma_global', 'sigma_tail']:
            total_energy = X_sparse.nnz
            max_rank = min(X_sparse.shape)
            # Chebyshev 기반에선 SVD를 구하지 않으므로, tail도 global 평균으로 Fallback
            self._custom_lam = total_energy / max_rank

        lambda_max = _ChebyCache.get_lambda(dataset_name, self.matrix_id)
        
        if lambda_max is None:
            lambda_max = self._estimate_lambda_max(self.X_torch_csr, self.Xt_torch_csr) if self.lambda_max_estimate == "auto" else float(self.lambda_max_estimate)
            _ChebyCache.put_lambda(dataset_name, self.matrix_id, lambda_max)
        elif verbose:
            print(f"[ChebyASPIRE] Lambda Cache Hit | λ_max={lambda_max:.2f}")

        self.t_mid.fill_(lambda_max / 2.0)
        self.t_half.fill_(lambda_max / 2.0)

        coeffs = self._compute_chebyshev_coeffs(lambda_max, self.degree)
        self.register_buffer("cheby_coeffs", torch.from_numpy(coeffs).float().to(dev))

        n = X_sparse.shape[1]
        
        if n <= 50000:
            try:
                if verbose: print(f"[ChebyASPIRE] Building item-wise filter (dense path, items={n})...")
                from src.utils.gpu_accel import _build_gram
                L = _build_gram(X_sparse, dev)
                
                if dev.type == 'cuda':
                    L = L.bfloat16()
                    coeffs_t = self.cheby_coeffs.bfloat16()
                    weights = self._dense_chebyshev(L, coeffs_t, n, dev)
                    self.item_weights = weights.float() 
                else:
                    self.item_weights = self._dense_chebyshev(L, self.cheby_coeffs, n, dev)
                
                del L
                if dev.type == 'cuda': torch.cuda.empty_cache()
                if verbose: print(f"[ChebyASPIRE] Item weights built on {dev.type} (Dense)")
            except Exception as e:
                print(f"[ChebyASPIRE] Dense build failed: {e}, falling back to Sparse.")
                self.item_weights = torch.empty(0, device=dev)
        
        if verbose:
            mode_desc = "Standard" if self.filter_mode != "gamma_only" else "Gamma-only"
            print(f"[ChebyASPIRE] Build done | Degree: {self.degree}, Mode: {mode_desc}, Device: {dev.type}")

    def _dense_chebyshev(self, L, coeffs, n, dev) -> torch.Tensor:
        """
        [Optimized] In-place Normalized Dense Chebyshev Expansion
        루프 진입 전 L을 L_scaled = (L - mid*I) / half 로 덮어씌워 $O(N^2)$ 연산과 메모리 할당을 최소화함.
        T_k = 2 * L_scaled @ T_{k-1} - T_{k-2}
        """
        mid = float(self.t_mid)
        half = float(self.t_half)
        
        # 1. In-place Normalization (VRAM 및 연산량 대폭 절약)
        L.diagonal().sub_(mid)
        L.mul_(1.0 / half)
        # 이제 L 행렬은 정규화된 L_scaled 로 작동함.
        
        T_prev = torch.eye(n, device=dev, dtype=L.dtype)
        T_curr = L.clone()
        
        # W = c0 * T0 + c1 * T1
        W = T_prev.clone().mul_(float(coeffs[0]))
        W.add_(T_curr, alpha=float(coeffs[1]))
        
        for k in range(2, self.degree + 1):
            # T_next = 2 * L_scaled @ T_curr - T_prev
            T_next = torch.mm(L, T_curr)
            T_next.mul_(2.0).sub_(T_prev)
            
            W.add_(T_next, alpha=float(coeffs[k]))
            
            T_prev = T_curr
            T_curr = T_next
            
        return W

    @torch.no_grad()
    def forward(self, X_batch: torch.Tensor) -> torch.Tensor:
        if self.item_weights.numel() > 0:
            return torch.mm(X_batch, self.item_weights)
        return self._spmv_forward(X_batch)

    def _spmv_forward(self, X):
        X_t = X.t().to(self.sparse_device)
        T_prev = X_t
        mid_val = float(self.t_mid)
        half_val = float(self.t_half)
        inv_half = 1.0 / half_val
        coeff_inv_half = 2.0 * inv_half
        
        with torch.no_grad():
            T_curr = torch.sparse.mm(self.Xt_torch_csr, torch.sparse.mm(self.X_torch_csr, T_prev))
            T_curr.sub_(T_prev, alpha=mid_val).mul_(inv_half)
            
            W = T_prev.clone().mul_(float(self.cheby_coeffs[0]))
            W.add_(T_curr, alpha=float(self.cheby_coeffs[1]))
            
            for k in range(2, self.degree + 1):
                temp = torch.sparse.mm(self.Xt_torch_csr, torch.sparse.mm(self.X_torch_csr, T_curr))
                T_next = temp.sub_(T_curr, alpha=mid_val).mul_(coeff_inv_half).sub_(T_prev)
                
                W.add_(T_next, alpha=float(self.cheby_coeffs[k]))
                T_prev, T_curr = T_curr, T_next
        return W.t().to(X.device)

    @torch.no_grad()
    def precompute(self, X_all_sparse, dataset_name=None, matrix_id=None, device=None):
        t0 = time.time()
        N_users, N_items = X_all_sparse.shape
        
        if self.item_weights.numel() == 0 and N_items <= 50000:
             print(f"[ChebyASPIRE] Item weights missing, attempted fallback build...")
             I_n = torch.eye(N_items, device=device or self.sparse_device)
             self.item_weights = self._spmv_forward(I_n).to(device or self.sparse_device)
             del I_n

        if N_users * N_items > 1.5e8: 
             print(f"[ChebyASPIRE] scores_cache risk ({N_users}x{N_items}), skipping cache, using on-the-fly weights.")
             self.scores_cache = None
             return

        print(f"[ChebyASPIRE] Precomputing scores_cache (Items: {N_items}, Users: {N_users})...")
        batch_size = 5000
        all_scores = torch.zeros((N_users, N_items), device="cpu")
        
        if self.item_weights.numel() > 0:
            weights = self.item_weights.to(device or self.sparse_device)
            for i in range(0, N_users, batch_size):
                end = min(i + batch_size, N_users)
                X_batch = torch.from_numpy(X_all_sparse[i:end].toarray()).float().to(weights.device)
                all_scores[i:end] = torch.mm(X_batch, weights).cpu()
                del X_batch
        else:
            for i in range(0, N_users, batch_size):
                end = min(i + batch_size, N_users)
                X_batch = torch.from_numpy(X_all_sparse[i:end].toarray()).float().to(self.sparse_device)
                all_scores[i:end] = self._spmv_forward(X_batch).cpu()
                del X_batch
        
        self.scores_cache = all_scores
        print(f"[ChebyASPIRE] Precomputation done: {time.time()-t0:.2f}s")

    @torch.no_grad()
    def visualize_matrices(self, X_sparse=None, save_dir=None, lightweight=False):
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
