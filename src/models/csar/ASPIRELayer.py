import torch
import torch.nn as nn
import numpy as np
import os
import json
import time
import matplotlib.pyplot as plt
from src.utils.gpu_accel import SVDCacheManager
from src.models.csar.lira_visualizer import LIRAVisualizer
from src.utils.cache_manager import GlobalCacheManager

class MNARGammaCacheManager(GlobalCacheManager):
    """
    Persistent cache for estimated MNAR gamma values, keyed by dataset_name.
    Stored in 'data_cache/mnar_gamma_{dataset_name}.json'.
    Global scope: shared across models for the same dataset.
    """
    _mem_cache = {}
    _cache_dir = 'data_cache'

    @classmethod
    def _get_path(cls, dataset_name):
        if not dataset_name: return None
        os.makedirs(cls._cache_dir, exist_ok=True)
        return os.path.join(cls._cache_dir, f"mnar_gamma_{dataset_name}.json")

    @classmethod
    def get(cls, dataset_name):
        if not dataset_name: return None
        # 1. Memory check
        if dataset_name in cls._mem_cache:
            return cls._mem_cache[dataset_name]
        
        # 2. Disk check
        path = cls._get_path(dataset_name)
        if path and os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    val = data.get('gamma')
                    cls._mem_cache[dataset_name] = val
                    return val
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
                with open(path, 'w') as f:
                    json.dump({'gamma': val, 'timestamp': time.time()}, f)
            except Exception as e:
                print(f"[MNARGammaCache] Failed to save cache: {e}")

    # --- CacheManager Interface ---
    def summary(self):
        import glob as _glob
        files = _glob.glob(os.path.join(self._cache_dir, "mnar_gamma_*.json"))
        return {"type": "MNAR_Gamma", "cached_datasets": list(self._mem_cache.keys()), "files": len(files)}

    def invalidate(self, key=None):
        if key:
            self._mem_cache.pop(key, None)
            path = self._get_path(key)
            if path and os.path.exists(path):
                os.remove(path)
        else:
            self._mem_cache.clear()
            import glob as _glob
            for f in _glob.glob(os.path.join(self._cache_dir, "mnar_gamma_*.json")):
                os.remove(f)

# Backward compatibility alias
_MNARGammaCache = MNARGammaCacheManager


class GramMatrixCacheManager(GlobalCacheManager):
    """
    Persistent cache for Gram Matrix (X^T X), keyed by dataset_name.
    Stored in 'data_cache/gram_{dataset_name}.pt'.
    """
    _mem_cache = {}
    _cache_dir = 'data_cache'

    @classmethod
    def _get_path(cls, dataset_name):
        if not dataset_name: return None
        os.makedirs(cls._cache_dir, exist_ok=True)
        return os.path.join(cls._cache_dir, f"gram_{dataset_name}.pt")

    @classmethod
    def get(cls, dataset_name, device='cpu'):
        if not dataset_name: return None
        if dataset_name in cls._mem_cache:
            return cls._mem_cache[dataset_name].to(device)
        
        path = cls._get_path(dataset_name)
        if path and os.path.exists(path):
            try:
                val = torch.load(path, map_location=device, weights_only=True)
                cls._mem_cache[dataset_name] = val.cpu() # Store in CPU mem
                return val
            except Exception:
                return None
        return None

    @classmethod
    def put(cls, dataset_name, val):
        if not dataset_name: return
        cls._mem_cache[dataset_name] = val.cpu()
        path = cls._get_path(dataset_name)
        if path:
            try:
                torch.save(val.cpu(), path)
            except Exception as e:
                print(f"[GramCache] Failed to save: {e}")

_GramCache = GramMatrixCacheManager


def estimate_alignment_slope(X_sparse=None, singular_values=None, item_popularity=None, dataset_name=None, alpha=None):
    """
    SWLS (Sensitivity-Weighted Least Squares) 기반 alignment slope 추정.
    
    log(n_i) = a * log(λ_i) + C 를 통해 슬로프 a 도출.
    가중치: w_i = h_i(1 - h_i), h_i = λ_i / (λ_i + α)  [Wiener 필터 민감도]
    
    [통계적 근거]
    - h(1-h)는 Wiener 필터의 β에 대한 Fisher Information에 비례
    - 필터가 "불확실한" 전이대(σ² ≈ α)에서 slope를 추정
    - 극단(σ 극대/극소)은 이미 결정(pass/reject)되어 추정에 기여하지 않음
    
    alpha 미지정 시 OLS fallback.
    """
    # alpha-dependent SWLS는 캐시하지 않음 (alpha가 trial마다 변경됨)
    if dataset_name and alpha is None:
        cached_a = _MNARGammaCache.get(f"{dataset_name}_slope")
        if cached_a is not None: 
            return cached_a
    
    # 1. 고유값 기반 추정 (ASPIRE / SVD 모델) - 가장 정확함
    if (X_sparse is not None or item_popularity is not None) and singular_values is not None:
        if item_popularity is not None:
            n_i = np.sort(item_popularity)[::-1]
        else:
            item_pops = np.array(X_sparse.sum(axis=0)).flatten()
            n_i = np.sort(item_pops)[::-1]
        
        k = len(singular_values)
        n_i_trunc = n_i[:k]
        lam_i = singular_values.pow(2).cpu().numpy()
        
        valid_mask = (n_i_trunc > 0) & (lam_i > 1e-10)
        n_i_valid = n_i_trunc[valid_mask]
        lam_i_valid = lam_i[valid_mask]
        
        if len(n_i_valid) >= 10:
            x = np.log(lam_i_valid)
            y = np.log(n_i_valid)
            
            if alpha is not None and alpha > 0:
                # SWLS: 필터 민감도 가중 회귀
                # h = λ/(λ+α), w = h(1-h) = λα/(λ+α)²
                h = lam_i_valid / (lam_i_valid + alpha)
                w = h * (1.0 - h)
                w = np.maximum(w, 1e-12)
                w /= w.sum()
                
                # Weighted OLS (closed-form)
                mean_x = np.sum(w * x)
                mean_y = np.sum(w * y)
                numer = np.sum(w * (x - mean_x) * (y - mean_y))
                denom = np.sum(w * (x - mean_x)**2)
                slope = float(numer / (denom + 1e-9))
            else:
                # Standard OLS fallback
                slope, _ = np.polyfit(x, y, 1)
                slope = float(slope)
            
            slope = max(0.5, slope) # a >= 0.5 (MCAR)
            
            if dataset_name and alpha is None:
                _MNARGammaCache.put(f"{dataset_name}_slope", slope)
            return slope

    # 2. 인기도 순위 기반 추정 (fallback, SVD 없이)
    if X_sparse is not None:
        item_pops = np.array(X_sparse.sum(axis=0)).flatten()
        item_pops = np.sort(item_pops)[::-1]
        item_pops = item_pops[item_pops > 0]
        if len(item_pops) < 10: return 0.5
        
        y = np.log(item_pops)
        x = np.log(np.arange(1, len(item_pops) + 1))
        z_slope, _ = np.polyfit(x, y, 1)
        # γ = -z_slope
        # a = (2γ + 1) / (2γ + 2)
        gamma = max(1e-5, float(-z_slope))
        slope = (2.0 * gamma + 1.0) / (2.0 * gamma + 2.0)
        
        if dataset_name:
            _MNARGammaCache.put(f"{dataset_name}_slope", slope)
        return slope

    return 0.5 # MCAR fallback

# Backward compatibility wrapper
def estimate_mnar_gamma(X_sparse=None, singular_values=None, dataset_name=None):
    a = estimate_alignment_slope(X_sparse, singular_values, dataset_name)
    # γ = (2a - 1) / (2(1 - a))
    gamma = (2.0 * a - 1.0) / (2.0 * (1.0 - a) + 1e-9)
    return max(0.0, gamma)

class ASPIRELayer(nn.Module):
    def __init__(self, k=200, alpha=500.0, beta=1.0, target_energy=0.99):
        super(ASPIRELayer, self).__init__()
        self.k = int(k[0] if isinstance(k, (list, np.ndarray)) else k)
        self.alpha = float(alpha[0] if isinstance(alpha, (list, np.ndarray)) else alpha)
        self.beta_config = beta[0] if isinstance(beta, (list, np.ndarray)) else beta
        self.beta = 1.0 # Placeholder, will be set in build()
        self.target_energy = float(target_energy[0] if isinstance(target_energy, (list, np.ndarray)) else target_energy)
        self.register_buffer('singular_values', torch.empty(0)) 
        self.register_buffer('filter_diag', torch.empty(0))
        self.register_buffer('V_raw', torch.empty(0, 0))
        self.gamma = 0.0 # Store estimated gamma
        self.alignment_slope = 0.5 # Store observed slope 'a'
        
    @property
    def V_k(self): return self.V_raw

    @torch.no_grad()
    def build(self, X_sparse, dataset_name=None):
        dev = self.singular_values.device
        manager = SVDCacheManager(device=dev)
        u, s, v, total_energy = manager.get_svd(X_sparse, k=None, target_energy=self.target_energy, dataset_name=dataset_name)
        self.k = len(s)
        self.register_buffer('singular_values', s.to(dev))
        self.register_buffer('V_raw', v.to(dev))
        
        # Auto Beta: β = max(0, 2a - 1) — MCAR 기준선(a=0.5)으로부터의 이탈량
        if isinstance(self.beta_config, str):
            a = estimate_alignment_slope(X_sparse=X_sparse, singular_values=self.singular_values, dataset_name=dataset_name, alpha=self.alpha)
            self.alignment_slope = a
            self.beta = max(0.0, 2.0 * a - 1.0)
            self.gamma = (2.0 * a - 1.0) / (2.0 * (1.0 - a) + 1e-9)  # logging only
            print(f"[{self.__class__.__name__}] SWLS a={a:.3f} -> β={self.beta:.3f} (γ={self.gamma:.3f})")
        else:
            self.beta = float(self.beta_config)
            self.gamma = 0.0
            self.alignment_slope = 0.5

        # As derived, penalty is α * |Γ W|_F^2 where Γ_kk = σ_k^β
        # This leads to h_k = σ_k^2 / (σ_k^2 + α * σ_k^{2β}) = σ_k^{2(1-β)} / (σ_k^{2(1-β)} + α)
        s_pow = torch.pow(self.singular_values, 2.0 * (1.0 - self.beta))
        self.register_buffer('filter_diag', s_pow / (s_pow + self.alpha))
        print(f"[{self.__class__.__name__}] ASPIRE build complete (k={self.k}). Device: {dev}")

    @torch.no_grad()
    def forward(self, X_batch, user_ids=None):
        if self.singular_values.numel() == 0: raise RuntimeError("build() first")
        XV = torch.mm(X_batch, self.V_raw)
        XV_filtered = XV * self.filter_diag
        return torch.mm(XV_filtered, self.V_raw.t())

    @torch.no_grad()
    def visualize_matrices(self, X_sparse=None, save_dir=None, lightweight=False):
        LIRAVisualizer.visualize_spectral_tikhonov(
            self.singular_values, 
            self.filter_diag, 
            self.alpha, 
            self.beta, 
            gamma=self.gamma,
            a=self.alignment_slope, # Pass alignment slope
            X_sparse=X_sparse, 
            save_dir=save_dir, 
            file_prefix='aspire'
        )

class ChebyASPIRELayer(nn.Module):
    def __init__(self, alpha=500.0, degree=20, beta=0.5, lambda_max_estimate='auto', threshold=1e-4):
        super().__init__()
        self.alpha = float(alpha)
        self.degree = int(degree)
        self.beta_config = beta
        self.beta = 0.5 # Placeholder
        self.lambda_max_estimate = lambda_max_estimate
        self.threshold = float(threshold)
        
        self.register_buffer('cheby_coeffs', torch.empty(0))
        self.register_buffer('t_mid', torch.tensor(0.0))
        self.register_buffer('t_half', torch.tensor(0.0))
        self.register_buffer('item_weights', torch.empty(0)) # Precomputed W matrix
        self.gamma = 0.0 # Store estimated gamma
        self.alignment_slope = 0.5 # Store observed slope 'a'
        
        # [OPTIMIZATION] 희소 행렬 자체를 클래스 변수로 캐싱 (상태 저장 불필요)
        self.X_torch_csr = None
        self.Xt_torch_csr = None
        self.sparse_device = None

    def _aspire_filter(self, lam):
        # 수학적 증명: lambda는 X^TX의 고유값(σ^2).
        # h(σ) = σ^{2(1-β)} / (σ^{2(1-β)} + α) 이므로,
        # λ를 기준으로 하면 exponent는 1 - β 가 됩니다!
        exponent = 1.0 - self.beta 
        lam_pow = np.power(np.maximum(lam, 0.0), exponent)
        return lam_pow / (lam_pow + self.alpha)

    @torch.no_grad()
    def _estimate_lambda_max(self, X_csr, Xt_csr):
        v = torch.randn(X_csr.shape[1], 1, device=X_csr.device)
        v = v / v.norm()
        for _ in range(30):
            # CSR 포맷 사용으로 속도 대폭 향상
            v = torch.sparse.mm(Xt_csr, torch.sparse.mm(X_csr, v))
            lambda_est = v.norm().item()
            v = v / lambda_est
        return lambda_est

    def _compute_chebyshev_coeffs(self, lambda_min, lambda_max, K):
        j = np.arange(K + 1)
        theta = np.pi * (j + 0.5) / (K + 1)
        t_nodes = np.cos(theta)
        
        mid, half = (lambda_max + lambda_min) / 2.0, (lambda_max - lambda_min) / 2.0
        lambda_nodes = mid + half * t_nodes
        f_nodes = self._aspire_filter(lambda_nodes)
        
        coeffs = np.zeros(K + 1)
        for k in range(K + 1):
            T_k = np.cos(k * theta) 
            coeffs[k] = (2.0 / (K + 1)) * np.sum(f_nodes * T_k)
        coeffs[0] /= 2.0
        return coeffs

    @torch.no_grad()
    def build(self, X_sparse, dataset_name=None):
        from src.utils.gpu_accel import get_device
        device = get_device('auto')

        # MPS는 torch.sparse.mm 미지원 — 희소 연산은 CPU에서 수행
        self.sparse_device = torch.device('cpu') if device.type == 'mps' else device
        
        # COO에서 바로 PyTorch CSR로 변환 (메모리 연속성 확보 및 연산 속도 극대화)
        X_coo = X_sparse.tocoo()
        indices = torch.from_numpy(np.vstack((X_coo.row, X_coo.col))).long()
        values = torch.from_numpy(X_coo.data).float()
        
        # 희소 행렬을 build 타임에 한 번만 생성하여 메모리에 상주
        X_coo_torch = torch.sparse_coo_tensor(indices, values, X_coo.shape, device=self.sparse_device).coalesce()
        Xt_coo_torch = torch.sparse_coo_tensor(
            torch.stack([indices[1], indices[0]]), values, (X_coo.shape[1], X_coo.shape[0]), device=self.sparse_device
        ).coalesce()
        
        # PyTorch 1.10+ 지원 CSR 포맷으로 변환 (sparse.mm 속도 향상)
        self.X_torch_csr = X_coo_torch.to_sparse_csr()
        self.Xt_torch_csr = Xt_coo_torch.to_sparse_csr()

        if self.lambda_max_estimate == 'auto':
            lambda_max = self._estimate_lambda_max(self.X_torch_csr, self.Xt_torch_csr)
        else:
            lambda_max = float(self.lambda_max_estimate)
            
        lambda_min = 0.0
        
        # Auto Beta: β = max(0, 2a - 1) — MCAR 기준선(a=0.5)으로부터의 이탈량
        if isinstance(self.beta_config, str):
            a = estimate_alignment_slope(X_sparse=X_sparse, dataset_name=dataset_name, alpha=self.alpha)
            self.alignment_slope = a
            self.beta = max(0.0, 2.0 * a - 1.0)
            self.gamma = (2.0 * a - 1.0) / (2.0 * (1.0 - a) + 1e-9)  # logging only
            print(f"[{self.__class__.__name__}] SWLS a={a:.3f} -> β={self.beta:.3f} (γ={self.gamma:.3f})")
        else:
            self.beta = float(self.beta_config)
            self.gamma = 0.0
            self.alignment_slope = 0.5

        coeffs = self._compute_chebyshev_coeffs(lambda_min, lambda_max, self.degree)
        self.register_buffer('cheby_coeffs', torch.from_numpy(coeffs).float().to(device))
        self.register_buffer('t_mid', torch.tensor((lambda_max + lambda_min) / 2.0, device=device))
        self.register_buffer('t_half', torch.tensor((lambda_max - lambda_min) / 2.0, device=device))
        
        # [SUPER OPTIMIZATION] 
        # 1. Compute/Get Gram Matrix (L = X^T X) - Dataset-dependent, Params-independent
        num_items = X_sparse.shape[1]
        L = _GramCache.get(dataset_name, device=device)
        
        if L is None:
            if num_items <= 15000:
                print(f"[{self.__class__.__name__}] Computing Gram Matrix (X^T X) for the first time...")
                # Compute using Sparse MM once
                X_csr = self.X_torch_csr.to(self.sparse_device)
                Xt_csr = self.Xt_torch_csr.to(self.sparse_device)
                
                # To Dense for fast future recurrence
                L = torch.sparse.mm(Xt_csr, X_csr.to_dense())
                if dataset_name:
                    _GramCache.put(dataset_name, L)
                L = L.to(device)
            else:
                L = None # Too large

        # 2. Compute Filter Weight Matrix W using Dense Recurrence
        # This part depends on alpha/beta, so it runs every trial, but it's 100% Dense GPU/MPS!
        if L is not None:
            print(f"[{self.__class__.__name__}] Precomputing W using Dense Recurrence on {device}...")
            # Recurrence: T_next = 2 * (L @ T_curr - t_mid * T_curr) / t_half - T_prev
            # Initial states
            T_prev = torch.eye(num_items, device=device)
            T_curr = (L - self.t_mid * T_prev) / self.t_half
            
            W = float(coeffs[0]) * T_prev + float(coeffs[1]) * T_curr
            
            for k in range(2, self.degree + 1):
                T_next = 2.0 * (torch.mm(L, T_curr) - self.t_mid * T_curr) / self.t_half - T_prev
                W.add_(T_next, alpha=float(coeffs[k]))
                T_prev = T_curr
                T_curr = T_next
            
            self.item_weights = W
            print(f"[{self.__class__.__name__}] Weight Matrix W ready on {device}. Inference will be lightning fast.")
        else:
            self.item_weights = torch.empty(0)
            print(f"[{self.__class__.__name__}] Item count {num_items} too large for Dense optimization.")
        
        print(f"[{self.__class__.__name__}] Build complete. coefficients computed.")

    @torch.no_grad()
    def forward(self, X_batch, user_ids=None):
        if getattr(self, 'X_torch_csr', None) is None:
            raise RuntimeError("build() first")

        batch_device = X_batch.device

        # [ULTRA OPTIMIZATION] If precomputed matrix exists, use DENSE-DENSE MM
        # Efficient and runs natively on any device (GPU/MPS/CPU)
        if self.item_weights.numel() > 0:
            return torch.mm(X_batch, self.item_weights).to(batch_device)

        sparse_dev = self.sparse_device
        
        # [OPTIMIZATION] 필요한 파라미터 미리 추출 및 Device 이동 최소화
        X_t = X_batch.t().to(sparse_dev) # (Items, Batch)
        coeffs = self.cheby_coeffs.cpu().numpy() # Scalar 연산은 CPU numpy가 안정적
        t_mid_val = self.t_mid.item()
        t_half_val = self.t_half.item()

        # --- Chebyshev Recurrence (Memory Optimized) ---
        # T_prev = T_0(S) * X_t = X_t
        T_prev = X_t
        
        # T_curr = T_1(S) * X_t = S(X_t)
        # S(v) = (Xt @ (X @ v) - t_mid * v) / t_half
        inner = torch.sparse.mm(self.X_torch_csr, T_prev)
        T_curr = torch.sparse.mm(self.Xt_torch_csr, inner)
        T_curr.add_(T_prev, alpha=-t_mid_val)
        T_curr.div_(t_half_val)
        
        # out = c_0 * T_0 + c_1 * T_1
        out = T_prev.clone()
        out.mul_(float(coeffs[0]))
        out.add_(T_curr, alpha=float(coeffs[1]))

        # T_next = 2 * S(T_curr) - T_prev
        for k in range(2, self.degree + 1):
            # 1. S(T_curr) 계산
            inner = torch.sparse.mm(self.X_torch_csr, T_curr)
            T_next = torch.sparse.mm(self.Xt_torch_csr, inner)
            T_next.add_(T_curr, alpha=-t_mid_val)
            T_next.div_(t_half_val)
            
            # 2. T_k 재귀 공식 적용: T_k = 2 * S * T_{k-1} - T_{k-2}
            T_next.mul_(2.0)
            T_next.sub_(T_prev)
            
            # 3. 누적합 업데이트
            out.add_(T_next, alpha=float(coeffs[k]))
            
            # 4. 핑퐁 업데이트
            T_prev = T_curr
            T_curr = T_next

        # 최종 연산 결과물만 원래 디바이스(batch_device)로 복귀
        return out.t().to(batch_device)

    @torch.no_grad()
    def visualize_matrices(self, X_sparse=None, save_dir=None, lightweight=False):
        # ChebyASPIRE visualization: Plot coefficients and the filter shape
        if not save_dir: return
        os.makedirs(save_dir, exist_ok=True)
        
        coeffs = self.cheby_coeffs.cpu().numpy()
        plt.figure(figsize=(18, 5))
        
        # 1. Coefficients Plot
        plt.subplot(1, 3, 1)
        plt.bar(range(len(coeffs)), coeffs)
        plt.title(f"Chebyshev Coefficients (degree={self.degree})")
        plt.xlabel("k")
        plt.ylabel("c_k")
        
        # --- 시각화 데이터 계산 로직 복구 ---
        # 1. 시그마 범위 설정 (ASPIRE와 방향 일치: Dominant -> Noise)
        lambda_max = (self.t_mid + self.t_half).detach().cpu().item()
        max_sigma = np.sqrt(lambda_max)
        sigmas = np.linspace(0, max_sigma, 200)
        sigmas_plot = sigmas[::-1] # Large to Small (Dominant -> Noise)
        lams = sigmas_plot**2
        
        # 2. 이론적 타겟 (Standard ASPIRE)
        f_target = self._aspire_filter(lams)
        
        # 3. 실제 Chebyshev 근사값 계산 (recurrence)
        t_scaled = (lams - self.t_mid.item()) / self.t_half.item()
        f_approx = np.zeros_like(lams)
        T_prev = np.ones_like(t_scaled)
        T_curr = t_scaled
        
        f_approx = coeffs[0] * T_prev + coeffs[1] * T_curr
        for k in range(2, self.degree + 1):
            T_next = 2 * t_scaled * T_curr - T_prev
            f_approx += coeffs[k] * T_next
            T_prev = T_curr
            T_curr = T_next
        # -------------------------------
        
        # --- 지표 계산 및 JSON 저장 ---
        fit_error = f_approx - f_target
        metrics = {
            "model": "ChebyASPIRE",
            "params": {
                "alpha": self.alpha,
                "alignment_slope": self.alignment_slope,
                "gamma": self.gamma, 
                "beta": self.beta,
                "degree": self.degree,
                "lambda_max": lambda_max
            },
            "coefficients": {
                "mean": float(coeffs.mean()),
                "std": float(coeffs.std()),
                "max": float(coeffs.max()),
                "abs_sum": float(np.abs(coeffs).sum())
            },
            "fit_quality": {
                "mean_error": float(np.mean(fit_error)),
                "mae": float(np.mean(np.abs(fit_error))),
                "rmse": float(np.sqrt(np.mean(fit_error**2))),
                "max_error": float(np.max(np.abs(fit_error)))
            }
        }
        with open(os.path.join(save_dir, "cheby_aspire_metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=4)
        # ----------------------------
        
        # 2. Filter Plot (Log Scale) 
        plt.subplot(1, 3, 2)
        plt.plot(sigmas_plot, f_target + 1e-12, 'k--', alpha=0.3, label='Target (ASPIRE)')
        plt.plot(sigmas_plot, f_approx + 1e-12, color='orange', linewidth=2, label='Cheby Fit')
        
        plt.yscale('log')
        plt.gca().invert_xaxis() # Large Sigma on Left (Consistent with ASPIRE)
        plt.title(fr"Filter Magnitude ($a={self.alignment_slope:.3f}, \gamma={self.gamma:.2f}, \beta={self.beta:.3f}$)")
        plt.xlabel(r"Singular Value $\sigma$ (Head $\rightarrow$ Tail)")
        plt.ylabel(r"Filter Value")
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()

        # 3. Fit Error (Theoretical - Approx)
        plt.subplot(1, 3, 3)
        plt.plot(sigmas_plot, f_approx - f_target, color='red', label='Fit Error (Fit - Target)')
        plt.gca().invert_xaxis() # Large Sigma on Left
        plt.title(r"Fit Approximation Error")
        plt.xlabel(r"Singular Value $\sigma$")
        plt.ylabel(r"$\Delta h(\sigma)$")
        plt.axhline(0, color='black', linewidth=1, linestyle='--')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "cheby_aspire_analysis.png"))
        plt.close()
        print(f"[{self.__class__.__name__}] Analysis saved to {save_dir}")
