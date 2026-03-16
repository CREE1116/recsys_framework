from __future__ import annotations

import os
import json
import time
import glob as _glob

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor

from src.utils.gpu_accel import SVDCacheManager
from src.models.csar.lira_visualizer import LIRAVisualizer
from src.utils.cache_manager import GlobalCacheManager
from src.models.csar import beta_estimators


# ══════════════════════════════════════════════════════════════════════════════
# 캐시 매니저
# ══════════════════════════════════════════════════════════════════════════════

class MNARGammaCacheManager(GlobalCacheManager):
    """β / a 영속 캐시. dataset_name 단위."""
    _mem_cache: dict = {}
    _cache_dir: str  = "data_cache"

    @classmethod
    def _get_path(cls, dataset_name):
        if not dataset_name: return None
        os.makedirs(cls._cache_dir, exist_ok=True)
        return os.path.join(cls._cache_dir, f"mnar_gamma_{dataset_name}.json")

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
                payload = val if isinstance(val, dict) else {"value": val}
                payload["timestamp"] = time.time()
                with open(path, "w", encoding='utf-8') as f:
                    json.dump(payload, f)
            except Exception as e:
                print(f"[MNARGammaCache] save failed: {e}")

    def summary(self):
        files = _glob.glob(os.path.join(self._cache_dir, "mnar_gamma_*.json"))
        return {"type": "MNAR_Gamma", "entries": len(self._mem_cache), "files": len(files)}

    def invalidate(self, key=None):
        if key:
            self._mem_cache.pop(key, None)
            path = self._get_path(key)
            if path and os.path.exists(path): os.remove(path)
        else:
            self._mem_cache.clear()
            for f in _glob.glob(os.path.join(self._cache_dir, "mnar_gamma_*.json")):
                os.remove(f)

_MNARGammaCache = MNARGammaCacheManager


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


# ══════════════════════════════════════════════════════════════════════════════
# AspireEngine
# ══════════════════════════════════════════════════════════════════════════════

class AspireEngine:
    """
    ASPIRE 파이프라인의 정적 유틸리티.

    SPP → β → h(σ)

    compute_spp : 방향 k의 인기도 오염 강도 p̃_k 측정
    estimate_beta: p̃_k의 멱법칙 피팅으로 β 추출 (스무딩)
    apply_filter : h(σ) = σ^{2-2β} / (σ^{2-2β} + α)

    두 번째 경로 (V 없을 때 — ChebyASPIRE):
    estimate_beta_from_slope: slope 비율로 β 추정
    """

    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        if torch.is_tensor(x):
            return x.detach().cpu().numpy().astype(float)
        return np.asarray(x, dtype=float)

    # ── 1. SPP ────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_spp(
        V,
        item_frequencies,
    ) -> np.ndarray:
        """
        Spectral Propensity Projection.

          p_i  = n_i / n_max          # 인기도 정규화 (damping 제거)
          p̃_k = Σ_i V_{ki}² · p_i   # 방향 k의 오염 강도

        p̃_k는 인기도 편향 + 신호 + 노이즈를 전부 흡수한 raw 측정값.
        이를 그대로 쓰지 않고 멱법칙으로 스무딩하여 β를 추출한다.

        Parameters
        ----------
        V               : (n, k) 우측 특이벡터
        item_frequencies: (n,) 아이템별 상호작용 수

        Returns
        -------
        p_tilde : (k,) 방향별 오염 강도
        """
        V_np    = AspireEngine._to_numpy(V)
        n_i     = AspireEngine._to_numpy(item_frequencies).flatten()
        
        # [정규화 복원] n_i 최대값을 1로 맞추어 이론적 Propensity로 변환
        n_max = n_i.max() + 1e-12
        p_i   = n_i / n_max
        
        p_tilde = (V_np ** 2).T @ p_i
        return p_tilde

    # ── 2. β 추정 (SPP 기반) ──────────────────────────────────────────────────

    @staticmethod
    def estimate_beta(
        singular_values: torch.Tensor,
        p_tilde: np.ndarray,
        trim: float = 0.0,
        verbose: bool = True,
        dataset_name: str = "",
        return_line: bool = False,
        estimator_type: str = "slope_ratio",
        weight_mode: str = "normal",
        item_freq: np.ndarray = None,
    ) -> tuple:
        """
        p̃_k의 멱법칙 피팅으로 β 추출.

          log p̃_k = 2β · log σ_k + C
          → β = slope / 2
        """
        s  = np.sort(np.abs(AspireEngine._to_numpy(singular_values)))[::-1]
        pt = np.asarray(p_tilde, dtype=float)

        k   = len(s)
        lo  = int(k * trim)
        hi  = max(lo + 4, int(k * (1 - trim)))
        s_  = s[lo:hi]
        pt_ = pt[lo:hi]

        # Use specialized estimators if requested
        if estimator_type == "ols":
            beta, r2 = beta_estimators.beta_ols(s_, pt_)
        elif estimator_type == "lad":
            beta, r2 = beta_estimators.beta_lad(s_, pt_)
        elif estimator_type == "spp_proj_shifted":
            beta, r2 = beta_estimators.beta_spp_projection_shifted(s_, pt_)
        elif estimator_type == "covariance":
            beta, r2 = beta_estimators.beta_covariance(s_, pt_)
        elif estimator_type == "pairwise":
            beta, r2 = beta_estimators.beta_pairwise_ratio(s_, pt_)
        elif estimator_type == "slope_ratio":
            if item_freq is not None:
                beta, r2 = beta_estimators.beta_slope_ratio(singular_values, item_freq)
            else:
                # Fallback to LAD if no item_freq provided
                beta, r2 = beta_estimators.beta_lad(s_, pt_)
        elif estimator_type == "fixed_0.5":
            beta, r2 = 0.5, 1.0
        else: # Default: Slope-Ratio
            if item_freq is not None:
                beta, r2 = beta_estimators.beta_slope_ratio(singular_values, item_freq)
            else:
                beta, r2 = beta_estimators.beta_lad(s_, pt_)

        if verbose:
            print(f"[ASPIRE] {dataset_name} ({estimator_type}): β={beta:.4f}  R²={r2:.4f}")

        if return_line:
            y_pred_full = hub.predict(np.log(s + 1e-9).reshape(-1, 1))
            return beta, r2, y_pred_full
            
        return beta, r2

    # ── 3. β 추정 (slope 비율 — V 없을 때 fallback) ───────────────────────────

    @staticmethod
    def estimate_beta_from_slope(
        singular_values,
        item_frequencies,
        X_sparse=None,
        svd_k: int = 200,
        verbose: bool = True,
        dataset_name: str = "",
        save_dir: str = None,
        estimator_type: str = "huber",
    ) -> tuple[float, float]:
        """
        slope 비율로 β 추정 — SVD-free ChebyASPIRE 전용.

          a = slope_n / slope_σ
          β = a / 2             (p̃_k ~ σ_k^a → p̃_k ~ σ_k^{2β} → β=a/2)

        V가 없는 초거대 행렬에서 사용. SPP보다 정보량이 적음.
        slope_n 계산 시 n_i / n_max 의 분포를 사용함.

        Returns
        -------
        (beta, a)
        """
        tag = dataset_name or "?"

        if singular_values is not None:
            s = np.sort(np.abs(AspireEngine._to_numpy(singular_values)))[::-1]
        elif X_sparse is not None:
            from scipy.sparse.linalg import svds as _svds
            k = min(svd_k, min(X_sparse.shape) - 1)
            _, _s, _ = _svds(X_sparse.astype(float), k=k)
            s = np.sort(np.abs(_s))[::-1]
        else:
            raise ValueError("singular_values 또는 X_sparse 필요")

        if item_frequencies is not None:
            n_raw = np.abs(AspireEngine._to_numpy(item_frequencies)).flatten()
        elif X_sparse is not None:
            n_raw = np.asarray(X_sparse.sum(axis=0)).flatten()
        else:
            raise ValueError("item_frequencies 또는 X_sparse 필요")
            
        # [NEW] 인기도 분포
        # slope_n을 구할 때 n_i의 정규화 분포를 기반으로 함
        p_i = n_raw / (n_raw.max() + 1e-9)
        p_all_sorted = np.sort(p_i)[::-1]

        # [NEW] Use standardized beta_slope_ratio
        beta, _ = beta_estimators.beta_slope_ratio(s, p_all_sorted)
        a = beta * 2.0

        if verbose:
            print(
                f"[ASPIRE-slope] {tag}: "
                f"a={a:.3f}  β=a/2={beta:.4f}"
            )

        if save_dir:
            try:
                AspireEngine._save_slope_plot(
                    s, p_all_sorted, sl_s, sl_n, a, beta, save_dir, tag
                )
            except Exception:
                pass

        return beta, a

    # ── 4. 필터 ───────────────────────────────────────────────────────────────

    @staticmethod
    def apply_filter(s, alpha: float, beta: float):
        """
        ASPIRE Spectral Scaling Filter.
        h(sigma) = sigma^{2/(1+beta)} / (sigma^{2/(1+beta)} + alpha)
        """
        exponent = float(2.0 / (1.0 + beta))
        if torch.is_tensor(s):
            # Explicitly use float32 for MPS compatibility
            s_f = s.float()
            alpha_f = float(alpha)
            sp = torch.pow(torch.clamp(s_f, min=1e-9), exponent)
            return (sp / (sp + alpha_f)).float()
        else:
            s_np = AspireEngine._to_numpy(s).astype(np.float32)
            sp = np.power(np.clip(s_np, 1e-9, None), exponent)
            return (sp / (sp + alpha)).astype(np.float32)

    @staticmethod
    def apply_direct_spp_filter(s, p_tilde, alpha: float, spp_pow: float):
        """
        [NEW] Direct SPP Filtering.
        """
        if torch.is_tensor(s):
            s_f = s.float()
            s2 = torch.pow(torch.clamp(s_f, min=1e-9), 2)
            pt = torch.from_numpy(AspireEngine._to_numpy(p_tilde)).float().to(s.device)
            pt_safe = torch.clamp(pt, min=1e-12)
            
            # Equalizer 부스트 항: 테일 에너지를 끌어올림
            boost = torch.pow(pt_safe, -float(spp_pow) / 2.0)
            # Wiener 필터 항: 노이즈 억제
            reg = float(alpha) * torch.pow(pt_safe, float(spp_pow))
            wiener = s2 / (s2 + reg)
            
            return (boost * wiener).float()
        else:
            s_np = AspireEngine._to_numpy(s).astype(np.float32)
            s2 = np.power(np.clip(s_np, 1e-9, None), 2)
            pt = np.clip(AspireEngine._to_numpy(p_tilde), 1e-12, None).astype(np.float32)
            
            boost = np.power(pt, -float(spp_pow) / 2.0)
            reg = float(alpha) * np.power(pt, float(spp_pow))
            wiener = s2 / (s2 + reg)
            
            return (boost * wiener).astype(np.float32)

    # ── 시각화 헬퍼 ───────────────────────────────────────────────────────────

    @staticmethod
    def _save_slope_plot(s, n, sl_s, sl_n, a, beta, save_dir, tag):
        os.makedirs(save_dir, exist_ok=True)
        K     = len(s)
        ranks = np.arange(1, K + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, vals, slope, label in zip(
            axes, [s, n[:K]], [sl_s, sl_n],
            ["Singular Values σ_k", "Propensity Values p_i"],
        ):
            ax.scatter(np.log(ranks), np.log(np.clip(vals, 1e-12, None)),
                       s=3, alpha=0.4)
            x_fit = np.linspace(np.log(ranks[0]), np.log(ranks[-1]), 100)
            ic = np.mean(np.log(np.clip(vals, 1e-12, None))) \
               + slope * np.mean(np.log(ranks))
            ax.plot(x_fit, -slope * x_fit + ic, "r--", label=f"slope={slope:.3f}")
            ax.set_xlabel("log rank"); ax.set_ylabel(f"log"); ax.set_title(label)
            ax.legend()
        fig.suptitle(f"{tag}: a={a:.3f}  β={beta:.3f}")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"aspire_slope_{tag}.png"))
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# ASPIRELayer
# ══════════════════════════════════════════════════════════════════════════════

class ASPIRELayer(nn.Module):
    """
    ASPIRE Layer (SVD 기반).

    파이프라인:
      1. SVD:  X = UΣVᵀ
      2. SPP:  p̃_k = Σ_i V_{ki}² · (n_i/n_max)
      3. 필터: 
         - Direct SPP: h(σ_k) = σ_k² / (σ_k² + α · p̃_k^spp_pow)
         - Beta 추정:  h(σ_k) = σ_k^{2-2β} / (σ_k^{2-2β} + α)
      5. 추론: r̂_u = (x_u @ V) ⊙ h · Vᵀ

    속성:
      beta        : MNAR 강도 요약 스칼라
      r_squared   : 멱법칙 가정 성립 정도 (진단용)
    """

    def __init__(
        self,
        k: int | list           = 200,
        alpha: float | list     = 500.0,
        beta: float | str | list = "auto",
        spp_pow: float | list | None = None,  # [Direct SPP] 인기도(SPP)를 편향으로 믿는 정도 (0~1)
        weight_mode: str = "normal",           # [NEW] E-WLS 가중치 모드 (normal | inverse)
        target_energy: float | list = 0.95,
        estimator_type: str = "slope_ratio",
        symmetric_norm: bool = False,
    ):
        super().__init__()
        self.k             = int(k[0] if isinstance(k, (list, np.ndarray)) else k)
        self.alpha         = float(alpha[0] if isinstance(alpha, (list, np.ndarray)) else alpha)
        self.beta_config   = beta[0] if isinstance(beta, (list, np.ndarray)) else beta
        self.spp_pow       = float(spp_pow[0] if isinstance(spp_pow, (list, np.ndarray)) else spp_pow) if spp_pow is not None else None
        self.target_energy = float(
            target_energy[0] if isinstance(target_energy, (list, np.ndarray))
            else target_energy
        )
        self.estimator_type = estimator_type
        self.symmetric_norm = symmetric_norm

        self.beta          = 0.5
        self.r_squared     = 0.0
        self.alignment_slope = 0.0  # 하위 호환

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
        """SVD → SPP → β → h."""
        dev     = next((p.device for p in self.parameters()), torch.device("cpu"))
        manager = SVDCacheManager(device=dev)

        # Get raw item frequencies for beta estimation (ASPIRE logic requires observed bias)
        item_pops_raw = np.array(X_sparse.sum(axis=0)).flatten().astype(float)

        if self.symmetric_norm:
            print(f"[ASPIRELayer] Applying Symmetric Normalization (D_u^-0.5 X D_i^-0.5)...")
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
            # Use _norm suffix for SVD cache to avoid contamination
            svd_dataset_name = f"{dataset_name}_norm" if dataset_name else None
            _, s, v, _ = manager.get_svd(X_target, k=None, target_energy=self.target_energy,
                                         dataset_name=svd_dataset_name)
            
            # Use NORMALIZED frequencies for beta estimation to only correct RESIDUAL bias
            item_pops = np.array(X_target.sum(axis=0)).flatten().astype(float)
        else:
            _, s, v, _ = manager.get_svd(X_sparse, k=None, target_energy=self.target_energy,
                                         dataset_name=dataset_name)
            item_pops = item_pops_raw
            X_target = X_sparse

        self.k = len(s)
        self.register_buffer("singular_values", s.to(dev))
        self.register_buffer("V_raw", v.to(dev))

        V_np      = self.V_raw.cpu().numpy()
        s_np      = self.singular_values.cpu().numpy()

        # ── 3. 진단 및 베타 추정 ─────────────────────────────────
        p_tilde = AspireEngine.compute_spp(V_np, item_pops)
        
        # [NEW] Direct Slope-Ratio (v3 Default) 또는 명시된 추정기 사용
        self.beta, self.r_squared = AspireEngine.estimate_beta(
            self.singular_values, p_tilde,
            verbose=True, dataset_name=dataset_name or "?",
            estimator_type=self.estimator_type or "slope_ratio",
            weight_mode=getattr(self, "weight_mode", "normal"),
            item_freq=item_pops
        )

        # ── 3. 필터 제진 (Integrated Equalizer-Wiener) ────────────────────────
        # spp_pow가 명시적으로 설정된 경우 부스트 항(Equalizer)이 포함된 통합 필터 사용
        if self.spp_pow is not None:
            h = AspireEngine.apply_direct_spp_filter(self.singular_values, p_tilde, self.alpha, self.spp_pow)
        else:
            # 기존 beta 방식 (근본 ASPIRE)
            applied_beta = float(self.beta_config) if not isinstance(self.beta_config, str) else self.beta
            self.beta = applied_beta  
            self.alignment_slope = self.beta * 2.0
            h = AspireEngine.apply_filter(self.singular_values, self.alpha, self.beta)

        self.register_buffer("filter_diag", h)

        print(
            f"[ASPIRELayer] build complete | "
            f"k={self.k}  β_est={self.beta:.4f}  R²={self.r_squared:.4f}  "
            f"Method={self.estimator_type or 'ewls'}  "
            f"spp_pow={self.spp_pow}  device={dev}"
        )

    @torch.no_grad()
    def forward(self, X_batch: torch.Tensor, user_ids=None) -> torch.Tensor:
        if self.singular_values.numel() == 0:
            raise RuntimeError("ASPIRELayer.build()를 먼저 호출하세요.")
        
        X = X_batch
        if self.symmetric_norm:
            # Normalize User and Item side
            if user_ids is not None:
                # D_u^-0.5
                u_w = self.user_norm_weights[user_ids].view(-1, 1)
                X = X * u_w
            # D_i^-0.5
            X = X * self.item_norm_weights.view(1, -1)
            
        XV = torch.mm(X, self.V_raw)
        scores = torch.mm(XV * self.filter_diag, self.V_raw.t())
        
        if self.symmetric_norm:
            # Back to original scale? (GF-CF: often just predicts in norm space, 
            # but to be truly symmetric in Gram: D_u^0.5 ... D_i^0.5)
            # Actually, standard bipartite norm GF-CF ends with normalized factors.
            # But let's re-normalize the output D_i^0.5 if we want to match interaction density.
            # The user said "gram_matrix에 대칭정규화를 적용하는거지". 
            # If G = D^-0.5 (X^T X) D^-0.5, then the filter is on this G.
            # Prediction is x @ D^-0.5 G_filt D^0.5? No.
            # Let's keep it simple: D_u^-0.5 X D_i^-0.5 -> h -> D_i^-0.5? 
            # Usually we don't multiply back by D_u^-0.5 in output score.
            pass
            
        return scores

    @torch.no_grad()
    def visualize_matrices(self, X_sparse=None, save_dir=None, lightweight=False):
        LIRAVisualizer.visualize_spectral_tikhonov(
            self.singular_values, self.filter_diag, self.alpha,
            beta=self.beta, a=self.alignment_slope,
            X_sparse=X_sparse, save_dir=save_dir, file_prefix="aspire",
        )


# ══════════════════════════════════════════════════════════════════════════════
# ChebyASPIRELayer
# ══════════════════════════════════════════════════════════════════════════════

class ChebyASPIRELayer(nn.Module):
    """
    ChebyASPIRE Layer (SVD-free, O(L·E) 복잡도).

    파이프라인:
      1. β:  slope 비율로 추정 (V 없음 → SPP 불가)
      2. h:  h(σ) = σ^{2/(1+β)}/(σ^{2/(1+β)}+α)  →  h(λ) = λ^{1/(1+β)}/(λ^{1/(1+β)}+α)
      3. W:  Chebyshev 다항식으로 h(XᵀX) 근사
      4. 추론: r̂_u = x_u @ W

    SVD가 불가능한 초거대 행렬을 위한 확장.
    β 추정이 slope 기반으로 SPP보다 정보량이 적다.
    SVD가 가능하면 ASPIRELayer를 쓸 것.
    """

    def __init__(
        self,
        alpha: float | list          = 500.0,
        degree: int | list           = 20,
        beta: float | str | list     = "auto",
        lambda_max_estimate: float | str = "auto",
        threshold: float             = 1e-4,
        estimator_type: str          = "huber",
        symmetric_norm: bool         = False,
    ):
        super().__init__()
        self.alpha               = float(alpha[0] if isinstance(alpha, (list, np.ndarray)) else alpha)
        self.degree              = int(degree[0]  if isinstance(degree, (list, np.ndarray)) else degree)
        self.beta_config         = beta[0] if isinstance(beta, (list, np.ndarray)) else beta
        self.lambda_max_estimate = lambda_max_estimate
        self.threshold           = float(threshold)
        self.estimator_type      = estimator_type
        self.symmetric_norm      = symmetric_norm

        self.beta            = 0.5
        self.alignment_slope = 0.0
        self.gamma           = 0.0  # 하위 호환

        self.register_buffer("cheby_coeffs", torch.empty(0))
        self.register_buffer("t_mid",        torch.tensor(0.0))
        self.register_buffer("t_half",       torch.tensor(0.0))
        self.register_buffer("item_weights", torch.empty(0))
        self.register_buffer("user_norm_weights", torch.empty(0))
        self.register_buffer("item_norm_weights", torch.empty(0))

        self.X_torch_csr  = None
        self.Xt_torch_csr = None
        self.sparse_device: torch.device | None = None

    def _aspire_filter(self, lam: np.ndarray) -> np.ndarray:
        """h(λ) = λ^{1/(1+β)} / (λ^{1/(1+β)} + α),  λ = σ²."""
        exp     = float(1.0 / (1.0 + self.beta))
        lam_pow = np.power(np.maximum(lam, 0.0), exp)
        return lam_pow / (lam_pow + self.alpha)

    @torch.no_grad()
    def _estimate_lambda_max(self, X_csr, Xt_csr) -> float:
        v = torch.randn(X_csr.shape[1], 1, device=X_csr.device)
        v = v / v.norm()
        for _ in range(30):
            v          = torch.sparse.mm(Xt_csr, torch.sparse.mm(X_csr, v))
            lambda_est = v.norm().item()
            v          = v / (lambda_est + 1e-12)
        return lambda_est

    def _compute_chebyshev_coeffs(self, lam_min, lam_max, K) -> np.ndarray:
        j         = np.arange(K + 1)
        theta     = np.pi * (j + 0.5) / (K + 1)
        mid, half = (lam_max + lam_min) / 2.0, (lam_max - lam_min) / 2.0
        lam_nodes = mid + half * np.cos(theta)
        f_nodes   = self._aspire_filter(lam_nodes)
        coeffs    = np.zeros(K + 1)
        for k in range(K + 1):
            coeffs[k] = (2.0 / (K + 1)) * np.sum(f_nodes * np.cos(k * theta))
        coeffs[0] /= 2.0
        return coeffs

    @torch.no_grad()
    def build(self, X_sparse, dataset_name: str | None = None):
        """β → Chebyshev 계수 → (소규모) W 사전 계산."""
        if torch.cuda.is_available():       device = "cuda"
        elif torch.backends.mps.is_available(): device = "mps"
        else:                               device = "cpu"
        self.sparse_device = torch.device("cpu" if "mps" in device else device)

        # ── Symmetric Normalization (GF-CF style) ─────────────
        if self.symmetric_norm:
            print(f"[ChebyASPIRELayer] Applying Symmetric Normalization (D_u^-0.5 X D_i^-0.5)...")
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
            
            self.register_buffer("user_norm_weights", torch.from_numpy(w_u).float().to(device))
            self.register_buffer("item_norm_weights", torch.from_numpy(w_i).float().to(device))
            
            X_sparse = sp.diags(w_u) @ X_sparse @ sp.diags(w_i)
            print(f"[ChebyASPIRELayer] Normalization applied to X_sparse.")

        # ── 희소 행렬 변환 ────────────────────────────────────────────────────
        X_coo   = X_sparse.tocoo()
        indices = torch.from_numpy(np.vstack((X_coo.row, X_coo.col))).long()
        values  = torch.from_numpy(X_coo.data).float()
        shape   = X_coo.shape

        X_t  = torch.sparse_coo_tensor(
            indices, values, shape, device=self.sparse_device
        ).coalesce()
        Xt_t = torch.sparse_coo_tensor(
            torch.stack([indices[1], indices[0]]), values,
            (shape[1], shape[0]), device=self.sparse_device
        ).coalesce()
        self.X_torch_csr  = X_t.to_sparse_csr()
        self.Xt_torch_csr = Xt_t.to_sparse_csr()

        # ── λ_max ─────────────────────────────────────────────────────────────
        if self.lambda_max_estimate == "auto":
            lambda_max = self._estimate_lambda_max(
                self.X_torch_csr, self.Xt_torch_csr
            )
        else:
            lambda_max = float(self.lambda_max_estimate)

        # β 결정
        # Update cache key for normalized case
        actual_data_name = f"{dataset_name}_norm" if (dataset_name and self.symmetric_norm) else dataset_name
        cache_key = f"{actual_data_name}_cheby_v13_raw_p1.0" if actual_data_name else None
        cached    = _MNARGammaCache.get(cache_key) if cache_key else None

        if isinstance(self.beta_config, str):  # "auto"
            if cached is not None:
                self.beta            = float(cached["beta"])
                self.alignment_slope = float(cached.get("a", 0.0))
            else:
                # Use current X_sparse (might be normalized) and its item frequencies
                curr_item_pops = np.array(X_sparse.sum(axis=0)).flatten()
                self.beta, a = AspireEngine.estimate_beta_from_slope(
                    singular_values=None,
                    item_frequencies=curr_item_pops, 
                    X_sparse=X_sparse,          
                    verbose=True,
                    dataset_name=actual_data_name or "?",
                    estimator_type=self.estimator_type,
                )
                if cache_key:
                    _MNARGammaCache.put(cache_key, {"beta": self.beta, "a": a})
                self.alignment_slope = a

            # ── 2. 보정 적용 ──────────────────────────────────────────────────
        else:
            self.beta = float(self.beta_config)

        print(f"[ChebyASPIRELayer] β={self.beta:.4f}  λ_max={lambda_max:.2f}")

        # ── Chebyshev 계수 ────────────────────────────────────────────────────
        coeffs = self._compute_chebyshev_coeffs(0.0, lambda_max, self.degree)
        self.register_buffer("cheby_coeffs",
                             torch.from_numpy(coeffs).float().to(device))
        self.register_buffer("t_mid",
                             torch.tensor((lambda_max) / 2.0, device=device))
        self.register_buffer("t_half",
                             torch.tensor((lambda_max) / 2.0, device=device))

        # ── W 사전 계산 (n ≤ 15000) ───────────────────────────────────────────
        n = X_sparse.shape[1]
        L = _GramCache.get(dataset_name, device=device)
        if L is None and n <= 15000:
            print(f"[ChebyASPIRELayer] Gram 행렬 계산 중 (n={n})...")
            # X_dense.t() @ X_dense 가 cuBLAS(CUDA) / Accelerate(MPS) 최적화 경로
            X_dense = self.X_torch_csr.to_dense().to(device)
            L = torch.mm(X_dense.t(), X_dense)
            del X_dense
            if dataset_name:
                _GramCache.put(dataset_name, L)

        if L is not None:
            print(f"[ChebyASPIRELayer] Dense recurrence로 W 계산 중 ({device})...")
            self.item_weights = self._dense_chebyshev(L, coeffs, n, device)
            print(f"[ChebyASPIRELayer] W 완료.")
        else:
            self.item_weights = torch.empty(0)
            print(f"[ChebyASPIRELayer] n={n}>15000, SpMV 모드.")

        print(f"[ChebyASPIRELayer] build 완료.")

    def _dense_chebyshev(self, L, coeffs, n, device) -> torch.Tensor:
        T_prev = torch.eye(n, device=device)
        T_curr = (L - self.t_mid * T_prev) / self.t_half
        W      = float(coeffs[0]) * T_prev + float(coeffs[1]) * T_curr
        for k in range(2, self.degree + 1):
            T_next = (
                2.0 * (torch.mm(L, T_curr) - self.t_mid * T_curr) / self.t_half
                - T_prev
            )
            W.add_(T_next, alpha=float(coeffs[k]))
            T_prev = T_curr
            T_curr = T_next
        return W

    @torch.no_grad()
    def forward(self, X_batch: torch.Tensor, user_ids=None) -> torch.Tensor:
        if self.X_torch_csr is None:
            raise RuntimeError("ChebyASPIRELayer.build()를 먼저 호출하세요.")
            
        X = X_batch
        if self.symmetric_norm:
            if user_ids is not None:
                X = X * self.user_norm_weights[user_ids].view(-1, 1)
            X = X * self.item_norm_weights.view(1, -1)
            
        if self.item_weights.numel() > 0:
            return torch.mm(X, self.item_weights.to(X.device))
        return self._sparsemv_forward(X)

    @torch.no_grad()
    def _sparsemv_forward(self, X_batch: torch.Tensor) -> torch.Tensor:
        dev        = X_batch.device
        coeffs     = self.cheby_coeffs.cpu().numpy()
        t_mid_val  = self.t_mid.item()
        t_half_val = self.t_half.item()
        X_t        = X_batch.t().to(self.sparse_device)

        T_prev = X_t
        inner  = torch.sparse.mm(self.X_torch_csr, T_prev)
        T_curr = torch.sparse.mm(self.Xt_torch_csr, inner)
        T_curr.add_(T_prev, alpha=-t_mid_val)
        T_curr.div_(t_half_val)

        out = T_prev.clone().mul_(float(coeffs[0]))
        out.add_(T_curr, alpha=float(coeffs[1]))

        for k in range(2, self.degree + 1):
            inner  = torch.sparse.mm(self.X_torch_csr, T_curr)
            T_next = torch.sparse.mm(self.Xt_torch_csr, inner)
            T_next.add_(T_curr, alpha=-t_mid_val)
            T_next.div_(t_half_val)
            T_next.mul_(2.0)
            T_next.sub_(T_prev)
            out.add_(T_next, alpha=float(coeffs[k]))
            T_prev = T_curr
            T_curr = T_next

        return out.t().to(dev)

    @torch.no_grad()
    def visualize_matrices(self, X_sparse=None, save_dir=None, lightweight=False):
        if not save_dir: return
        os.makedirs(save_dir, exist_ok=True)

        coeffs     = self.cheby_coeffs.cpu().numpy()
        lambda_max = (self.t_mid + self.t_half).item()
        sigmas     = np.linspace(0, np.sqrt(max(lambda_max, 1e-9)), 200)[::-1]
        lams       = sigmas ** 2

        f_target = self._aspire_filter(lams)
        t_scaled = (lams - self.t_mid.item()) / self.t_half.item()
        T_prev   = np.ones_like(t_scaled)
        T_curr   = t_scaled.copy()
        f_approx = coeffs[0] * T_prev + coeffs[1] * T_curr
        for k in range(2, self.degree + 1):
            T_next    = 2 * t_scaled * T_curr - T_prev
            f_approx += coeffs[k] * T_next
            T_prev = T_curr
            T_curr = T_next

        fit_err = f_approx - f_target
        metrics = {
            "model":  "ChebyASPIRE",
            "params": {"alpha": self.alpha, "beta": self.beta,
                       "degree": self.degree, "lambda_max": float(lambda_max)},
            "fit_quality": {
                "mae":  float(np.mean(np.abs(fit_err))),
                "rmse": float(np.sqrt(np.mean(fit_err ** 2))),
                "max":  float(np.max(np.abs(fit_err))),
            },
        }
        with open(os.path.join(save_dir, "cheby_aspire_metrics.json"), "w", encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].bar(range(len(coeffs)), coeffs)
        axes[0].set_title(f"Chebyshev Coefficients (L={self.degree})")
        axes[0].set_xlabel("k"); axes[0].set_ylabel("c_k")

        axes[1].plot(sigmas, f_target + 1e-12, "k--", alpha=0.4,
                     label="ASPIRE target")
        axes[1].plot(sigmas, f_approx + 1e-12, color="orange",
                     lw=2, label="Cheby fit")
        axes[1].set_yscale("log"); axes[1].invert_xaxis()
        axes[1].set_title(rf"Filter  ($\beta={self.beta:.3f}$)")
        axes[1].set_xlabel(r"$\sigma$ (head→tail)")
        axes[1].set_ylabel("h(σ)"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

        axes[2].plot(sigmas, fit_err, color="red", label="fit error")
        axes[2].axhline(0, color="black", lw=1, ls="--")
        axes[2].invert_xaxis()
        axes[2].set_title("Fit Error"); axes[2].set_xlabel(r"$\sigma$")
        axes[2].legend(); axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "cheby_aspire_analysis.png"))
        plt.close(fig)
        print(f"[ChebyASPIRELayer] 시각화 저장: {save_dir}")
