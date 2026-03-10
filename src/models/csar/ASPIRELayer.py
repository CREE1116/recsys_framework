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
                with open(path) as f:
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
                with open(path, "w") as f:
                    json.dump(payload, f)
            except Exception as e:
                print(f"[MNARGammaCache] save failed: {e}")

    def summary(self):
        files = _glob.glob(os.path.join(self._cache_dir, "mnar_gamma_*.json"))
        return {"type": "MNAR_Gamma",
                "cached_datasets": list(self._mem_cache.keys()),
                "files": len(files)}

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
        p_i     = n_i / (n_i.max() + 1e-9)
        p_tilde = (V_np ** 2).T @ p_i
        return p_tilde

    # ── 2. β 추정 (SPP 기반) ──────────────────────────────────────────────────

    @staticmethod
    def estimate_beta(
        singular_values: torch.Tensor,
        p_tilde: np.ndarray,
        trim: float = 0.05,
        verbose: bool = True,
        dataset_name: str = "",
        return_line: bool = False,
    ) -> tuple:
        """
        p̃_k의 멱법칙 피팅으로 β 추출.

          log p̃_k = 2β · log σ_k + C
          → β = coef / 2
        """
        s  = np.sort(np.abs(AspireEngine._to_numpy(singular_values)))[::-1]
        pt = np.asarray(p_tilde, dtype=float)

        k   = len(s)
        lo  = max(1, int(k * trim))
        hi  = max(lo + 4, int(k * (1 - trim)))
        s_  = s[lo:hi]
        pt_ = pt[lo:hi]

        mask = (s_ > 1e-9) & (pt_ > 1e-9)
        if mask.sum() < 4:
            if verbose:
                print(f"[ASPIRE] {dataset_name}: 유효 포인트 부족 → β=0.5 fallback")
            return (0.5, 0.0, np.zeros_like(pt)) if return_line else (0.5, 0.0)

        log_s  = np.log(s_[mask]).reshape(-1, 1)
        log_pt = np.log(pt_[mask])

        hub = HuberRegressor(epsilon=1.35, max_iter=300, fit_intercept=True)
        hub.fit(log_s, log_pt)

        beta = float(np.clip(hub.coef_[0] / 2.0, 0.0, 0.999))
        r2   = float(hub.score(log_s, log_pt))

        if verbose:
            print(f"[ASPIRE] {dataset_name}: β={beta:.4f}  R²={r2:.4f}")

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

        hub = HuberRegressor(epsilon=1.35, max_iter=300, fit_intercept=True)

        def _slope(vals: np.ndarray) -> float:
            L  = len(vals)
            lo = max(1, int(L * 0.05))
            hi = max(lo + 4, int(L * 0.95))
            r  = np.arange(lo + 1, hi + 1, dtype=float)
            lv = np.log(np.clip(vals[lo:hi], 1e-12, None))
            w  = np.sqrt(r)   # tail-weighted
            hub.fit(np.log(r).reshape(-1, 1), lv, sample_weight=w)
            # 멱법칙 특성상 slope는 음수지만, 우리는 그 '강도'(절댓값)를 원함
            return abs(float(hub.coef_[0]))

        sl_s = _slope(s)
        sl_n = _slope(p_all_sorted)

        if sl_s < 1e-6:
            if verbose:
                print(f"[ASPIRE-slope] {tag}: slope_σ≈0 → MCAR β=0")
            return 0.0, 0.0

        a    = sl_n / sl_s
        beta = float(np.clip(a / 2.0, 0.0, 0.999))

        if verbose:
            print(
                f"[ASPIRE-slope] {tag}: "
                f"slope_σ={sl_s:.3f}  slope_n={sl_n:.3f}  "
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

          h(σ) = σ^{2-2β} / (σ^{2-2β} + α)

        β=0: h = σ²/(σ²+α)  — 표준 Tikhonov (MCAR)
        β→1: h → 1/(1+α)    — 모든 방향 동일 감쇠 (극단 MNAR)
        """
        exponent = float(np.clip(2.0 - 2.0 * beta, 0.01, 2.0))
        if torch.is_tensor(s):
            sp = torch.pow(torch.clamp(s, min=1e-9), exponent)
            return sp / (sp + alpha)
        else:
            sp = np.power(np.clip(AspireEngine._to_numpy(s), 1e-9, None), exponent)
            return sp / (sp + alpha)

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
      2. SPP:  p̃_k = Σ_i V_{ki}² · (n_i/n_max)^spp_pow
      3. β:    log p̃_k = 2β·log σ_k + C  →  β = coef/2  (Huber)
      4. h:    h(σ) = σ^{2-2β} / (σ^{2-2β} + α)
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
        target_energy: float | list = 0.95,
    ):
        super().__init__()
        self.k             = int(k[0] if isinstance(k, (list, np.ndarray)) else k)
        self.alpha         = float(alpha[0] if isinstance(alpha, (list, np.ndarray)) else alpha)
        self.beta_config   = beta[0] if isinstance(beta, (list, np.ndarray)) else beta
        self.target_energy = float(
            target_energy[0] if isinstance(target_energy, (list, np.ndarray))
            else target_energy
        )

        self.beta          = 0.5
        self.r_squared     = 0.0
        self.alignment_slope = 0.0  # 하위 호환

        self.register_buffer("singular_values", torch.empty(0))
        self.register_buffer("V_raw",           torch.empty(0, 0))
        self.register_buffer("filter_diag",     torch.empty(0))

    @property
    def V_k(self) -> torch.Tensor:
        return self.V_raw

    @torch.no_grad()
    def build(self, X_sparse, dataset_name: str | None = None):
        """SVD → SPP → β → h."""
        dev     = next((p.device for p in self.parameters()), torch.device("cpu"))
        manager = SVDCacheManager(device=dev)

        # ── 1. SVD ───────────────────────────────────────────────────────────
        _, s, v, _ = manager.get_svd(
            X_sparse, k=None,
            target_energy=self.target_energy,
            dataset_name=dataset_name,
        )
        self.k = len(s)
        self.register_buffer("singular_values", s.to(dev))
        self.register_buffer("V_raw", v.to(dev))

        item_pops = np.array(X_sparse.sum(axis=0)).flatten().astype(float)
        V_np      = self.V_raw.cpu().numpy()
        s_np      = self.singular_values.cpu().numpy()

        # β 결정
        cache_key = f"{dataset_name}_aspire_v13_p1.0" if dataset_name else None
        cached    = _MNARGammaCache.get(cache_key) if cache_key else None

        if isinstance(self.beta_config, str):  # "auto"
            if cached is not None:
                self.beta        = float(cached["beta"])
                self.r_squared   = float(cached.get("r2", 0.0))
            else:
                # SPP → β
                p_tilde          = AspireEngine.compute_spp(V_np, item_pops)
                self.beta, self.r_squared = AspireEngine.estimate_beta(
                    s_np, p_tilde,
                    verbose=True, dataset_name=dataset_name or "?",
                )
                self.alignment_slope = self.beta * 2.0
                if cache_key:
                    _MNARGammaCache.put(cache_key, {
                        "beta": self.beta,
                        "r2":   self.r_squared,
                    })
        else:
            # 수동 지정 (HPO)
            self.beta = float(self.beta_config)

        # ── 3. 필터 ───────────────────────────────────────────────────────────
        h = AspireEngine.apply_filter(self.singular_values, self.alpha, self.beta)
        self.register_buffer("filter_diag", h)

        print(
            f"[ASPIRELayer] build complete | "
            f"k={self.k}  β={self.beta:.4f}  R²={self.r_squared:.4f}  "
            f"device={dev}"
        )

    @torch.no_grad()
    def forward(self, X_batch: torch.Tensor, user_ids=None) -> torch.Tensor:
        if self.singular_values.numel() == 0:
            raise RuntimeError("ASPIRELayer.build()를 먼저 호출하세요.")
        XV = torch.mm(X_batch, self.V_raw)
        return torch.mm(XV * self.filter_diag, self.V_raw.t())

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
      2. h:  h(σ) = σ^{2-2β}/(σ^{2-2β}+α)  →  h(λ) = λ^{1-β}/(λ^{1-β}+α)
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
    ):
        super().__init__()
        self.alpha               = float(alpha[0] if isinstance(alpha, (list, np.ndarray)) else alpha)
        self.degree              = int(degree[0]  if isinstance(degree, (list, np.ndarray)) else degree)
        self.beta_config         = beta[0] if isinstance(beta, (list, np.ndarray)) else beta
        self.lambda_max_estimate = lambda_max_estimate
        self.threshold           = float(threshold)

        self.beta            = 0.5
        self.alignment_slope = 0.0
        self.gamma           = 0.0  # 하위 호환

        self.register_buffer("cheby_coeffs", torch.empty(0))
        self.register_buffer("t_mid",        torch.tensor(0.0))
        self.register_buffer("t_half",       torch.tensor(0.0))
        self.register_buffer("item_weights", torch.empty(0))

        self.X_torch_csr  = None
        self.Xt_torch_csr = None
        self.sparse_device: torch.device | None = None

    def _aspire_filter(self, lam: np.ndarray) -> np.ndarray:
        """h(λ) = λ^{1-β} / (λ^{1-β} + α),  λ = σ²."""
        exp     = float(np.clip(1.0 - self.beta, 0.005, 1.0))
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
        cache_key = f"{dataset_name}_cheby_v13_p1.0" if dataset_name else None
        cached    = _MNARGammaCache.get(cache_key) if cache_key else None

        if isinstance(self.beta_config, str):  # "auto"
            if cached is not None:
                self.beta            = float(cached["beta"])
                self.alignment_slope = float(cached.get("a", 0.0))
            else:
                item_pops = np.array(X_sparse.sum(axis=0)).flatten()
                self.beta, a = AspireEngine.estimate_beta_from_slope(
                    singular_values=None,
                    item_frequencies=item_pops,
                    X_sparse=X_sparse,
                    verbose=True,
                    dataset_name=dataset_name or "?",
                )
                self.alignment_slope = a
                if cache_key:
                    _MNARGammaCache.put(cache_key, {"beta": self.beta, "a": a})
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
            L = torch.sparse.mm(
                self.Xt_torch_csr.to(self.sparse_device),
                self.X_torch_csr.to_dense().to(self.sparse_device),
            ).to(device)
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
        if self.item_weights.numel() > 0:
            return torch.mm(X_batch, self.item_weights.to(X_batch.device))
        return self._sparsemv_forward(X_batch)

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
        with open(os.path.join(save_dir, "cheby_aspire_metrics.json"), "w") as f:
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
