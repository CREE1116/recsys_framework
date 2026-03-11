"""
Beta Estimators for ASPIRE
입력: sigma_k (np.ndarray), p_tilde_k (np.ndarray)  ← SPP 결과물
출력: beta (float)
"""

import numpy as np
from scipy.stats import theilslopes
from scipy.optimize import linprog


def _log_xy(sigma_k, p_tilde_k):
    """공통 전처리: log 변환 + 유효 인덱스 필터링"""
    mask = (sigma_k > 0) & (p_tilde_k > 0)
    if mask.sum() < 2:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])  # Fallback
    x = np.log(sigma_k[mask])  # log σ_k
    y = np.log(p_tilde_k[mask])  # log p̃_k
    return x, y


def _compute_r2(x, y, slope, intercept):
    """결정계수(R^2) 계산"""
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-12))


# ------------------------------------------------------------------
# 1. Slope 비율 (Corollary 2 직접) — 비스펙트럴, 이론 하한
# ------------------------------------------------------------------
def beta_slope_ratio(item_freq, sigma_k):
    """
    Corollary 2: β = η / (2α)
    η = slope(log n_i ~ log rank_i)
    α = slope(log σ_k ~ log k)
    SPP 결과물 대신 item_freq 직접 사용 — 비스펙트럴
    """
    n = np.sort(item_freq)[::-1]
    rank = np.arange(1, len(n) + 1)
    mask_n = n > 0
    if mask_n.sum() < 2: return 0.5
    
    eta, _, _, _ = np.linalg.lstsq(
        np.column_stack([np.log(rank[mask_n]), np.ones(mask_n.sum())]),
        np.log(n[mask_n]), rcond=None
    )
    eta = -eta[0]  # 부호 반전 (감소 멱법칙)

    k = np.arange(1, len(sigma_k) + 1)
    mask_s = sigma_k > 0
    if mask_s.sum() < 2: return 0.5
    
    alpha, _, _, _ = np.linalg.lstsq(
        np.column_stack([np.log(k[mask_s]), np.ones(mask_s.sum())]),
        np.log(sigma_k[mask_s]), rcond=None
    )
    alpha = -alpha[0]

    beta = eta / (2 * alpha)
    return float(np.clip(beta, 0, 1)), 0.0 # Slope ratio doesn't have a single R2




# ------------------------------------------------------------------
# 5. MAD-Adaptive Huber (파라미터 없음)
# ------------------------------------------------------------------
def beta_huber_mad(sigma_k, p_tilde_k):
    """
    Step 1: OLS pilot → 잔차
    Step 2: δ_H = 1.4826 × MAD(잔차)  ← 데이터 주도, 파라미터 없음
    Step 3: Huber IRLS(δ_H) 최종 추정
    """
    x, y = _log_xy(sigma_k, p_tilde_k)
    if len(x) < 2: return 0.5
    A = np.column_stack([x, np.ones_like(x)])

    # Step 1: OLS pilot
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    residuals = y - A @ coef

    # Step 2: δ_H = 1.4826 × MAD (신호 방향 오염 없는 scale 추정)
    mad = np.median(np.abs(residuals - np.median(residuals)))
    delta_h = max(1.4826 * mad, 1e-6)

    # Step 3: Huber IRLS (delta_h 절대 임계값)
    for _ in range(100):
        resid = y - A @ coef
        w = np.where(np.abs(resid) <= delta_h, 1.0, delta_h / np.clip(np.abs(resid), 1e-9, None))
        W = np.diag(w)
        coef_new = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ y, rcond=None)[0]
        if np.max(np.abs(coef_new - coef)) < 1e-8:
            break
        coef = coef_new

    slope, intercept = coef[0], coef[1]
    r2 = _compute_r2(x, y, slope, intercept)
    return float(np.clip(slope / 2, 0, 1)), float(r2)


# ------------------------------------------------------------------
# 6. LAD — L+S 분해의 convex relaxation (이론 최적)
# ------------------------------------------------------------------
def beta_lad(sigma_k, p_tilde_k):
    """
    min Σ|log p̃_k − 2β·log σ_k − C|  →  L1 최소화 = LAD
    """
    x, y = _log_xy(sigma_k, p_tilde_k)
    K = len(x)
    if K < 2: return 0.5

    # 변수: [β, C, t_1, ..., t_K]  (K+2 variables)
    n_vars = K + 2
    c = np.zeros(n_vars)
    c[2:] = 1.0

    A_ub = np.zeros((2 * K, n_vars))
    b_ub = np.zeros(2 * K)

    for i in range(K):
        A_ub[i, 0] = -2 * x[i]
        A_ub[i, 1] = -1.0
        A_ub[i, 2 + i] = -1.0
        b_ub[i] = -y[i]

        A_ub[K + i, 0] = 2 * x[i]
        A_ub[K + i, 1] = 1.0
        A_ub[K + i, 2 + i] = -1.0
        b_ub[K + i] = y[i]

    bounds = [(None, None), (None, None)] + [(0, None)] * K
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.success:
        beta, intercept = result.x[0], result.x[1]
        r2 = _compute_r2(x, y, 2 * beta, intercept)
        return float(np.clip(beta, 0, 1)), float(r2)
    else:
        return 0.5, 0.0


# ------------------------------------------------------------------
# 7. SPP + LAD 통합 (메인 추정량)
# ------------------------------------------------------------------
def beta_spp_lad(V, p_i, sigma_k):
    """
    β* = argmin_β || log diag(V^T P V) - 2β·log σ - C·1 ||_1
    """
    if V.shape[0] == len(p_i):
        p_tilde = (V ** 2).T @ p_i
    else:
        p_tilde = (V ** 2) @ p_i

    mask = (sigma_k > 0) & (p_tilde > 0)
    x = np.log(sigma_k[mask])
    y = np.log(p_tilde[mask])
    K = mask.sum()
    if K < 2: return 0.5

    n_vars = K + 2
    c_obj = np.zeros(n_vars)
    c_obj[2:] = 1.0

    A_ub = np.zeros((2 * K, n_vars))
    b_ub = np.zeros(2 * K)
    for i in range(K):
        A_ub[i,   0] = -2*x[i]; A_ub[i,   1] = -1.0; A_ub[i,   2+i] = -1.0; b_ub[i]   = -y[i]
        A_ub[K+i, 0] =  2*x[i]; A_ub[K+i, 1] =  1.0; A_ub[K+i, 2+i] = -1.0; b_ub[K+i] =  y[i]

    bounds = [(None, None), (None, None)] + [(0, None)] * K
    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.success:
        beta, intercept = result.x[0], result.x[1]
        r2 = _compute_r2(x, y, 2 * beta, intercept)
        return float(np.clip(beta, 0, 1)), float(r2)
    else:
        return 0.5, 0.0


# ------------------------------------------------------------------
# 8. β = 0.5 고정 (IPS heuristic 기준선)
# ------------------------------------------------------------------
def beta_fixed(sigma_k=None, p_tilde_k=None, value=0.5):
    return float(value), 1.0


def estimate_all(sigma_k, p_tilde_k, item_freq=None, V=None, p_i=None):
    results = {
        "huber_mad":     beta_huber_mad(sigma_k, p_tilde_k),
        "lad":           beta_lad(sigma_k, p_tilde_k),
        "fixed_0.5":     beta_fixed(),
    }
    if item_freq is not None:
        results["slope_ratio"] = beta_slope_ratio(item_freq, sigma_k)
    if V is not None and p_i is not None:
        results["spp_lad"] = beta_spp_lad(V, p_i, sigma_k)
    return results
