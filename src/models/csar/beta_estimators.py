"""
Beta Estimators for ASPIRE (Clean Reset)
이론적 정합성: 
  p̃_k ∝ σ_k^{2β} 
  log p̃_k = (2β) * log σ_k + C
  slope = 2β  =>  β = slope / 2
"""

import numpy as np
import torch
from scipy.optimize import linprog

def _log_xy(sigma_k, p_tilde_k):
    """log 변환 및 유효 데이터 필터링"""
    # sigma_k and p_tilde_k can be torch tensors
    if torch.is_tensor(sigma_k): sigma_k = sigma_k.detach().cpu().numpy()
    if torch.is_tensor(p_tilde_k): p_tilde_k = p_tilde_k.detach().cpu().numpy()
    
    mask = (sigma_k > 1e-12) & (p_tilde_k > 1e-12)
    if mask.sum() < 2:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])
    
    x = np.log(sigma_k[mask])
    y = np.log(p_tilde_k[mask])
    return x, y

def _compute_r2(x, y, beta):
    """β 정의에 맞춘 R² 계산 (slope = 2β)"""
    slope = 2.0 * beta
    intercept = np.mean(y - slope * x)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-12))

def beta_ols(sigma_k, p_tilde_k):
    """OLS: Ordinary Least Squares (L2)"""
    x, y = _log_xy(sigma_k, p_tilde_k)
    A = np.column_stack([x, np.ones_like(x)])
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    
    slope = coef[0]
    beta = slope / 2.0
    return float(beta), float(_compute_r2(x, y, beta))

def beta_lad(sigma_k, p_tilde_k):
    """LAD: Least Absolute Deviations (L1 Robust)"""
    x, y = _log_xy(sigma_k, p_tilde_k)
    K = len(x)
    # 변수: [slope, C, t_1, ..., t_K]
    n_vars = K + 2
    c = np.zeros(n_vars); c[2:] = 1.0
    
    A_ub = np.zeros((2*K, n_vars))
    b_ub = np.zeros(2*K)
    for i in range(K):
        # slope*x + C - t_i <= y
        # -slope*x - C - t_i <= -y
        A_ub[i, 0] = x[i];   A_ub[i, 1] = 1.0;  A_ub[i, 2+i] = -1.0; b_ub[i] = y[i]
        A_ub[K+i, 0] = -x[i]; A_ub[K+i, 1] = -1.0; A_ub[K+i, 2+i] = -1.0; b_ub[K+i] = -y[i]
        
    bounds = [(None, None), (None, None)] + [(0, None)] * K
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        slope = res.x[0]
        beta = slope / 2.0
        return float(beta), float(_compute_r2(x, y, beta))
    return 0.5, 0.0


def beta_spp_projection_shifted(sigma_k, p_tilde_k):
    """
    Shifted ASPIRE estimator (Anchored at pop=1.0)
    이전 실험에서 '아름다운 보간'을 보여준 버전.
    x축은 sigma_max를 0으로 맞추고, y축은 원본을 사용하여 (log sigma_max, 0) 지점을 관통하게 함.
    """
    x_raw, y_raw = _log_xy(sigma_k, p_tilde_k)
    if len(x_raw) < 1: return 0.5, 0.0
    
    # x is shifted relative to first point, y is absolute pop-intensity
    x = x_raw - x_raw[0]
    y = y_raw

    num = np.sum(x * y)
    den = np.sum(x * x)

    slope = num / (den + 1e-12)
    beta = slope / 2.0

    return float(beta), float(_compute_r2(x_raw, y_raw, beta))

def beta_covariance(sigma_k, p_tilde_k):
    """
    β = 1/2 * Cov(log σ, log p̃) / Var(log σ)
    """
    x, y = _log_xy(sigma_k, p_tilde_k)

    x_mean = x.mean()
    y_mean = y.mean()

    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sum((x - x_mean) ** 2)

    slope = num / (den + 1e-12)
    beta = slope / 2.0

    return float(beta), float(_compute_r2(x, y, beta))


def beta_pairwise_ratio(sigma_k, p_tilde_k):
    """
    Pairwise log-ratio estimator
    """
    x, y = _log_xy(sigma_k, p_tilde_k)

    K = len(x)
    if K < 2:
        return 0.5, 0.0

    betas = []
    for i in range(K):
        for j in range(i+1, K):
            dx = x[i] - x[j]
            if abs(dx) < 1e-12:
                continue
            dy = y[i] - y[j]
            slope = dy / dx
            betas.append(slope / 2.0)

    if len(betas) == 0:
        return 0.5, 0.0

    beta = np.median(betas)
    return float(beta), float(_compute_r2(x, y, beta))

def beta_slope_ratio(sigma_k, item_freq):
    """
    Direct Slope-Ratio Estimator (β_direct = η / 2α)
    
    Step 1. 스펙트럴 감쇠율(α) 직접 측정: log σ_k ~ -α log k
    Step 2. 아이템 편향률(η) 직접 측정: log p_i ~ -η log i
    Step 3. 냅다 나누기: β = η / 2α
    """
    s = np.sort(np.abs(sigma_k))[::-1]
    n = np.sort(np.abs(item_freq))[::-1]
    
    def _get_abs_slope(vals):
        L = len(vals)
        if L < 2: return 1.0
        x = np.log(np.arange(1, L + 1))
        y = np.log(np.clip(vals, 1e-12, None))
        
        # Simple OLS for "Direct" method as requested
        A = np.column_stack([x, np.ones_like(x)])
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        return abs(float(slope))

    alpha = _get_abs_slope(s)
    eta   = _get_abs_slope(n)
    
    if alpha < 1e-12:
        return 0.5, 0.0
        
    beta = eta / (2.0 * alpha)
    return float(beta), 1.0

def estimate_all(sigma_k, p_tilde_k, item_freq=None):
    """가용한 모든 추정기 리스트링"""
    results = {
        "ols": beta_ols(sigma_k, p_tilde_k),
        "lad": beta_lad(sigma_k, p_tilde_k),
        "spp_proj_shifted": beta_spp_projection_shifted(sigma_k, p_tilde_k),
        "covariance": beta_covariance(sigma_k, p_tilde_k),
        "pairwise": beta_pairwise_ratio(sigma_k, p_tilde_k),
    }
    if item_freq is not None:
        results["slope_ratio"] = beta_slope_ratio(sigma_k, item_freq)
        
    return results
