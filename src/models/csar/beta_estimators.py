"""
Beta Estimators for ASPIRE (v2 - 이론 정합성 수정)

이론적 기반:
  왜곡 모델: σ_obs = σ_true^{1+β}
  Assumption A: p_i ∝ σ_true^{2β}
  Corollary 1: p̃_k ∝ σ_obs^{2β/(1+β)}

  → SPP 공간 회귀: log p̃_k = slope · log σ_obs + C
    slope = 2β/(1+β)  →  β = slope / (2 - slope)  [기존 slope/2는 오류]

  → slope_ratio: η = 2β·α_s/(1+β)
    →  β = η / (2α_s - η)
"""

import numpy as np
from scipy.optimize import linprog


def _log_xy(sigma_k, p_tilde_k):
    """log 변환 및 유효 데이터 필터링"""
    try:
        import torch
        if torch.is_tensor(sigma_k):  sigma_k  = sigma_k.detach().cpu().numpy()
        if torch.is_tensor(p_tilde_k): p_tilde_k = p_tilde_k.detach().cpu().numpy()
    except ImportError:
        pass

    sigma_k   = np.asarray(sigma_k,   dtype=float)
    p_tilde_k = np.asarray(p_tilde_k, dtype=float)

    mask = (sigma_k > 1e-12) & (p_tilde_k > 1e-12)
    if mask.sum() < 2:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])

    return np.log(sigma_k[mask]), np.log(p_tilde_k[mask])


def _slope_to_beta(slope):
    """
    Corollary 1 기준 역산:
      slope = 2β/(1+β)  →  β = slope / (2 - slope)
    slope ≥ 2이면 β → ∞ (이론 범위 초과, 클램프)
    slope < 0이면 MCAR fallback β = 0
    """
    if slope <= 0:
        return 0.0
    if slope >= 2.0:
        return 10.0  # 사실상 무한대, β > 10은 의미 없음
    return slope / (2.0 - slope)


def _compute_r2(x, y, beta):
    """
    Corollary 1 기준 R²:
      y_pred = slope·x + C,  slope = 2β/(1+β)
    """
    slope = 2.0 * beta / (1.0 + beta)
    intercept = np.mean(y) - slope * np.mean(x)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


# ------------------------------------------------------------------
# SPP 공간 추정량들 (slope → β = slope/(2-slope))
# ------------------------------------------------------------------

def beta_ols(sigma_k, p_tilde_k):
    """OLS: slope = 2β/(1+β) → β = slope/(2-slope)"""
    x, y = _log_xy(sigma_k, p_tilde_k)
    A = np.column_stack([x, np.ones_like(x)])
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    beta = _slope_to_beta(coef[0])
    return float(beta), _compute_r2(x, y, beta)


def beta_lad(sigma_k, p_tilde_k):
    """LAD: L1 robust, slope → β = slope/(2-slope)"""
    x, y = _log_xy(sigma_k, p_tilde_k)
    K = len(x)
    n_vars = K + 2
    c = np.zeros(n_vars); c[2:] = 1.0

    A_ub = np.zeros((2 * K, n_vars))
    b_ub = np.zeros(2 * K)
    for i in range(K):
        A_ub[i,   0] =  x[i]; A_ub[i,   1] =  1.0; A_ub[i,   2+i] = -1.0; b_ub[i]   =  y[i]
        A_ub[K+i, 0] = -x[i]; A_ub[K+i, 1] = -1.0; A_ub[K+i, 2+i] = -1.0; b_ub[K+i] = -y[i]

    bounds = [(None, None), (None, None)] + [(0, None)] * K
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if res.success:
        beta = _slope_to_beta(res.x[0])
        return float(beta), _compute_r2(x, y, beta)
    return 0.0, 0.0


def beta_pairwise_ratio(sigma_k, p_tilde_k):
    """Theil-Sen 스타일: 쌍별 기울기 중앙값 → β = slope/(2-slope)"""
    x, y = _log_xy(sigma_k, p_tilde_k)
    K = len(x)
    if K < 2:
        return 0.0, 0.0

    slopes = []
    for i in range(K):
        for j in range(i + 1, K):
            dx = x[i] - x[j]
            if abs(dx) < 1e-12:
                continue
            slopes.append((y[i] - y[j]) / dx)

    if not slopes:
        return 0.0, 0.0

    beta = _slope_to_beta(float(np.median(slopes)))
    return beta, _compute_r2(x, y, beta)


# ------------------------------------------------------------------
# 전체 추정
# ------------------------------------------------------------------

def estimate_all(sigma_k, p_tilde_k, item_freq=None):
    results = {
        "ols":              beta_ols(sigma_k, p_tilde_k),
        "lad":              beta_lad(sigma_k, p_tilde_k),
        "pairwise":         beta_pairwise_ratio(sigma_k, p_tilde_k),
    }
    return results


# ------------------------------------------------------------------
# 테스트
# ------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    K   = 300
    true_beta = 0.8
    alpha_s   = 0.5

    # 왜곡 모델: σ_obs = σ_true^{1+β}
    sigma_true = np.exp(-alpha_s * np.log(np.arange(1, K+1)))
    sigma_obs  = sigma_true ** (1 + true_beta)

    # Corollary 1: p̃_k ∝ σ_obs^{2β/(1+β)}
    exp_corollary = 2 * true_beta / (1 + true_beta)
    p_tilde = sigma_obs ** exp_corollary * np.exp(rng.normal(0, 0.05, K))
    p_tilde = np.clip(p_tilde, 1e-12, None)

    print(f"true β = {true_beta},  α_s = {alpha_s}")
    print(f"기대 slope (Corollary 1) = 2β/(1+β) = {exp_corollary:.4f}\n")

    print(f"{'방법':<22}  {'β_hat':>7}  {'err':>7}  {'R²':>6}")
    print("-" * 50)
    res = estimate_all(sigma_obs, p_tilde)
    for name, val in res.items():
        b, r2 = val
        print(f"  {name:<20}  {b:7.4f}  {b-true_beta:+7.4f}  {r2:6.3f}")

