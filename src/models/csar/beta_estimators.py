import numpy as np
from scipy.optimize import linprog

def _log_xy(sigma_k, p_k, trim_tail=0.0):
    """log 변환 및 유효 데이터 필터링"""
    try:
        import torch
        if torch.is_tensor(sigma_k): sigma_k = sigma_k.detach().cpu().numpy()
        if torch.is_tensor(p_k):     p_k     = p_k.detach().cpu().numpy()
    except ImportError:
        pass

    sigma_k = np.asarray(sigma_k, dtype=float)
    p_k     = np.asarray(p_k,     dtype=float)

    mask = (sigma_k > 1e-15) & (p_k > 1e-15)
    x = np.log(sigma_k[mask])
    y = np.log(p_k[mask])

    if len(x) < 2:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])

    if trim_tail > 0:
        n = len(x)
        end_idx = max(int(n * (1.0 - trim_tail)), 2)
        x, y = x[:end_idx], y[:end_idx]

    return x, y

def _compute_r2(x, y, slope):
    """R² calculation for linear fit."""
    intercept = np.mean(y) - slope * np.mean(x)
    y_pred    = slope * x + intercept
    ss_res    = np.sum((y - y_pred) ** 2)
    ss_tot    = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))

def _lad_solve(x, y):
    """LAD LP 풀기 → slope 반환"""
    K      = len(x)
    n_vars = K + 2
    c      = np.zeros(n_vars); c[2:] = 1.0
    A_ub = np.zeros((2 * K, n_vars))
    b_ub = np.zeros(2 * K)
    for i in range(K):
        A_ub[i,   0] =  x[i]; A_ub[i,   1] =  1.0; A_ub[i,   2+i] = -1.0; b_ub[i]   =  y[i]
        A_ub[K+i, 0] = -x[i]; A_ub[K+i, 1] = -1.0; A_ub[K+i, 2+i] = -1.0; b_ub[K+i] = -y[i]
    bounds = [(None, None), (None, None)] + [(0, None)] * K
    res    = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    return float(res.x[0]) if res.success else None

def beta_ols(sigma_k, p_k, trim_tail=0.0):
    """OLS: returns raw slope"""
    x, y  = _log_xy(sigma_k, p_k, trim_tail)
    A     = np.column_stack([x, np.ones_like(x)])
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    slope = float(coef[0])
    return slope, _compute_r2(x, y, slope)

def beta_lad(sigma_k, p_k, trim_tail=0.0):
    """LAD: returns raw slope"""
    x, y  = _log_xy(sigma_k, p_k, trim_tail)
    slope = _lad_solve(x, y)
    if slope is None: return 0.0, 0.0
    return slope, _compute_r2(x, y, slope)

def beta_log_derivative(sigma_k, p_tilde_k, q=0.5, **kwargs):
    """
    [ASPIRE v3] Pure Finite Difference Estimator (v2 Mapping)
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    
    # 1. Input Normalization & Log Transform
    # s is assumed descending.
    s = np.asarray(sigma_k, dtype=float).flatten()
    pt = np.asarray(p_tilde_k, dtype=float).flatten()
    idx = np.argsort(s)[::-1]
    s_sorted = s[idx]
    pt_aligned = pt[idx]
    
    mask = (s_sorted > 1e-15) & (pt_aligned > 1e-15)
    if mask.sum() < 5:
        return 0.5, 0.0, {"error": "too_few_samples"}
        
    x_raw = np.log(s_sorted[mask])
    y_raw = np.log(pt_aligned[mask])
    
    # 2. Moving Average Smoothing
    # Stabilization against point-to-point discretization noise
    smooth_window = int(kwargs.get('smooth_window', 3))
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / float(smooth_window)
        x = np.convolve(x_raw, kernel, mode='valid')
        y = np.convolve(y_raw, kernel, mode='valid')
    else:
        x, y = x_raw, y_raw
    
    # Ranking Consistency Check
    from scipy.stats import spearmanr
    rho, _ = spearmanr(x_raw, y_raw)
    
    dx = np.diff(x)
    dy = np.diff(y)
    gamma_hat_raw = dy / (dx + 1e-15)
    
    # 3. Aggregate
    # Filter by dx floor for numerical stability
    valid = np.abs(dx) > 1e-4
    gamma_hat_valid = gamma_hat_raw[valid]
    
    # Enforce non-negative slopes as requested by user ("음수는 0으로 두고")
    gamma_hat = np.maximum(0.0, gamma_hat_valid)
    
    if len(gamma_hat) == 0:
        return 0.5, 0.0, {"error": "no_valid_slopes", "n_total": len(dx), "spearman_rho": float(rho)}
    
    # [USER] q="auto" mode: Rank-based Analytic Q-Fitting
    if q == "auto":
        # 1. Target: Global Zipfian Slope of P
        ln_k_full = np.log(np.arange(1, len(y) + 1))
        A = np.column_stack([ln_k_full, np.ones_like(ln_k_full)])
        target_slope_p = -np.linalg.lstsq(A, y, rcond=None)[0][0]
        
        # 1b. Target: Global Zipfian Slope of S (to get the ratio for beta)
        target_slope_s = -np.linalg.lstsq(A, x, rcond=None)[0][0]
        gamma_target = target_slope_p / (target_slope_s + 1e-12)
        
        # 2. Local slopes (Zipfian)
        dy_k = np.diff(y)
        dx_k = np.diff(ln_k_full)
        local_slopes_p = - dy_k / (dx_k + 1e-15)
        
        # Match with target_slope_p to find q
        valid_slopes = np.sort(local_slopes_p[local_slopes_p > 1e-6])
        q_star = np.searchsorted(valid_slopes, target_slope_p) / (len(valid_slopes) + 1e-12)
        q_used = float(np.clip(q_star, 0.01, 0.99))
        
        # 3. Final Beta from the ratio target
        gamma_capped = min(gamma_target, 1.99)
        beta = gamma_capped / (2.0 - gamma_capped)
        r2 = _compute_r2(x, y, 2.0 * beta / (1.0 + beta))
    else:
        # Standard Quantile Mode
        gamma_tilde = float(np.nanquantile(gamma_hat, q))
        gamma_capped = min(gamma_tilde, 1.99)
        beta = gamma_capped / (2.0 - gamma_capped)
        r2 = _compute_r2(x, y, 2.0 * beta / (1.0 + beta))
        q_used = q
    
    diag = {
        "q_used": q_used,
        "n_valid": int(valid.sum()),
        "valid_ratio": float(valid.sum()) / len(dx) if len(dx) > 0 else 0.0,
        "spearman_rho": float(rho),
        "method": "moving_average_pos_only",
        "smooth_window": smooth_window
    }
    return beta, r2, diag

def beta_simple_slope(sigma_k, y_k, trim_tail=0.0):
    return beta_ols(sigma_k, y_k, trim_tail)

def beta_dynamic_derivative(sigma_k, p_tilde_k, q=0.5, **kwargs):
    """
    [ASPIRE v3] Parameter-Free Dynamic-Stride Estimator
    Comparing k and k//2 points to stabilize d_ln_k and avoid head outliers.
    """
    # 1. Prepare and Sort
    # sigma_k might be torch tensor or numpy
    s = np.sort(np.abs(sigma_k.cpu().numpy() if hasattr(sigma_k, 'cpu') else sigma_k))[::-1]
    y_raw = p_tilde_k.cpu().numpy() if hasattr(p_tilde_k, 'cpu') else p_tilde_k
    
    # Simple alignment check: ensure y is aligned with sorted s
    # In ASPIRE, p_tilde is already calculated per-k (the k-th component)
    # So we just take it as is.
    ln_y = np.log(np.clip(y_raw, 1e-15, None))
    n = len(ln_y)
    
    if n < 3:
        return 0.5, 0.0, {"error": "insufficient_data"}
        
    # 2. Dynamic Stride Slopes
    # k and k//2 comparison stabilizes denominator (ln 2)
    # Start from index 2 to naturally defend against head outliers
    indices = np.arange(2, n)
    half_indices = indices // 2
    
    # d_ln_y / d_ln_k
    # Note: ln_y is log(p_tilde), x is log(sigma)
    # We want d_log_p / d_log_sigma
    ln_x = np.log(s + 1e-15)
    
    # Local slope gamma = (ln_p[k] - ln_p[k//2]) / (ln_s[k] - ln_s[k//2])
    d_ln_y = ln_y[indices] - ln_y[half_indices]
    d_ln_x = ln_x[indices] - ln_x[half_indices]
    
    # Avoid division by zero
    valid_mask = np.abs(d_ln_x) > 1e-7
    if not np.any(valid_mask):
        return 0.5, 0.0, {"error": "no_variation_in_x"}
        
    slopes = d_ln_y[valid_mask] / d_ln_x[valid_mask]

    # 3. Robust Aggregation
    # Filter only positive slopes as per power-law decay expectation
    valid_slopes = slopes[slopes > 1e-6]
    
    if len(valid_slopes) == 0:
        return 0.5, 0.0, {"error": "no_valid_slopes"}

    # [USER] q="auto" mode: Rank-based Analytic Q-Fitting for Dynamic Stride
    if q == "auto":
        ln_p = ln_y
        ln_k = np.log(np.arange(1, n + 1))
        
        # 1. Target: Global Zipfian Slopes
        A = np.column_stack([ln_k, np.ones_like(ln_k)])
        target_slope_p = -np.linalg.lstsq(A, ln_p, rcond=None)[0][0]
        target_slope_s = -np.linalg.lstsq(A, ln_x, rcond=None)[0][0]
        gamma_target = target_slope_p / (target_slope_s + 1e-12)
        
        # 2. Match with Local Distribution (Zipfian)
        idx_match = np.arange(2, n)
        h_match = idx_match // 2
        local_slopes_p = - (ln_p[idx_match] - ln_p[h_match]) / (np.log(idx_match) - np.log(h_match) + 1e-15)
        
        sorted_valid = np.sort(local_slopes_p[local_slopes_p > 1e-6])
        q_star = np.searchsorted(sorted_valid, target_slope_p) / (len(sorted_valid) + 1e-12)
        q_used = float(np.clip(q_star, 0.01, 0.99))
        
        # 3. Final Beta
        gamma_capped = min(gamma_target, 1.95)
        beta = gamma_capped / (2.0 - gamma_capped)
        r2 = _compute_r2(ln_x, ln_y, 2.0 * beta / (1.0 + beta))
    else:
        # Final slope gamma_tilde
        final_slope = float(np.nanquantile(valid_slopes, q))
        gamma_capped = min(final_slope, 1.95)
        beta = gamma_capped / (2.0 - gamma_capped)
        r2 = _compute_r2(ln_x, ln_y, 2.0 * beta / (1.0 + beta))
        q_used = q
    
    return float(beta), r2, {"q_used": q_used, "n_valid": len(valid_slopes), "method": "dynamic_stride"}

def beta_slope_ratio(sigma_k, p_k, trim_tail=0.05):
    """
    Estimate beta from the ratio of Zipfian slopes:
    p ~ k^-Sp,  sigma ~ k^-Ss  =>  p ~ sigma^(Sp/Ss)
    gamma = Sp / Ss
    beta = gamma / (2 - gamma)
    """
    # 1. Estimate Sp (Propensity slope vs rank)
    ranks = np.arange(1, len(p_k) + 1)
    log_ranks = np.log(ranks)
    log_p = np.log(np.clip(p_k, 1e-12, None))
    
    # Simple OLS for Sp
    A_p = np.column_stack([log_ranks, np.ones_like(log_ranks)])
    coef_p, _, _, _ = np.linalg.lstsq(A_p, log_p, rcond=None)
    sp = -float(coef_p[0]) # Zipf slope is positive
    
    # 2. Estimate Ss (Singular value slope vs rank)
    log_s = np.log(np.clip(sigma_k, 1e-12, None))
    A_s = np.column_stack([log_ranks, np.ones_like(log_ranks)])
    coef_s, _, _, _ = np.linalg.lstsq(A_s, log_s, rcond=None)
    ss = -float(coef_s[0])
    
    # 3. Ratio to Gamma
    gamma = sp / (ss + 1e-12)
    
    # 4. Map to Beta (Unified)
    gamma_capped = np.clip(gamma, 0.0, 1.98)
    beta = gamma_capped / (2.0 - gamma_capped)
    
    return float(beta), float(gamma)