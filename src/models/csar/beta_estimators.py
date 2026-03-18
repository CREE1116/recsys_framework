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
    
    # 2. Moving Average Smoothing (window=3) - User Specified Logic
    kernel = np.ones(3) / 3.0
    x = np.convolve(x_raw, kernel, mode='valid')
    y = np.convolve(y_raw, kernel, mode='valid')
    
    # Ranking Consistency Check (User's Hypothesis)
    from scipy.stats import spearmanr
    rho, _ = spearmanr(x_raw, y_raw)
    
    dx = np.diff(x)
    dy = np.diff(y)
    gamma_hat_raw = dy / (dx + 1e-15)
    
    # 3. Filter & Aggregate
    # Filter by dx floor for numerical stability
    # Use all local slopes (including negative ones) to avoid biasing the median
    valid = np.abs(dx) > 1e-4
    gamma_hat = gamma_hat_raw[valid]
    
    if len(gamma_hat) == 0:
        return 0.5, 0.0, {"error": "no_valid_slopes", "n_total": len(dx), "spearman_rho": float(rho)}
    
    # Find quantile of valid slopes
    gamma_tilde = float(np.nanquantile(gamma_hat, q))
    
    # Static mapping back to beta (v2: gamma = 2b/(1+b))
    beta = gamma_tilde / (2.0 - gamma_tilde) if gamma_tilde < 2.0 else 10.0
    beta = float(max(0.0, beta))
    
    # R2 on smoothed curve
    slope = 2.0 * beta / (1.0 + beta)
    intercept = np.mean(y) - slope * np.mean(x)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
    
    diag = {
        "gamma_tilde": gamma_tilde, 
        "n_valid": int(valid.sum()),
        "valid_ratio": float(valid.sum()) / len(dx) if len(dx) > 0 else 0.0,
        "spearman_rho": float(rho),
        "method": "moving_average_valid_w3"
    }
    return beta, r2, diag

def beta_simple_slope(sigma_k, y_k, trim_tail=0.0):
    return beta_ols(sigma_k, y_k, trim_tail)