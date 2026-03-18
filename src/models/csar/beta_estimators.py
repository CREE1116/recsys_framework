import numpy as np
from scipy.optimize import linprog

def _log_xy(sigma_k, p_tilde_k, trim_tail=0.0):
    """
    log 변환 및 유효 데이터 필터링
    trim_tail: 하위 tail 비율 제거
    """
    try:
        import torch
        if torch.is_tensor(sigma_k):   sigma_k   = sigma_k.detach().cpu().numpy()
        if torch.is_tensor(p_tilde_k): p_tilde_k = p_tilde_k.detach().cpu().numpy()
    except ImportError:
        pass

    sigma_k   = np.asarray(sigma_k,   dtype=float)
    p_tilde_k = np.asarray(p_tilde_k, dtype=float)

    mask = (sigma_k > 1e-12) & (p_tilde_k > 1e-12)
    x = np.log(sigma_k[mask])
    y = np.log(p_tilde_k[mask])

    if len(x) < 2:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])

    if trim_tail > 0:
        n = len(x)
        end_idx = n - int(n * trim_tail)
        end_idx = max(end_idx, 2)
        x = x[:end_idx]
        y = y[:end_idx]

    return x, y

def _slope_to_beta_v2(slope):
    """
    v2 Mapping: Based on SPP (p_tilde) vs sigma.
    slope = 2*beta / (1 + beta)  =>  beta = slope / (2 - slope)
    """
    if slope <= 0: return 0.0
    denom = 2.0 - slope
    if denom <= 1e-9:
        return 10.0
    return np.clip(slope / denom, 0.0, 10.0)

def _slope_to_beta_v3(slope):
    """
    v3 Mapping: Based on Item Frequency (n_k) vs sigma.
    slope = 1 + beta  =>  beta = slope - 1
    """
    return np.clip(slope - 1.0, 0.0, 10.0)

def _compute_r2(x, y, beta, version='v2'):
    """R² calculation based on the assumed theory version."""
    if version == 'v2':
        slope = 2.0 * beta / (1.0 + beta + 1e-12)
    else: # v3
        slope = 1.0 + beta
        
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

def beta_ols(sigma_k, p_tilde_k, trim_tail=0.0):
    """OLS Estimator"""
    x, y  = _log_xy(sigma_k, p_tilde_k, trim_tail)
    A     = np.column_stack([x, np.ones_like(x)])
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    beta  = _slope_to_beta_v2(coef[0])
    return float(beta), _compute_r2(x, y, beta, version='v2')

def beta_lad(sigma_k, p_tilde_k, trim_tail=0.0):
    """LAD Estimator"""
    x, y  = _log_xy(sigma_k, p_tilde_k, trim_tail)
    slope = _lad_solve(x, y)
    if slope is None:
        return 0.0, 0.0
    beta  = _slope_to_beta_v2(slope)
    return float(beta), _compute_r2(x, y, beta, version='v2')

def beta_rank_index(sigma_k, item_freq, top_k=1000, skip_head=5):
    """
    Rank-Index 기반 Beta 추정
    β = η / (2α_s - η)
    """
    try:
        import torch
        if torch.is_tensor(sigma_k):   sigma_k = sigma_k.detach().cpu().numpy()
        if torch.is_tensor(item_freq): item_freq = item_freq.detach().cpu().numpy()
    except ImportError:
        pass

    def _estimate_power_law(vals, max_k, skip):
        v = np.sort(np.abs(vals))[::-1]
        v_nz = v[v > 1e-12]
        if len(v_nz) <= skip + 5: return 0.0, 0.0, 0
        
        n = min(len(v_nz), max_k)
        indices = np.arange(skip + 1, n + 1)
        log_k = np.log(indices)
        log_v = np.log(v_nz[skip:n])
        
        A = np.column_stack([log_k, np.ones_like(log_k)])
        c, _, _, _ = np.linalg.lstsq(A, log_v, rcond=None)
        
        y_pred = c[0] * log_k + c[1]
        ss_res = np.sum((log_v - y_pred) ** 2)
        ss_tot = np.sum((log_v - np.mean(log_v)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        
        return -float(c[0]), float(r2), n - skip

    alpha_s, r2_s, n_s = _estimate_power_law(sigma_k, top_k, skip_head)
    eta, r2_f, n_f = _estimate_power_law(item_freq, top_k, skip_head)

    if alpha_s <= 0.05 or eta <= 0.0:
        return 0.0, 0.0, {"alpha_s": alpha_s, "eta": eta, "error": "too_small"}
        
    denom = 2.0 * alpha_s - eta
    if denom <= 1e-3: 
        beta = 10.0
    else:
        beta = eta / denom
    
    beta = max(0.0, min(10.0, beta))
    
    diag = {
        "alpha_s": alpha_s, 
        "eta": eta, 
        "r2_s": r2_s, 
        "r2_f": r2_f,
        "n_samples": n_s,
        "denom": denom
    }
    return float(beta), float(r2_s), diag

def beta_log_derivative(sigma_k, p_tilde_k, q=0.5, version='v2', trim_tail=0.0, **kwargs):
    """
    Pure Finite Difference 기반 Beta 추정.
    PCHIP 보간이나 Smoothing 없이, 인접한 두 점 사이의 기울기를 직접 계산함.
    
    q: Quantile level (0.5 for median)
    """
    try:
        import torch
        if torch.is_tensor(sigma_k):   sigma_k = sigma_k.detach().cpu().numpy()
        if torch.is_tensor(p_tilde_k): p_tilde_k = p_tilde_k.detach().cpu().numpy()
    except ImportError:
        pass

    s = np.asarray(sigma_k, dtype=float).flatten()
    pt = np.asarray(p_tilde_k, dtype=float).flatten()
    
    # 1. 정렬 및 유효 데이터 필터링
    idx = np.argsort(s)[::-1]
    s_sorted = s[idx]
    pt_aligned = pt[idx]
    
    # Noise floor filtering
    mask = (s_sorted > 1e-12) & (pt_aligned > 1e-12)
    if mask.sum() < 5:
        return 0.5, 0.0, {"beta": 0.5, "error": "too_few_samples"}
        
    s_clean = s_sorted[mask]
    p_clean = pt_aligned[mask]

    if trim_tail > 0:
        n = len(s_clean)
        keep = max(int(n * (1.0 - trim_tail)), 5)
        s_clean = s_clean[:keep]
        p_clean = p_clean[:keep]

    log_s = np.log(s_clean)
    log_p = np.log(p_clean)

    # 2. 순수 차분
    d_log_s = np.diff(log_s)
    d_log_p = np.diff(log_p)
    slopes = d_log_p / (d_log_s - 1e-15)
    
    # 4. Aggregation (Pure Quantile)
    valid_mask = (slopes > 0.0) 
    if version == 'v2':
        valid_mask &= (slopes < 2.5)
        
    v_slopes = slopes[valid_mask]
    if len(v_slopes) == 0:
        return 0.5, 0.0, {"error": "No valid slopes"}
        
    final_slope = np.nanquantile(v_slopes, q)

    # 5. Mapping
    if version == 'v2':
        beta = _slope_to_beta_v2(final_slope)
    elif version == 'v3':
        beta = _slope_to_beta_v3(final_slope)
    elif version == 'identity':
        beta = float(final_slope)
    else:
        beta = _slope_to_beta_v2(final_slope)

    diag = {
        "slope": float(final_slope),
        "beta": float(beta),
        "n_points": len(s_clean),
        "n_slopes": len(v_slopes),
        "version": version,
        "method": "pure_finite_diff",
        "q": float(q)
    }
    
    r2 = _compute_r2(log_s, log_p, beta, version=version)
    return float(beta), float(r2), diag

def beta_simple_slope(sigma_k, y_k, trim_tail=0.0, version='v2'):
    """
    Fits a simple global OLS slope and returns it AS beta (Identity Mapping).
    """
    s = np.asarray(sigma_k, dtype=float).flatten()
    y = np.asarray(y_k, dtype=float).flatten()
    log_s = np.log(s + 1e-12)
    log_y = np.log(y + 1e-12)
    
    if len(log_s) < 2: return 0.0, 0.0
    
    A = np.column_stack([log_s, np.ones_like(log_s)])
    coef, _, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
    beta = float(coef[0])
    
    y_pred = beta * log_s + (np.mean(log_y) - beta * np.mean(log_s))
    res = np.sum((log_y - y_pred)**2)
    tot = np.sum((log_y - np.mean(log_y))**2)
    r2 = 1.0 - res / (tot + 1e-12)
    
    return float(beta), float(r2)