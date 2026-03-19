import numpy as np
from scipy.optimize import linprog

def _ensure_numpy(x):
    if x is None: return None
    if hasattr(x, "detach"): # Torch-like
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)

def _log_xy(sigma_k, p_tilde_k, trim_tail=0.0):
    """
    Log-transform x and y, optionally trim tails.
    """
    sigma_k   = _ensure_numpy(sigma_k)
    p_tilde_k = _ensure_numpy(p_tilde_k)

    mask = (sigma_k > 1e-12) & (p_tilde_k > 1e-12)
    log_x = np.log(p_tilde_k[mask])
    log_y = np.log(sigma_k[mask])
    
    if trim_tail > 0:
        n = len(log_x)
        t = int(n * trim_tail)
        if 2 * t < n - 2:
            return log_x[t:-t], log_y[t:-t]
    return log_x, log_y

# Helper removed: raw slope is now used as beta directly.

def _compute_r2(x, y, beta):
    slope     = beta
    intercept = np.mean(y) - slope * np.mean(x)
    y_pred    = slope * x + intercept
    ss_res    = np.sum((y - y_pred) ** 2)
    ss_tot    = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))

def _lad_solve(x, y):
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

def estimate_beta_ols(sigma_k, p_tilde_k, trim_tail=0.0):
    x, y  = _log_xy(sigma_k, p_tilde_k, trim_tail)
    if len(x) < 2: return 0.5, 0.0
    A     = np.column_stack([x, np.ones_like(x)])
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    beta  = float(max(0.0, coef[0]))
    return float(beta), _compute_r2(x, y, beta)

def beta_lad(sigma_k, p_tilde_k, trim_tail=0.0):
    x, y  = _log_xy(sigma_k, p_tilde_k, trim_tail)
    slope = _lad_solve(x, y)
    if slope is None:
        return 0.0, 0.0
    beta  = float(max(0.0, slope))
    return float(beta), _compute_r2(x, y, beta)



from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar

def estimate_beta_with_detrending(vals, is_svd=True, sigma_smooth=10):
    """
    가우시안 필터로 트렌드를 제거한 후, 잔차의 독립성을 최적화
    SVD 특이값인 경우 배율 1+β, 인기도인 경우 1+2β 사용 (Duality)
    """
    v = np.sort(np.abs(vals))[::-1]
    z = np.log(v + 1e-12)
    x = np.arange(len(z))

    def objective(beta):
        scaling = (1.0 + beta) if is_svd else (1.0 + 2.0 * beta)
        z_corr = z / scaling
        trend = gaussian_filter1d(z_corr, sigma=sigma_smooth)
        residual = z_corr - trend
        return np.abs(np.corrcoef(residual, x)[0, 1])

    res = minimize_scalar(objective, bounds=(0.0, 5.0), method='bounded')
    return float(res.x), 0.0

def estimate_beta_no_detrending(vals, is_svd=True):
    """
    데트렌딩 없이 전체 로그 스펙트럼의 선형성을 최적화
    """
    v = np.sort(np.abs(vals))[::-1]
    z = np.log(v + 1e-12)
    x = np.arange(len(z))

    def objective(beta):
        scaling = (1.0 + beta) if is_svd else (1.0 + 2.0 * beta)
        z_corr = z / scaling
        return np.abs(np.corrcoef(z_corr, x)[0, 1])

    res = minimize_scalar(objective, bounds=(0.0, 5.0), method='bounded')
    return float(res.x), 0.0

def estimate_beta_blue(sigma_obs, spectral_propensity, **kwargs):
    """
    Gauss-Markov 정리에 기반한 최적 선형 불편 추정량 (BLUE)
    로그 변환에 따른 이분산성(Heteroskedasticity)을 교정하기 위해
    델타 방법(Delta Method)에 의해 유도된 최적 가중치(sigma^2)를 사용하는
    가중 최소제곱법(WLS) 구현체. 
    """
    eps = 1e-12
    s_safe = np.maximum(_ensure_numpy(sigma_obs), eps)
    p_safe = np.maximum(_ensure_numpy(spectral_propensity), eps)
    
    Y = np.log(s_safe)
    X = np.log(p_safe)
    
def estimate_beta_max_median(popularity_counts):
    """
    Max-Median 기반 고속 Zipf 지수 추정
    """
    s = np.sort(popularity_counts)[::-1]
    n = len(s)
    max_p = s[0]
    # 유효한 중앙값 찾기 (0 제외)
    med_p = s[n // 2] if s[n // 2] > 0 else 1.0
    
    # Zipf 지수 zeta 추정
    zeta_hat = np.log(max_p / med_p) / np.log(n / 2)
    
    # ASPIRE Bridge Lemma: beta = zeta / (2 - zeta)
    beta_hat = zeta_hat / (2 - zeta_hat + 1e-9)
    beta_hat = max(0.0, beta_hat)
    return float(beta_hat), 0.0

def estimate_all(sigma_k, p_tilde_k, item_freq=None, n_items=None, n_users=None, trim_tail=0.0):
    res = {
        "ols": estimate_beta_ols(sigma_k, p_tilde_k, trim_tail),
        "lad": beta_lad(sigma_k, p_tilde_k, trim_tail),
        "iso_detrend": estimate_beta_with_detrending(sigma_k, is_svd=True),
        "iso_no_detrend": estimate_beta_no_detrending(sigma_k, is_svd=True),
    }
    if item_freq is not None:
        res["max_median"] = estimate_beta_max_median(item_freq)
        res["iso_pop_detrend"] = estimate_beta_with_detrending(item_freq, is_svd=False)
        res["iso_pop_no_detrend"] = estimate_beta_no_detrending(item_freq, is_svd=False)
    return res