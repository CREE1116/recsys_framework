import numpy as np
from scipy.optimize import linprog

def _log_xy(sigma_k, p_tilde_k, trim_tail=0.05):
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

def _slope_to_beta(slope):
    if slope <= 0:
        return 0.0
    if slope >= 1.818: # β >= 10.0
        return 10.0
    return slope / (2.0 - slope)

def _compute_r2(x, y, beta):
    slope     = 2.0 * beta / (1.0 + beta)
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

def beta_ols(sigma_k, p_tilde_k, trim_tail=0.05):
    x, y  = _log_xy(sigma_k, p_tilde_k, trim_tail)
    A     = np.column_stack([x, np.ones_like(x)])
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    beta  = _slope_to_beta(coef[0])
    return float(beta), _compute_r2(x, y, beta)

def beta_lad(sigma_k, p_tilde_k, trim_tail=0.05):
    x, y  = _log_xy(sigma_k, p_tilde_k, trim_tail)
    slope = _lad_solve(x, y)
    if slope is None:
        return 0.0, 0.0
    beta  = _slope_to_beta(slope)
    return float(beta), _compute_r2(x, y, beta)

def smooth_estimate_vector_opt(sigma_obs, p_tilde, lambda_smooth=0.1):
    """
    vector_opt 성공 버전에 수치적 안정성과 Smoothness를 더한 최종 엔진
    """
    try:
        import torch
        if torch.is_tensor(sigma_obs):   sigma_obs = sigma_obs.detach().cpu().numpy()
        if torch.is_tensor(p_tilde):     p_tilde = p_tilde.detach().cpu().numpy()
    except Exception:
        pass

    log_s = np.log(sigma_obs + 1e-12)
    log_p = np.log(p_tilde + 1e-12)
    
    # 1. 초기값 설정 (스칼라 Beta에서 출발)
    try:
        initial_beta_val = np.abs(np.corrcoef(log_s, log_p)[0, 1]) 
    except Exception:
        initial_beta_val = 0.5

    initial_beta = np.ones_like(log_s) * initial_beta_val

    # 2. 목적 함수: Beta_k들의 변동성 최소화 + 전역 평균 유지
    def objective(beta_vec):
        term_variance = np.sum((beta_vec - np.mean(beta_vec))**2)
        term_smooth = lambda_smooth * np.sum(np.diff(beta_vec)**2)
        return term_variance + term_smooth

    # 3. 제약 조건: 전역적 직교성
    def constraint_orthogonality(beta_vec):
        corrected_signal = log_s - beta_vec * log_p
        return np.corrcoef(corrected_signal, log_p)[0, 1]

    cons = {'type': 'eq', 'fun': constraint_orthogonality}
    
    from scipy.optimize import minimize
    res = minimize(objective, initial_beta, 
                   constraints=cons, 
                   method='SLSQP', 
                   options={'maxiter': 1000, 'ftol': 1e-9})

    vector_beta = res.x if res.success else initial_beta
    mean_beta = np.mean(vector_beta)
    return vector_beta, _compute_r2(log_s, log_p, mean_beta), {"mean_beta": float(mean_beta)}

def estimate_vector_beta(sigma_obs, p_tilde):
    # Legacy / Simple version without smoothness penalty
    return smooth_estimate_vector_opt(sigma_obs, p_tilde, lambda_smooth=0.0)

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

def estimate_beta_decoupling(sigma_obs, spectral_propensity):
    """
    보정된 신호와 인기도 사이의 상관관계 최소화
    """
    log_s = np.log(sigma_obs + 1e-12)
    log_p = np.log(spectral_propensity + 1e-12)

    def objective(beta):
        # log_s_true = log_s_obs - beta * log_p
        corrected_signal = log_s - beta * log_p
        return np.abs(np.corrcoef(corrected_signal, log_p)[0, 1])

    res = minimize_scalar(objective, bounds=(0.0, 5.0), method='bounded')
    return float(res.x), _compute_r2(log_s, log_p, float(res.x))

def estimate_all(sigma_k, p_tilde_k, item_freq=None, n_items=None, n_users=None, trim_tail=0.05):
    res = {
        "ols": beta_ols(sigma_k, p_tilde_k, trim_tail),
        "lad": beta_lad(sigma_k, p_tilde_k, trim_tail),
        "vector_opt": estimate_vector_beta(sigma_k, p_tilde_k),
        "smooth_vector": smooth_estimate_vector_opt(sigma_k, p_tilde_k),
        "iso_detrend": estimate_beta_with_detrending(sigma_k, is_svd=True),
        "iso_no_detrend": estimate_beta_no_detrending(sigma_k, is_svd=True),
        "decoupling": estimate_beta_decoupling(sigma_k, p_tilde_k),
    }
    if item_freq is not None:
        res["max_median"] = estimate_beta_max_median(item_freq)
        res["iso_pop_detrend"] = estimate_beta_with_detrending(item_freq, is_svd=False)
        res["iso_pop_no_detrend"] = estimate_beta_no_detrending(item_freq, is_svd=False)
    return res


    