import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.models.csar.beta_estimators import beta_log_derivative

def test_beta_log_derivative():
    print("Testing beta_log_derivative with synthetic data...")
    
    # True Beta 0.7 => slope (gamma) = 1.4 / 1.7 = 0.8235
    true_beta = 0.7
    true_slope = 2.0 * true_beta / (1.0 + true_beta)
    
    # Generate sigma_k behavior (~ k^-0.5)
    indices = np.arange(1, 401)
    sigma_k = 10 * (indices ** -0.5)
    # p_k = C * sigma_k^true_slope
    p_k = 1.0 * (sigma_k ** true_slope)
    
    p_k_noisy = p_k 
    
    # 1. Test median (q=0.5)
    beta_est_05, r2, diag_05 = beta_log_derivative(sigma_k, p_k_noisy, q=0.5)
    print(f"True Beta: {true_beta:.4f}")
    print(f"Est Beta (q=0.5): {beta_est_05:.4f}, R2: {r2:.4f}")
    print(f"Diag: {diag_05}")
    
    # 2. Test other quantiles
    beta_est_02, _, _ = beta_log_derivative(sigma_k, p_k_noisy, q=0.2)
    print(f"Est Beta (q=0.2): {beta_est_02:.4f}")
    beta_est_08, _, _ = beta_log_derivative(sigma_k, p_k_noisy, q=0.8)
    print(f"Est Beta (q=0.8): {beta_est_08:.4f}")
    
    assert abs(beta_est_05 - true_beta) < 0.05, f"Beta estimation failed: {beta_est_05} vs {true_beta}"
    print("Verification successful!")

if __name__ == "__main__":
    test_beta_log_derivative()
