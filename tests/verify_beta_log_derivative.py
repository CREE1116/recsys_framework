import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.models.csar.beta_estimators import beta_log_derivative

def test_beta_log_derivative():
    print("Testing beta_log_derivative with synthetic data...")
    
    # Generate synthetic power-law data
    # sigma_k ~ k^-alpha_s
    # p_k ~ sigma_k^slope
    # v3 theory: slope = 1 + beta
    
    k = np.arange(1, 201)
    alpha_s = 0.5
    true_beta = 0.7
    # v3: slope = 1 + beta
    true_slope = 1.0 + true_beta
    
    sigma_k = k**(-alpha_s)
    # p_k = C * sigma_k^true_slope
    p_k = 1.0 * (sigma_k**true_slope)
    
    # Clean data for verification of pure finite difference limit
    p_k_noisy = p_k 
    
    # 1. Test median (q=0.5)
    beta_est_05, r2, diag_05 = beta_log_derivative(sigma_k, p_k_noisy, q=0.5, version='v3')
    print(f"True Beta: {true_beta:.4f}")
    print(f"Est Beta (q=0.5): {beta_est_05:.4f}, R2: {r2:.4f}")
    print(f"Diag: {diag_05}")
    
    # 2. Test quantile range (should be stable for this clean data)
    beta_est_02, _, _ = beta_log_derivative(sigma_k, p_k_noisy, q=0.2, version='v3')
    beta_est_08, _, _ = beta_log_derivative(sigma_k, p_k_noisy, q=0.8, version='v3')
    print(f"Est Beta (q=0.2): {beta_est_02:.4f}")
    print(f"Est Beta (q=0.8): {beta_est_08:.4f}")
    
    assert abs(beta_est_05 - true_beta) < 0.05, f"Beta estimation failed: {beta_est_05} vs {true_beta}"
    print("Verification successful!")

if __name__ == "__main__":
    test_beta_log_derivative()
