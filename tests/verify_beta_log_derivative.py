import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.models.csar.beta_estimators import beta_log_derivative

def test_sweep():
    windows = [1, 3, 5]
    noises = [0.0, 0.02, 0.05]
    
    print(f"{'Method':<10} | {'Window':<8} | {'Noise':<8} | {'Est Beta':<10} | {'R2':<8} | {'Status'}")
    print("-" * 75)
    
    true_beta = 0.7
    true_slope = 2.0 * true_beta / (1.0 + true_beta)
    
    for noise in noises:
        for window in windows:
            # Generate data
            np.random.seed(42)
            indices = np.arange(1, 401)
            sigma_k = 10 * (indices ** -0.5)
            p_k = 1.0 * (sigma_k ** true_slope)
            if noise > 0:
                p_k = p_k * np.exp(np.random.normal(0, noise, size=p_k.shape))
            
            # Test log_derivative
            try:
                beta_est, r2, _ = beta_log_derivative(sigma_k, p_k, q=0.5, smooth_window=window)
                diff = abs(beta_est - true_beta)
                status = "OK" if diff < 0.15 else "BIASED"
                print(f"{'LogDeriv':<10} | {window:<8} | {noise:<8.3f} | {beta_est:<10.4f} | {r2:<8.3f} | {status}")
            except Exception as e:
                print(f"LogDeriv  | {window:<8} | {noise:<8.3f} | ERROR: {e}")
                
        # Test dynamic_derivative (Window independent)
        try:
            from src.models.csar.beta_estimators import beta_dynamic_derivative
            beta_est, r2, _ = beta_dynamic_derivative(sigma_k, p_k, q=0.5)
            diff = abs(beta_est - true_beta)
            status = "OK" if diff < 0.15 else "BIASED"
            print(f"{'Dynamic':<10} | {'N/A':<8} | {noise:<8.3f} | {beta_est:<10.4f} | {r2:<8.3f} | {status}")
        except Exception as e:
            print(f"Dynamic   | {'N/A':<8} | {noise:<8.3f} | ERROR: {e}")

if __name__ == "__main__":
    test_sweep()
