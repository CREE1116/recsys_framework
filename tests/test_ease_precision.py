import torch
import numpy as np
import scipy.sparse as sp
from src.utils.gpu_accel import gpu_gram_solve

def test_gram_precision():
    # Large enough to bypass some paths, small enough to verify
    M = 1000
    N = 500
    X_np = np.random.binomial(1, 0.1, (N, M)).astype(np.float32)
    X_sparse = sp.csr_matrix(X_np)
    reg_lambda = 100.0
    
    # 1. Direct Solve (NumPy)
    G = (X_np.T @ X_np) + reg_lambda * np.eye(M)
    P_exact = np.linalg.inv(G)
    
    # 2. gpu_gram_solve (Eigen path)
    P_gpu = gpu_gram_solve(X_sparse, reg_lambda, device='cpu', return_tensor=False)
    
    # Measure error
    err = np.abs(P_exact - P_gpu).max()
    print(f"Max Absolute Error (Eigen path): {err:.2e}")
    
    # Check Symmetry
    sym_err = np.abs(P_gpu - P_gpu.T).max()
    print(f"Symmetry Error: {sym_err:.2e}")

if __name__ == "__main__":
    test_gram_precision()
