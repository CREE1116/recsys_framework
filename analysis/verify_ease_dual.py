import torch
import numpy as np
import time

def verify_ease_dual():
    print("=== Verifying EASE Primal vs Dual Forms (Woodbury Identity) ===")
    
    # 1. Setup Data
    # Use dimensions where Dual is faster (N < M) to show utility
    # Or small enough to inspect (N=100, M=200)
    N, M = 100, 200
    reg_lambda = 50.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available(): device = 'mps'
    
    print(f"Data: N={N}, M={M}, Lambda={reg_lambda}, Device={device}")
    
    X = torch.randn(N, M).to(device) # Dense for simplicity
    
    # --- 2. Primal EASE (Standard) ---
    # Complexity: O(M^3) -> 200^3 = 8,000,000 ops
    t0 = time.time()
    G = torch.mm(X.t(), X) # Gram Matrix M x M
    G += reg_lambda * torch.eye(M).to(device)
    P = torch.linalg.inv(G) # The inverse matrix
    
    # B = I - P @ diag(1/diag(P))
    # Effectively: B_ij = - P_ij / P_jj if i != j else 0
    diag_P = torch.diag(P)
    B_primal = P / diag_P.unsqueeze(0) # Divide columns by diag
    B_primal = -B_primal
    B_primal.fill_diagonal_(0.0)
    
    t_primal = time.time() - t0
    print(f"Primal Time: {t_primal:.6f}s")
    
    # --- 3. Dual EASE (Woodbury) ---
    # Complexity: O(N^3) + O(N^2 M) -> 100^3 + 100^2*200 = 1,000,000 + 2,000,000 = 3M ops (Faster!)
    # Formula: (X'X + \lambda I)^-1 = 1/\lambda * (I - X' (XX' + \lambda I)^-1 X)
    t0 = time.time()
    
    K = torch.mm(X, X.t()) # Dual Gram N x N
    K += reg_lambda * torch.eye(N).to(device)
    C = torch.linalg.inv(K) # Inverse N x N
    
    # Compute P_dual using Woodbury
    # Term = X.T @ C @ X  (M x N) @ (N x M) -> (M x M)
    # Caution: This step (X.T @ C @ X) is O(M^2 N), which might be slow if M is huge.
    # But we only need this if we want the full B matrix. 
    # For prediction we might optimize, but for now let's verify B equivalence.
    
    Term = torch.mm(torch.mm(X.t(), C), X)
    P_dual = (torch.eye(M).to(device) - Term) / reg_lambda
    
    # Calculate B from P_dual
    diag_P_dual = torch.diag(P_dual)
    B_dual = P_dual / diag_P_dual.unsqueeze(0)
    B_dual = -B_dual
    B_dual.fill_diagonal_(0.0)
    
    t_dual = time.time() - t0
    print(f"Dual Time: {t_dual:.6f}s")
    
    # --- 4. Comparison ---
    diff = torch.abs(B_primal - B_dual)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\nMax Difference: {max_diff:.8f}")
    print(f"Mean Difference: {mean_diff:.8f}")
    
    if max_diff < 1e-4:
        print("✅ SUCCESS: Primal and Dual EASE are equivalent!")
    else:
        print("❌ FAILURE: Divergence detected.")

if __name__ == "__main__":
    verify_ease_dual()
