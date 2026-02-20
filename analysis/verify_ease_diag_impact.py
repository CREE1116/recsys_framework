import torch
import numpy as np

def verify_diag_impact():
    print("=== Analyzing Impact of Diagonal Constraint (EASE vs Pure Dual) ===")
    
    # Setup
    N, M = 100, 200 # More items than users, typical for Dual advantage
    reg_lambda = 50.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available(): device = 'mps'
    
    print(f"Data: N={N}, M={M}, Lambda={reg_lambda}, Device={device}")
    X = torch.randn(N, M).to(device)
    
    # --- 1. Pure Dual Solution (Unconstrained Ridge Regression) ---
    # Objective: min ||X - XB||^2 + lambda ||B||^2 (No diag constraint)
    # Solution: B = (X'X + lambda I)^-1 X'X = X' (XX' + lambda I)^-1 X
    print("\n--- 1. Pure Dual Solution (No Diag Constraint) ---")
    
    # Compute using Dual path for speed
    K = torch.mm(X, X.t())
    K_reg = K + reg_lambda * torch.eye(N).to(device)
    C = torch.linalg.inv(K_reg)
    
    # S_pure = X^T @ C @ X
    S_pure = torch.mm(torch.mm(X.t(), C), X)
    
    diag_vals = torch.diag(S_pure)
    print(f"Diagonal Mean: {diag_vals.mean().item():.4f}")
    print(f"Diagonal Max:  {diag_vals.max().item():.4f}")
    print(f"Diagonal Min:  {diag_vals.min().item():.4f}")
    print("-> Pure Dual solution has non-zero diagonal (Self-Reconstruction allowed)")
    
    # --- 2. Standard EASE (Primal with Constraint) ---
    # Objective: min ||X - XB||^2 + lambda ||B||^2 s.t. diag(B) = 0
    # Solution derived from Lagrangians results in: B_ij = -P_ij / P_jj
    print("\n--- 2. Standard EASE (Diag Constraint = 0) ---")
    
    # Calculates P = (X'X + lambda I)^-1
    # We can use Woodbury to get P from C: P = (1/lambda) * (I - X'CX)
    # This is numerically identical to Primal inversion.
    Term = torch.mm(torch.mm(X.t(), C), X)
    P = (torch.eye(M).to(device) - Term) / reg_lambda
    
    # Apply Constraint
    diag_P = torch.diag(P)
    S_ease = P / diag_P.unsqueeze(0)
    S_ease = -S_ease
    S_ease.fill_diagonal_(0.0)
    
    print(f"Diagonal Mean: {torch.diag(S_ease).mean().item():.4f}")
    
    # --- 3. Comparison ---
    print("\n--- Comparison (Pure vs EASE) ---")
    diff = torch.abs(S_pure - S_ease)
    
    # Off-diagonal difference
    mask = ~torch.eye(M, dtype=torch.bool).to(device)
    off_diag_diff = diff[mask]
    
    print(f"Total Matrix Diff (Norm): {torch.norm(diff).item():.4f}")
    print(f"Off-Diagonal Mean Diff:   {off_diag_diff.mean().item():.4f}")
    print(f"Off-Diagonal Max Diff:    {off_diag_diff.max().item():.4f}")
    
    # Interpretation
    print("\n--- Conclusion ---")
    if off_diag_diff.mean().item() > 0.01:
        print("SIGNIFICANT DIFFERENCE in learned weights.")
        print("Forcing diagonal=0 fundamentally changes the optimal weights for other items.")
        print("Pure Dual (LiRA without mask) != EASE")
    else:
        print("Weights are relatively similar, but not identical.")

if __name__ == "__main__":
    verify_diag_impact()
