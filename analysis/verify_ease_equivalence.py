import torch
import numpy as np

def verify_ease_methods():
    print("=== FINAL VERIFICATION: EASE Primal vs Dual (Woodbury) vs Naive Zeroing ===")
    
    # 1. Setup
    N, M = 100, 200
    reg_lambda = 50.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available(): device = 'mps'
    
    print(f"Data: N={N}, M={M}, Lambda={reg_lambda}")
    X = torch.randn(N, M).to(device)
    
    # --- Method A: Primal EASE (Standard) ---
    # 1. Compute Gram G = X'X
    # 2. Invert P = (G + lambda I)^-1
    # 3. B = I - P @ diag(1/diag(P))
    G = torch.mm(X.t(), X)
    G += reg_lambda * torch.eye(M).to(device)
    P_primal = torch.linalg.inv(G)
    
    diag_P = torch.diag(P_primal)
    B_primal = P_primal / diag_P.unsqueeze(0)
    B_primal = -B_primal
    B_primal.fill_diagonal_(0.0)
    
    print("Method A (Primal EASE) Computed.")

    # --- Method B: Dual EASE (Woodbury Trick) ---
    # 1. Compute Dual Gram K = XX'
    # 2. Invert C = (K + lambda I)^-1
    # 3. Compute P using Woodbury: P = 1/lambda * (I - X' C X)
    # 4. B = I - P @ diag(1/diag(P))   <-- SAME constraint logic
    K = torch.mm(X, X.t())
    K += reg_lambda * torch.eye(N).to(device)
    C = torch.linalg.inv(K)
    
    # Woodbury to get P
    Term = torch.mm(torch.mm(X.t(), C), X)
    P_dual_derived = (torch.eye(M).to(device) - Term) / reg_lambda
    
    diag_P_dual = torch.diag(P_dual_derived)
    B_dual_woodbury = P_dual_derived / diag_P_dual.unsqueeze(0)
    B_dual_woodbury = -B_dual_woodbury
    B_dual_woodbury.fill_diagonal_(0.0)
    
    print("Method B (Dual Woodbury) Computed.")
    
    # --- Method C: Naive Dual Zeroing ---
    # 1. Solve Unconstrained Dual S = X' C X
    # 2. Just set diagonal to 0
    S_naive = torch.mm(torch.mm(X.t(), C), X)
    B_naive = S_naive.clone()
    B_naive.fill_diagonal_(0.0)
    
    print("Method C (Naive Dual Zeroing) Computed.")

    # --- Comparisons ---
    diff_AB = torch.abs(B_primal - B_dual_woodbury).max().item()
    diff_AC = torch.abs(B_primal - B_naive).max().item()
    
    print("\n-------------------------------------------------------------")
    print(f"Diff (Primal vs Dual Woodbury): {diff_AB:.8f}")
    if diff_AB < 1e-5:
        print(">> ✅ IDENTICAL. The calculation path does not change the result.")
    else:
        print(">> ❌ DIFFERENT.")
        
    print(f"Diff (Primal vs Naive Zeroing): {diff_AC:.8f}")
    if diff_AC < 1e-5:
        print(">> ✅ IDENTICAL.")
    else:
        print(">> ❌ DIFFERENT. Naively zeroing diagonal is NOT EASE.")
    print("-------------------------------------------------------------")

if __name__ == "__main__":
    verify_ease_methods()
