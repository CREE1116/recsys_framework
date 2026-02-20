import torch

def test_mps_ops():
    if not torch.backends.mps.is_available():
        print("MPS not available")
        return

    device = torch.device("mps")
    print(f"Testing on {device}...")
    
    # Create test matrix
    A = torch.randn(100, 50, device=device)
    
    # Test QR
    try:
        Q, R = torch.linalg.qr(A)
        print("QR: SUCCESS")
    except Exception as e:
        print(f"QR: FAILED - {e}")

    # Test Cholesky on Gram matrix
    G = A.t() @ A
    try:
        L = torch.linalg.cholesky(G)
        print("Cholesky: SUCCESS")
        
        # Test solve_triangular
        try:
            B_solve = torch.randn(50, 100, device=device)
            X_solve = torch.linalg.solve_triangular(L, B_solve, upper=False)
            print("SolveTriangular: SUCCESS")
        except Exception as e:
            print(f"SolveTriangular: FAILED - {e}")
            
    except Exception as e:
        print(f"Cholesky: FAILED - {e}")

    # Test EIGH on Gram matrix
    try:
        Lval, Lvec = torch.linalg.eigh(G)
        print("EIGH: SUCCESS")
    except Exception as e:
        print(f"EIGH: FAILED - {e}")

    # Test SVD
    try:
        U, S, V = torch.linalg.svd(A, full_matrices=False)
        print("SVD: SUCCESS")
    except Exception as e:
        print(f"SVD: FAILED - {e}")

if __name__ == "__main__":
    test_mps_ops()
