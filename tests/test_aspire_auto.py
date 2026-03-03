import torch
import numpy as np
import scipy.sparse as sp
from src.models.csar.LIRALayer import ASPIRELayer, ChebyASPIRELayer, estimate_mnar_gamma

def generate_power_law_data(n_users, n_items, gamma):
    """
    Generate synthetic data where item popularity follows power law with index p.
    p = (1 + gamma) / 2 -> gamma = 2p - 1
    """
    p = (1.0 + gamma) / 2.0
    item_probs = np.arange(1, n_items + 1) ** (-p)
    item_probs /= item_probs.sum()
    
    rows, cols = [], []
    nnz_per_user = 50
    for u in range(n_users):
        selected_items = np.random.choice(n_items, nnz_per_user, p=item_probs, replace=False)
        rows.extend([u] * nnz_per_user)
        cols.extend(selected_items)
        
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_users, n_items))

def test_gamma_estimation():
    print("\n[Test] Gamma Estimation Accuracy")
    true_gammas = [0.5, 1.0, 1.5]
    for tg in true_gammas:
        X = generate_power_law_data(1000, 2000, tg)
        est_g = estimate_mnar_gamma(X_sparse=X)
        print(f"True Gamma: {tg:.2f}, Estimated: {est_g:.4f}")

def test_aspire_auto_beta():
    print("\n[Test] ASPIRE Layer Auto-Beta")
    X = generate_power_law_data(1000, 2000, 1.0) # gamma=1 -> beta=0.5
    
    # Test ASPIRE (Spectral)
    layer = ASPIRELayer(alpha=500.0, beta='auto_compromise', target_energy=0.9)
    layer.build(X)
    print(f"ASPIRE beta (Compromise): {layer.beta:.4f} (Expected ~0.5)")
    
    # Test ChebyASPIRE (Count-based)
    cheby = ChebyASPIRELayer(alpha=500.0, beta='auto_bias')
    cheby.build(X)
    print(f"Cheby beta (Bias): {cheby.beta:.4f} (Expected ~0.5 for gamma=1)")

if __name__ == "__main__":
    test_gamma_estimation()
    test_aspire_auto_beta()
