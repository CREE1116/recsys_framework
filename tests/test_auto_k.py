import torch
import numpy as np
from scipy.sparse import csr_matrix
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.models.csar.LIRALayer import LightLIRALayer

def test_auto_k():
    print("Testing Auto-K Selection...")
    
    # 1. Create dummy data (User x Item)
    n_users = 2000
    n_items = 5000
    U = np.random.randn(n_users, 50)
    V = np.random.randn(n_items, 50)
    X_dense = U @ V.T + 0.1 * np.random.randn(n_users, n_items)
    X_sparse = csr_matrix(X_dense)
    
    reg_lambda = 10.0

    # Test Strategy 'b'
    print("\n--- Testing Strategy 'b' (Cache Write) ---")
    layer_b = LightLIRALayer(k='b', reg_lambda=reg_lambda, normalize=True)
    layer_b.build(X_sparse, dataset_name="test_dummy")
    k_b = getattr(layer_b, 'k_final_val', layer_b.k_final.item())
    print(f"Selected K (b): {k_b}")
    assert k_b > 0

    print("\n--- Testing Strategy 'b' (Cache Read) ---")
    layer_b2 = LightLIRALayer(k='b', reg_lambda=reg_lambda, normalize=True)
    layer_b2.build(X_sparse, dataset_name="test_dummy")
    k_b2 = getattr(layer_b2, 'k_final_val', layer_b2.k_final.item())
    assert k_b2 == k_b

    # Test 'auto' (should default to 'b')
    print("\n--- Testing 'auto' (default b) ---")
    layer_auto = LightLIRALayer(k='auto', reg_lambda=reg_lambda, normalize=True)
    layer_auto.build(X_sparse)
    k_auto = getattr(layer_auto, 'k_final_val', layer_auto.k_final.item())
    print(f"Selected K (auto): {k_auto}")
    assert k_auto == k_b

    # Test Visualization
    print("\n--- Testing Visualization (Detailed) ---")
    test_viz_dir = "tests/test_analysis"
    layer_auto.visualize_matrices(X_sparse=X_sparse, save_dir=test_viz_dir, lightweight=False)
    
    files_to_check = [
        "spectral_cut_analysis.png",
        "analysis_summary.json",
        "viz_item_embeddings.png",
        "viz_S_raw.png",
        "viz_S_filtered.png",
        "viz_C_matrix.png"
    ]
    for f in files_to_check:
        path = os.path.join(test_viz_dir, f)
        exists = os.path.exists(path)
        print(f"Checking {f}: {'EXISTS' if exists else 'MISSING'}")
        assert exists
    
    print("\nAll Auto-K strategy and visualization tests PASSED!")

if __name__ == "__main__":
    test_auto_k()
