
import torch
import numpy as np
import scipy.sparse as sp
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.csar.LIRA import LIRA
from src.models.csar.LIRALayer import LIRALayer

class MockDataLoader:
    def __init__(self, n_users, n_items):
        self.n_users = n_users
        self.n_items = n_items
        # Create random interaction data
        rows = np.random.randint(0, n_users, 1000)
        cols = np.random.randint(0, n_items, 1000)
        
        import pandas as pd
        self.train_df = pd.DataFrame({'user_id': rows, 'item_id': cols})

def test_lyra_dual_ridge():
    print("Testing LYRA Dual Ridge (VCV) Mode...")
    
    n_users = 100
    n_items = 50
    data_loader = MockDataLoader(n_users, n_items)
    
    config = {
        'model': {
            'name': 'lira',
            'reg_lambda': 100.0,
            'k_dim': 300, # Should act as Full Rank
            'alpha': 50.0
        },
        'device': 'cpu'
    }
    
    # Mock train_matrix_csr
    rows = data_loader.train_df['user_id'].values
    cols = data_loader.train_df['item_id'].values
    vals = np.ones(len(rows))
    train_matrix_csr = sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
    data_loader.train_matrix_csr = train_matrix_csr
    
    # 1. Initialize
    print("Initializing LIRA...")
    model = LIRA(config, data_loader)
    
    # 1. Test Fit
    print("Running fit()...")
    # LIRA.fit() calls lira_layer.build()
    # But usually fit() is called by the framework. Let's call build manually if needed or just fit.
    # LIRA constructor calls build() already? keeping consistency with LIRA.py
    # LIRA.py: self.lira_layer.build(self.get_train_matrix(data_loader)) in __init__
    
    S = model.lira_layer.S
    print(f"S shape: {S.shape}")
    assert S.shape == (n_items, n_items)
    
    # 3. Symmetry Check
    print("Checking Symmetry of S...")
    assert np.allclose(S.detach().cpu().numpy(), S.detach().cpu().numpy().T, atol=1e-5)
    
    assert hasattr(model.lira_layer, 'C'), "C matrix should be stored for visualization"
    C = model.lira_layer.C
    
    # 3. Test Normalization (Default True)
    print("Testing Normalization (Default True)...")
    assert model.lira_layer.normalize == True, "Default normalization should be True"
    
    # 4. Test Visualization
    print("Testing Visualization...")
    output_dir = os.path.join(os.path.dirname(__file__), '../output/test_viz')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'lira_viz_test.png')
    print("[LIRA] Visualization saved to", save_path)
    model.lira_layer.visualize_matrices(X_sparse=train_matrix_csr, save_dir=os.path.dirname(save_path))
    # Check for one of the generated files
    assert os.path.exists(os.path.join(os.path.dirname(save_path), 'viz_S_Norm.png')), "Visualization file viz_S_Norm.png should be created"
    
    print("Test Passed!")
    # Test Forward
    user_ids = torch.arange(10)
    # LIRA forward expects users tensor
    scores = model.forward(user_ids, items=None)
    print(f"Scores shape: {scores.shape}")
    assert scores.shape == (10, n_items)
    
    print("All Tests Passed!")

if __name__ == "__main__":
    test_lyra_dual_ridge()
