import torch
import yaml
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DataLoader
from src.models.csar.GroupedLIRA import GroupedLIRA

def test_explain():
    # Load dataset config
    with open('configs/dataset/ml100k.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Add model config
    config['model'] = {
        'name': 'grouped_lira',
        'num_interests': 10,
        'embedding_dim': 64,
        'reg_lambda': 500.0
    }
    config['evaluation'] = {
        'validation_method': 'sampled',
        'final_method': 'full'
    }
    config['train'] = {
        'loss_type': 'pairwise'
    }
    config['device'] = 'cpu'
    
    print("Loading data...")
    data_loader = DataLoader(config)
    
    print("Building model...")
    model = GroupedLIRA(config, data_loader)
    
    user_id = 0
    item_id = 10
    
    print(f"Testing explain for user {user_id}, item {item_id}...")
    explanation = model.explain(user_id, item_id)
    
    print("\nExplanation:")
    for k, v in explanation.items():
        if k.startswith('interest'):
             if abs(v) > 1e-6:
                print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v:.6f}")
    
    # Verify reconstruction
    x_u = torch.from_numpy(model.train_matrix_csr[user_id].toarray()).float()
    u_proj = x_u @ model.V_rot.cpu()
    i_proj = model.V_rot[item_id].cpu()
    actual_score = (u_proj * model.f.cpu() * i_proj).sum().item()
    
    print(f"\nActual Score: {actual_score:.6f}")
    print(f"Recovered Score: {explanation['total_recovered']:.6f}")
    
    assert abs(actual_score - explanation['total_recovered']) < 1e-5
    print("\nReconstruction successful!")

if __name__ == "__main__":
    test_explain()
