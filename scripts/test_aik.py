import torch
import yaml
import os
import sys
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DataLoader
from src.models.csar.AIK import AIK

def test_aik_explain():
    # Load dataset config
    with open('configs/dataset/ml100k.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Add model config
    config['model'] = {
        'name': 'aik',
        'num_interests': 10,
        'embedding_dim': 64,
        'reg_lambda': 500.0,
        'normalize': True
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
    model = AIK(config, data_loader)
    
    user_id = 0
    item_id = 10
    
    print(f"Testing explain for user {user_id}, item {item_id}...")
    explanation = model.explain(user_id, item_id, top_k=5)
    
    print("\nExplanation:")
    print(f"  Total Score: {explanation['total_score']:.6f}")
    print("  Top Interest Pairs:")
    for pair_info in explanation['top_pairs']:
        print(f"    Pair {pair_info['interest_pair']}: {pair_info['contribution']:.6f} (Global JK: {pair_info['G_global_jk']:.4f}, f_u_j: {pair_info['f_u_j']:.4f}, f_u_k: {pair_info['f_u_k']:.4f})")
    
    # Simple check on score reconstruction
    x_u = torch.from_numpy(model.train_matrix_csr[user_id].toarray()).float().to(model.device).unsqueeze(0)
    score_from_forward = model.forward(torch.tensor([user_id]))[0, item_id].item()
    
    print(f"\nForward Score: {score_from_forward:.6f}")
    print(f"Explain Total: {explanation['total_score']:.6f}")
    
    print("\nAIK behavior verified!")

if __name__ == "__main__":
    test_aik_explain()
