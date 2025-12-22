import torch
import torch.nn as nn
import numpy as np
import os
import sys
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

# 프로젝트의 src 폴더를 python 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_loader import DataLoader
from src.models import get_model

def diagnose_distribution(experiment_dir):
    print(f"Diagnosing Distribution for: {experiment_dir}")
    
    config_path = os.path.join(experiment_dir, 'config.yaml')
    model_path = os.path.join(experiment_dir, 'best_model.pt')
    
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load Model
    data_loader = DataLoader(config)
    model = get_model(config['model']['name'], config, data_loader)
    
    # Check if model file exists, if not, we analyze initialized state
    if os.path.exists(model_path):
        print("Loading trained model weights...")
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        print("Model weights not found. Analyzing RANDOM INITIALIZATION.")

    model.eval()
    
    # Extract Tensors
    if hasattr(model, 'item_embedding'):
        item_embs = model.item_embedding.weight.detach() # [N, D]
    else:
        print("No item embedding found.")
        return

    keys = None
    if hasattr(model, 'attention_layer') and hasattr(model.attention_layer, 'interest_keys'):
        keys = model.attention_layer.interest_keys.detach() # [K, D]
    elif hasattr(model, 'interest_keys'):
        keys = model.interest_keys.detach()
        
    if keys is None:
        print("No interest keys found.")
        return
        
    # --- Analysis ---
    print("\n--- Statistics ---")
    print(f"Item Embeddings: shape={item_embs.shape}")
    print(f"   Mean: {item_embs.mean():.4f}, Std: {item_embs.std():.4f}")
    print(f"   Min:  {item_embs.min():.4f},  Max: {item_embs.max():.4f}")
    
    print(f"Interest Keys: shape={keys.shape}")
    print(f"   Mean: {keys.mean():.4f}, Std: {keys.std():.4f}")
    print(f"   Min:  {keys.min():.4f},  Max: {keys.max():.4f}")
    
    # Norms
    item_norms = torch.norm(item_embs, p=2, dim=1)
    key_norms = torch.norm(keys, p=2, dim=1)
    print(f"Item Norms: Mean={item_norms.mean():.4f}, Std={item_norms.std():.4f}")
    print(f"Key Norms:  Mean={key_norms.mean():.4f}, Std={key_norms.std():.4f}")
    
    # Cosine Similarity (Key vs Key)
    keys_norm = torch.nn.functional.normalize(keys, p=2, dim=1)
    key_sim = torch.matmul(keys_norm, keys_norm.t())
    off_diag_sim = key_sim[~torch.eye(keys.shape[0], dtype=bool)].abs().mean()
    print(f"Key-Key Orthogonality (Mean Abs Off-Diag): {off_diag_sim:.4f}")
    
    # Cosine Similarity (Item vs Key)
    # Check if keys are covering the item space
    items_norm = torch.nn.functional.normalize(item_embs, p=2, dim=1)
    item_key_sim = torch.matmul(items_norm, keys_norm.t()) # [N, K]
    
    max_sim_values, _ = item_key_sim.max(dim=1)
    print(f"Item-Key Max Similarity (Coverage): Mean={max_sim_values.mean():.4f}")
    
    # Histogram
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(item_embs.flatten().numpy(), bins=100, color='blue', alpha=0.5, label='Items')
    sns.histplot(keys.flatten().numpy(), bins=100, color='red', alpha=0.5, label='Keys')
    plt.title("Value Distribution (Flattened)")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    sns.histplot(key_norms.numpy(), bins=20, color='red', label='Key Norms')
    sns.histplot(item_norms.numpy(), bins=50, color='blue', label='Item Norms')
    plt.title("Norm Distribution")
    plt.legend()
    
    output_plot = os.path.join(experiment_dir, 'distribution_diagnosis.png')
    plt.savefig(output_plot)
    print(f"Saved distribution plot to {output_plot}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, help="Experiment directory")
    args = parser.parse_args()
    
    diagnose_distribution(args.dir)
