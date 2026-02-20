import torch
import torch.nn as nn
import numpy as np
import os
import sys
import yaml
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_loader import DataLoader
from src.models.general.item_knn import ItemKNN

def check_itemknn_scores(config_path):
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure train/eval keys
    if 'train' not in config: config['train'] = {}
    if 'evaluation' not in config: config['evaluation'] = {}
    config['device'] = 'cpu' # Default device
    config['model'] = {'k': 50, 'similarity_metric': 'cosine'}
    
    # Init DataLoader
    
    # Init DataLoader
    print("Initializing DataLoader...")
    data_loader = DataLoader(config)
    
    # Init ItemKNN
    print("Initializing ItemKNN...")
    model = ItemKNN(config, data_loader)
    
    # Fit (computation of similarity matrix)
    model.fit(data_loader)
    
    # Test on a few users
    test_users = torch.tensor([0, 10, 100, 1000]) # Sample user IDs
    
    print("\n--- Inspecting Scores ---")
    with torch.no_grad():
        # Move to same device
        if hasattr(model, 'device'):
            test_users = test_users.to(model.device)
            
        scores = model.forward(test_users) # [Batch, N_items]
        
        for i, user_id in enumerate(test_users):
            user_scores = scores[i]
            
            non_zero_count = (user_scores > 0).sum().item()
            zero_count = (user_scores == 0).sum().item()
            total_items = len(user_scores)
            
            print(f"User {user_id.item()}:")
            print(f"  Total Items: {total_items}")
            print(f"  Non-zero Scores: {non_zero_count} ({non_zero_count/total_items*100:.2f}%)")
            print(f"  Zero Scores: {zero_count} ({zero_count/total_items*100:.2f}%)")
            
            # Inspect Top-K
            k = 50
            top_vals, top_inds = torch.topk(user_scores, k=k)
            
            print(f"  Top-{k} Values: {top_vals[:10]} ... {top_vals[-5:]}")
            print(f"  Top-{k} Indices: {top_inds[:10]} ... {top_inds[-5:]}")
            
            # Check if zeros are in Top-K
            zeros_in_topk = (top_vals == 0).sum().item()
            if zeros_in_topk > 0:
                print(f"  [WARNING] {zeros_in_topk} items in Top-{k} have 0 score!")
                print(f"  Indices of 0-score items: {top_inds[top_vals == 0]}")
            else:
                print("  All Top-K items have > 0 score.")
            print("-" * 30)

if __name__ == '__main__':
    # Use Amazon Books config
    check_itemknn_scores('configs/dataset/amazon_books.yaml')
