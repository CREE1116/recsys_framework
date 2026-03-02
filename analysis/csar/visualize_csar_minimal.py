import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import numpy as np
import sys
import json
from yaml import safe_load

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models import get_model
from src.data_loader import DataLoader

def visualize_minimal_g(model_path, output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(model_path)
    
    # 1. Load Config
    config_path = os.path.join(os.path.dirname(model_path), 'config.yaml')
    with open(config_path, 'r') as f:
        config = safe_load(f)
    config['device'] = 'cpu'

    # 2. Load Data (Minimal loader)
    print(f"Loading data loader for {config['dataset_name']}...")
    data_loader = DataLoader(config)

    # 3. Load Model and Extract G
    print(f"Loading model from {model_path}...")
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Extract G from buffer
    G = state_dict.get('_cached_G')
    if G is None:
        # Try state_dict format
        G = state_dict.get('model_state_dict', {}).get('_cached_G')
    
    if G is None:
        print("Error: Could not find _cached_G in checkpoint.")
        return

    G = G.numpy()
    dim = G.shape[0]

    # 4. Compute Metrics
    diag_vals = np.diag(G)
    off_diag = G[~np.eye(dim, dtype=bool)]
    
    stats = {
        "diag_mean": float(np.mean(diag_vals)),
        "diag_std": float(np.std(diag_vals)),
        "off_diag_abs_mean": float(np.mean(np.abs(off_diag))),
        "off_diag_max": float(np.max(off_diag)),
        "off_diag_min": float(np.min(off_diag)),
        "sparsity_0.01": float((np.abs(off_diag) < 0.01).mean())
    }

    # 5. Plotting
    fig = plt.figure(figsize=(18, 5))
    
    # - Heatmap (Off-diagonal focus)
    plt.subplot(1, 3, 1)
    G_plot = G.copy()
    np.fill_diagonal(G_plot, 0) # Zero diagonal for better contrast
    vmax = np.percentile(np.abs(G_plot), 99.5) * 1.5
    vmax = max(vmax, 1e-4)
    sns.heatmap(G_plot, cmap='coolwarm', center=0, vmax=vmax, vmin=-vmax, square=True)
    plt.title(f"G Matrix (Off-diagonal)\nRelational Signal Strength")

    # - SVD Spectrum
    plt.subplot(1, 3, 2)
    U, S, V = np.linalg.svd(G)
    plt.plot(S, 'o-', markersize=4)
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.title(f"SVD Singular Values (Spectrum)\nEnergy Distribution")
    plt.xlabel("Rank Index")
    plt.ylabel("Value (log scale)")

    # - Weight Distribution
    plt.subplot(1, 3, 3)
    plt.hist(diag_vals, bins=30, alpha=0.5, label='Diagonal', color='blue', density=True)
    plt.hist(off_diag.flatten(), bins=50, alpha=0.5, label='Off-Diagonal', color='red', density=True)
    plt.yscale('log')
    plt.legend()
    plt.title(f"Weight Distribution (Log scale)\nDiagonal vs Relational")
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'csar_minimal_g_viz.png')
    plt.savefig(save_path, dpi=150)
    plt.close()

    # Save stats
    with open(os.path.join(output_dir, 'g_matrix_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)

    print(f"✓ Visualization saved to: {save_path}")
    print(f"✓ Stats saved to: {os.path.join(output_dir, 'g_matrix_stats.json')}")
    return save_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to best_model.pt')
    parser.add_argument('--output_dir', type=str, help='Custom output directory')
    args = parser.parse_args()
    
    visualize_minimal_g(args.model_path, args.output_dir)
