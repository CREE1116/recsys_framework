import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix, diags
import argparse
import yaml
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.data_loader import DataLoader
from src.models.csar.LIRA import LIRA

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def symmetric_normalize(adj):
    """Applies symmetric normalization: D^-0.5 * A * D^-0.5"""
    if hasattr(adj, 'toarray'):
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = diags(d_inv_sqrt)
        return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    else:
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

def compute_cosine_similarity(X_sparse):
    """
    Compute Cosine Similarity Matrix:
    Cos = D^-0.5 * (X^T @ X) * D^-0.5
    """
    print("Computing Cosine Similarity (Standard 1-Hop)...")
    X = X_sparse.copy()
    X.data = np.ones_like(X.data)
    A = X.T @ X
    A.setdiag(0)
    A.eliminate_zeros()
    return symmetric_normalize(A)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config', type=str, default='configs/dataset/ml1m.yaml')
    parser.add_argument('--lambda_val', type=float, default=100.0)
    parser.add_argument('--sample_ratio', type=float, default=0.1, help="Ratio of points to plot")
    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading dataset from {args.dataset_config}...")
    config = load_config(args.dataset_config)
    config['model'] = {'name': 'lira', 'reg_lambda': args.lambda_val, 'normalize': True}
    config['device'] = 'cpu'
    
    if 'evaluation' not in config:
        config['evaluation'] = {'validation_method': 'sampled', 'final_method': 'full'}
    
    data_loader = DataLoader(config)
    model = LIRA(config, data_loader)
    
    # 2. Build LIRA S
    print(f"Building LIRA S (lambda={args.lambda_val})...")
    model.lira_layer.build(model.train_matrix_csr)
    S_lira = model.lira_layer.S.detach().cpu().numpy()
    np.fill_diagonal(S_lira, 0) # Focus on off-diagonal
    
    # 3. Build Cosine Sim
    S_cosine_sparse = compute_cosine_similarity(model.train_matrix_csr)
    if hasattr(S_cosine_sparse, 'toarray'):
        S_cosine = S_cosine_sparse.toarray()
    else:
        S_cosine = S_cosine_sparse
    np.fill_diagonal(S_cosine, 0)

    # 4. Filter & Metrics
    print("Calculating Shrinkage Metrics...")
    
    # Mask where Cosine > 0 to have valid comparison
    mask = S_cosine > 0
    
    cosine_vals = S_cosine[mask]
    lira_vals = S_lira[mask]
    
    # Ratio: LIRA / Cosine  ( < 1 means suppression)
    ratios = lira_vals / (cosine_vals + 1e-10)
    
    # Popularity Analysis
    item_freq = np.array(model.train_matrix_csr.sum(axis=0)).flatten()
    rows, cols = np.where(mask)
    pop_i = item_freq[rows]
    pop_j = item_freq[cols]
    joint_pop = np.log1p(pop_i * pop_j)
    
    # Define Head/Tail based on quartiles of Joint Popularity
    pop_threshold_high = np.percentile(joint_pop, 80) # Top 20% most popular pairs
    pop_threshold_low = np.percentile(joint_pop, 20)  # Bottom 20% least popular pairs
    
    head_mask = joint_pop >= pop_threshold_high
    tail_mask = joint_pop <= pop_threshold_low
    
    # Calculate Metrics
    metrics = {
        "global_shrinkage_mean": float(np.mean(ratios)),
        "global_shrinkage_median": float(np.median(ratios)),
        "head_shrinkage_mean": float(np.mean(ratios[head_mask])),
        "tail_shrinkage_mean": float(np.mean(ratios[tail_mask])),
        "debiasing_ratio (head/tail)": float(np.mean(ratios[head_mask]) / np.mean(ratios[tail_mask]))
    }
    
    print("\n=== LIRA vs Cosine Metrics ===")
    print(json.dumps(metrics, indent=4))
    
    metrics_path = os.path.join('analysis', 'lira_vs_cosine_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

    # 5. Plot (Downsampled)
    
    # Filter Outliers (Top 0.1%)
    threshold_lira = np.percentile(lira_vals, 99.9)
    threshold_cosine = np.percentile(cosine_vals, 99.9)
    valid_mask = (lira_vals <= threshold_lira) & (cosine_vals <= threshold_cosine)
    
    print(f"Filtering outliers: Removed {np.sum(~valid_mask)} points (>{threshold_lira:.4f} LIRA or >{threshold_cosine:.4f} Cosine)")
    
    cosine_vals = cosine_vals[valid_mask]
    lira_vals = lira_vals[valid_mask]
    joint_pop = joint_pop[valid_mask]

    total_points = len(cosine_vals)
    if total_points > 100000:
        print(f"Downsampling from {total_points} points for plot...")
        idx = np.random.choice(total_points, size=int(total_points * args.sample_ratio), replace=False)
        cosine_vals_plot = cosine_vals[idx]
        lira_vals_plot = lira_vals[idx]
        joint_pop_plot = joint_pop[idx]
    else:
        cosine_vals_plot = cosine_vals
        lira_vals_plot = lira_vals
        joint_pop_plot = joint_pop
        
    print(f"Plotting {len(cosine_vals_plot)} points...")
    plt.figure(figsize=(10, 8))
    
    # Scatter
    scatter = plt.scatter(cosine_vals_plot, lira_vals_plot, c=joint_pop_plot, cmap='coolwarm', s=1, alpha=0.3, label='Data Points')
    
    # Identity Line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='y=x (Identity)')
    
    # --- TREND LINES ---
    # Recalculate masks for PLOTTING data (since we downsampled)
    pop_threshold_high = np.percentile(joint_pop, 80)
    pop_threshold_low = np.percentile(joint_pop, 20)
    
    head_mask_plot = joint_pop_plot >= pop_threshold_high
    tail_mask_plot = joint_pop_plot <= pop_threshold_low
    
    # Head Trend
    if np.sum(head_mask_plot) > 10:
        z_head = np.polyfit(cosine_vals_plot[head_mask_plot], lira_vals_plot[head_mask_plot], 1)
        p_head = np.poly1d(z_head)
        x_range = np.linspace(0, max(cosine_vals_plot), 100)
        plt.plot(x_range, p_head(x_range), 'r-', linewidth=2, label=f'Head Trend (Slope={z_head[0]:.2f})')
        
    # Tail Trend
    if np.sum(tail_mask_plot) > 10:
        z_tail = np.polyfit(cosine_vals_plot[tail_mask_plot], lira_vals_plot[tail_mask_plot], 1)
        p_tail = np.poly1d(z_tail)
        x_range = np.linspace(0, max(cosine_vals_plot), 100)
        plt.plot(x_range, p_tail(x_range), 'b-', linewidth=2, label=f'Tail Trend (Slope={z_tail[0]:.2f})')

    cbar = plt.colorbar(scatter)
    cbar.set_label('Log(Joint Popularity)', rotation=270, labelpad=15)
    
    plt.xlabel(f'Standard Cosine Similarity (Normalized A)')
    plt.ylabel(f'LIRA S Weight (lambda={args.lambda_val})')
    plt.title(f'LIRA vs Cosine Weights (Outliers Removed)\n(Head Slope: {z_head[0]:.2f}, Tail Slope: {z_tail[0]:.2f})')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max(cosine_vals_plot.max(), lira_vals_plot.max()) * 1.05)
    plt.ylim(0, max(cosine_vals_plot.max(), lira_vals_plot.max()) * 1.05)
    
    output_path = os.path.join('analysis', 'lira_vs_cosine_scatter.png')
    os.makedirs('analysis', exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Saved scatter plot to {output_path}")

if __name__ == "__main__":
    main()
