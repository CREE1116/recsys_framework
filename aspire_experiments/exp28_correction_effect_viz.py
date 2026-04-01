import os
import sys
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import LinearRegression

# Add root directory to sys.path
sys.path.append(os.getcwd())

from src.data_loader import DataLoader
from src.models.csar.ASPIRE_Zero import ASPIRE_Zero
from aspire_experiments.exp_utils import ensure_dir

def run_correction_viz(dataset_name):
    # 1. Setup
    print(f"\n" + "="*60)
    print(f"Exp 28: ASPIRE-Zero Spectral Correction Effect Visualizer")
    print(f"Dataset: {dataset_name}")
    print(f"="*60)
    
    out_dir = ensure_dir("aspire_experiments/output/exp28")
    config_path = f"configs/dataset/{dataset_name}.yaml"
    
    with open(config_path, 'r') as f:
        ds_config = yaml.safe_load(f)
    
    # Generic model config for visualization
    model_config = {
        'model': {
            'name': 'aspire_zero',
            'max_iter': 15,
            'visualize': False # We will do custom plotting here
        },
        'dataset_name': dataset_name,
        'device': 'cpu',
        'train': {
            'embedding_l2': 0.0
        }
    }
    
    # 2. Load Data and Base EVD
    loader = DataLoader(ds_config)
    model = ASPIRE_Zero(model_config, loader)
    
    # Extract Raw Spectrum
    sigma_raw = np.sqrt(model.eigenvalues.cpu().numpy())
    h = model.filter_diag.cpu().numpy()
    
    # Apply Correction
    sigma_corrected = sigma_raw * h
    
    # 3. Slope Analysis (Full Range)
    def calculate_slope(s):
        valid = s > 1e-12
        ranks = np.arange(1, len(s) + 1)
        log_r = np.log(ranks[valid]).reshape(-1, 1)
        log_s = np.log(s[valid]).reshape(-1, 1)
        reg = LinearRegression().fit(log_r, log_s)
        return float(reg.coef_.item())

    slope_raw = calculate_slope(sigma_raw)
    slope_new = calculate_slope(sigma_corrected)
    
    print(f"  -> Raw Spectrum Slope (b) : {slope_raw:.6f}")
    print(f"  -> Corrected Slope (b)    : {slope_new:.6f}")
    print(f"  -> Delta b (Flattening)   : {slope_new - slope_raw:.6f}")

    # 4. Plotting
    plt.figure(figsize=(10, 8))
    
    ranks = np.arange(1, len(sigma_raw) + 1)
    
    plt.loglog(ranks, sigma_raw, label=f"Raw Spectrum (b={slope_raw:.3f})", color='blue', alpha=0.4, linewidth=1.5)
    plt.loglog(ranks, sigma_corrected, label=f"ASPIRE Corrected (b={slope_new:.3f})", color='red', linewidth=3)
    
    # Ideal MCAR Reference Line (Target: -0.2)
    # We draw a line starting from the top signal to show the target decay
    x_ideal = np.array([1, len(sigma_raw)])
    y_ideal = sigma_raw[0] * (x_ideal ** -0.2)
    plt.loglog(x_ideal, y_ideal, label="De-biased Target (b=-0.200)", color='green', linestyle=':', linewidth=2)
    
    plt.title(f"Spectral Flattening Effect Verification: {dataset_name}\n$\gamma_{{inferred}}$ = {model.gamma:.4f}")
    plt.xlabel("Rank (k)")
    plt.ylabel("Singular Value ($\sigma_k$)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.2)
    
    plot_path = os.path.join(out_dir, f"correction_effect_{dataset_name}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nVisualization saved to: {plot_path}")
    
    return slope_raw, slope_new, model.gamma

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (ml100k, ml1m, etc.)')
    args = parser.parse_args()
    
    run_correction_viz(args.dataset)
