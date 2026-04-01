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

def run_corrected_analysis(dataset_name):
    print(f"\n" + "="*60)
    print(f"Exp 29: Corrected Spectral Slope Analysis (with gamma/2 factor)")
    print(f"Dataset: {dataset_name}")
    print(f"="*60)
    
    out_dir = ensure_dir("aspire_experiments/output/exp29")
    config_path = f"configs/dataset/{dataset_name}.yaml"
    
    with open(config_path, 'r') as f:
        ds_config = yaml.safe_load(f)
    
    model_config = {
        'model': {
            'name': 'aspire_zero',
            'max_iter': 30,
            'visualize': False,
            'lambda_base': None # Auto sigma_1 based or HPO
        },
        'dataset_name': dataset_name,
        'device': 'cpu',
        'train': {'embedding_l2': 0.0}
    }
    
    loader = DataLoader(ds_config)
    model = ASPIRE_Zero(model_config, loader)
    
    # 1. Extract Spectrums
    sigma_raw = np.sqrt(model.eigenvalues.cpu().numpy())
    
    # [User Correction] 
    # Show pure "Space Correction" (Power-law scaling) instead of Wiener filter.
    # Formula: sigma_corrected = sigma_raw ** (gamma / 2.0)
    gamma = model.gamma
    exponent = gamma / 2.0
    sigma_new = sigma_raw ** exponent
    
    # 2. Slope Calculation (Middle-band: [20, 500])
    def calculate_slope(s, min_rank=20, max_rank=500):
        ranks = np.arange(1, len(s) + 1)
        # Filter for middle band
        mask = (ranks >= min_rank) & (ranks <= max_rank) & (s > 1e-12)
        
        if not np.any(mask): 
            return 0.0
            
        log_r = np.log(ranks[mask]).reshape(-1, 1)
        log_s = np.log(s[mask]).reshape(-1, 1)
        reg = LinearRegression().fit(log_r, log_s)
        return float(reg.coef_.item())

    slope_raw = calculate_slope(sigma_raw)
    slope_corrected = calculate_slope(sigma_new)
    
    print(f"  -> Inferred Gamma      : {gamma:.4f}")
    print(f"  -> Power Factor (G/2)  : {exponent:.4f}")
    print(f"  -> Raw Middle Slope    : {slope_raw:.6f} ([20, 500])")
    print(f"  -> Corrected Mid Slope : {slope_corrected:.6f} ([20, 500])")
    print(f"  -> Target Reference    : -0.200000")

    # 3. Plotting
    plt.figure(figsize=(10, 8))
    ranks = np.arange(1, len(sigma_raw) + 1)
    
    plt.loglog(ranks, sigma_raw, label=f"Raw Spectrum (b={slope_raw:.3f})", color='blue', alpha=0.4)
    plt.loglog(ranks, sigma_new, label=f"Space Corrected $\sigma^{{\gamma/2}}$ (b={slope_corrected:.3f})", color='red', linewidth=2)
    
    # Target Line (-0.2)
    x_ideal = np.array([1, len(sigma_raw)])
    y_ideal = sigma_raw[0] * (x_ideal ** -0.2)
    plt.loglog(x_ideal, y_ideal, label="De-biased Target (b=-0.200)", color='green', linestyle=':', linewidth=2)
    
    plt.title(f"Corrected Spectral Slope Analysis: {dataset_name}\nGamma={gamma:.4f} -> Factor(gamma/2)={gamma/2.0:.3f}")
    plt.xlabel("Rank (k)")
    plt.ylabel("Singular Value ($\sigma_k$)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.2)
    
    plot_path = os.path.join(out_dir, f"corrected_slope_{dataset_name}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nVisualization saved to: {plot_path}")
    
    return slope_raw, slope_corrected

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    run_corrected_analysis(args.dataset)
