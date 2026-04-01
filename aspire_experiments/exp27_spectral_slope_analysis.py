import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Add root directory to sys.path
sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import get_loader_and_svd, ensure_dir

def analyze_full_spectral_slope(dataset='ml100k'):
    print(f"\nAnalyzing Full Spectral Slope: {dataset}")
    
    # 1. Load Full SVD/EVD data
    # We use k=None to get Full spectrum
    try:
        loader, R_train, S_full_torch, V_full_torch, _ = get_loader_and_svd(dataset, k=None)
        sigma = np.sqrt(S_full_torch.detach().cpu().numpy())
    except Exception as e:
        print(f"Error loading {dataset}: {e}")
        return None

    # 2. Prepare Rank and Log-values (Full Range)
    N = len(sigma)
    ranks = np.arange(1, N + 1)
    
    # Ensure no zeros for log taking
    valid_mask = sigma > 1e-12
    log_ranks = np.log(ranks[valid_mask]).reshape(-1, 1)
    log_sigmas = np.log(sigma[valid_mask]).reshape(-1, 1)

    # 3. Linear Regression (Full Range)
    reg = LinearRegression().fit(log_ranks, log_sigmas)
    slope = reg.coef_[0][0]
    r_squared = reg.score(log_ranks, log_sigmas)

    print(f"  -> Full Slope (b): {slope:.6f}")
    print(f"  -> R-squared     : {r_squared:.6f}")
    print(f"  -> Total Elements: {N}")

    return {
        'dataset': dataset,
        'sigma': sigma,
        'ranks': ranks,
        'slope': slope,
        'r_squared': r_squared,
        'N': N
    }

def analyze_specific_mcar(dataset='yahoo_r3'):
    """Special function to extract and analyze the MCAR (Random) test set for Yahoo R3"""
    print(f"\nAnalyzing Specific MCAR Set: {dataset}")
    from src.data_loader import DataLoader
    import yaml
    from scipy.sparse import csr_matrix
    import torch
    
    # 1. Load config and data
    config_path = f"configs/dataset/{dataset}.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    loader = DataLoader(config)
    test_df = loader.test_df # This is the MCAR set
    
    # 2. Build sparse matrix from MCAR
    # We use the same dimensions as the train set
    R_mcar = csr_matrix(
        (np.ones(len(test_df)), (test_df['user_id'].values, test_df['item_id'].values)),
        shape=(loader.n_users, loader.n_items)
    )
    
    # 3. Perform EVD on MCAR
    # Gram matrix C = R^T R
    C_mcar = R_mcar.T @ R_mcar
    # Use scipy for full EVD if small enough, or torch
    # Since Yahoo-R3 items=1000, it's fast
    from scipy.linalg import eigh
    evals, _ = eigh(C_mcar.toarray())
    sigma_mcar = np.sqrt(np.maximum(evals[::-1], 0))
    
    # 4. Slope Analysis
    N = len(sigma_mcar)
    ranks = np.arange(1, N + 1)
    valid_mask = sigma_mcar > 1e-10
    log_ranks = np.log(ranks[valid_mask]).reshape(-1, 1)
    log_sigmas = np.log(sigma_mcar[valid_mask]).reshape(-1, 1)
    
    reg = LinearRegression().fit(log_ranks, log_sigmas)
    slope = reg.coef_[0][0]
    r2 = reg.score(log_ranks, log_sigmas)
    
    print(f"  -> MCAR Slope (b) : {slope:.6f}")
    print(f"  -> MCAR R-squared : {r2:.6f}")
    
    return {
        'dataset': f"{dataset}_MCAR",
        'sigma': sigma_mcar,
        'ranks': ranks,
        'slope': slope,
        'r_squared': r2,
        'N': N
    }

def run_analysis():
    datasets = ['ml100k', 'ml1m', 'steam', 'yahoo_r3']
    results = []
    
    out_dir = ensure_dir("aspire_experiments/output/exp27")
    
    plt.figure(figsize=(10, 8))
    
    # Standard Datasets (MNAR/Biased Train)
    for ds in datasets:
        res = analyze_full_spectral_slope(ds)
        if res:
            results.append(res)
            plt.loglog(res['ranks'], res['sigma'], label=f"{ds} (MNAR) b={res['slope']:.3f}", alpha=0.6)
            
    # Special MCAR Analysis
    mcar_res = analyze_specific_mcar('yahoo_r3')
    if mcar_res:
        results.append(mcar_res)
        plt.loglog(mcar_res['ranks'], mcar_res['sigma'], 
                  label=f"yahoo_r3 (MCAR) b={mcar_res['slope']:.3f}", 
                  linewidth=3, linestyle='--', color='red')
            
    plt.title("Spectral Slope Comparison: MNAR vs MCAR (Full Range)")
    plt.xlabel("Rank (k)")
    plt.ylabel("Singular Value ($\sigma_k$)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    
    plot_path = os.path.join(out_dir, "full_spectral_slopes.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nFinal analysis plot saved to: {plot_path}")
    
    # Print Summary Table
    print("\n" + "="*60)
    print(f"{'Dataset':<12} | {'Slope (b)':<10} | {'R-squared':<10} | {'Size':<6}")
    print("-" * 60)
    for r in results:
        print(f"{r['dataset']:<12} | {r['slope']:<10.4f} | {r['r_squared']:<10.4f} | {r['N']:<6}")
    print("="*60)

if __name__ == "__main__":
    run_analysis()
