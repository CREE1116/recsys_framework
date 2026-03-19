import os
import sys
import numpy as np
import torch

# Ensure root is in path
sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import get_loader_and_svd, get_trimmed_data
from src.models.csar.ASPIRELayer import AspireEngine

def debug_slopes(dataset_name):
    print(f"\n--- Debugging Slopes for {dataset_name} ---")
    loader, R, S, V, config = get_loader_and_svd(dataset_name)
    item_pops = np.array(R.sum(axis=0)).flatten().astype(float)
    item_pops = item_pops / (item_pops.max() + 1e-12)
    s_np = S.cpu().numpy()
    s_norm = s_np / (s_np.max() + 1e-12)
    p_tilde = AspireEngine.compute_spp(V, item_pops)
    p_tilde = p_tilde / (p_tilde.max() + 1e-12)
    
    indices = np.arange(len(p_tilde))
    t_idx, _ = get_trimmed_data(indices, indices)
    
    log_pt = np.log(np.clip(p_tilde[t_idx], 1e-12, None))
    log_s = np.log(np.clip(s_norm[t_idx], 1e-12, None))
    
    # Raw OLS Slope
    cov_matrix = np.cov(log_s, log_pt)
    slope = cov_matrix[0, 1] / (cov_matrix[1, 1] + 1e-12)
    
    print(f"Dataset: {dataset_name}")
    print(f"Raw Slope (log10 \u03c3 / log10 \u1e57): {slope:.6f}")
    
    # Check if this slope results in beta > 10
    # beta = slope / (2 - slope)
    if slope >= 2.0:
        print("ALERT: Slope >= 2.0! ASPIRE v3 bridge will overflow.")
    elif slope >= 1.818:
        print("ALERT: Slope >= 1.818! Beta will be clipped to 10.0.")
    else:
        beta = slope / (2.0 - slope)
        print(f"Calculated Beta (\u03b2): {beta:.6f}")

if __name__ == "__main__":
    for ds in ["ml100k", "ml1m"]:
        debug_slopes(ds)
