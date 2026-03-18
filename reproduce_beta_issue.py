import os
import sys
import numpy as np
import torch

sys.path.append(os.getcwd())

from src.models.csar import beta_estimators
from src.models.csar.ASPIRELayer import AspireEngine
from aspire_experiments.exp_utils import get_loader_and_svd

def debug_ml1m():
    dataset = "ml1m"
    print(f"Loading {dataset}...")
    try:
        loader, R, S, V, config = get_loader_and_svd(dataset, seed=42)
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    item_pops = np.array(R.sum(axis=0)).flatten()
    s_np = S.cpu().numpy()
    p_tilde = AspireEngine.compute_spp(V, item_pops)

    print(f"S shape: {s_np.shape}, p_tilde shape: {p_tilde.shape}")
    print(f"S top 5: {s_np[:5]}")
    print(f"p_tilde top 5: {p_tilde[:5]}")

    # Test Truncated LAD
    k = 100
    b_trunc, r2_trunc = beta_estimators.beta_truncated_lad(s_np, p_tilde, k=k)
    print(f"Truncated LAD (k={k}): beta={b_trunc:.4f}, r2={r2_trunc:.4f}")

    # Diagnostic LAD call trace
    s_trunc = s_np[:k]
    pt_trunc = p_tilde[:k]
    
    from src.models.csar.beta_estimators import _log_xy, _lad_solve, _slope_to_beta
    x, y = _log_xy(s_trunc, pt_trunc, trim_tail=0.0)
    print(f"Log points (mask applied): {len(x)}")
    if len(x) > 0:
        print(f"x range: [{x.min():.4f}, {x.max():.4f}]")
        print(f"y range: [{y.min():.4f}, {y.max():.4f}]")
    
    slope = _lad_solve(x, y)
    print(f"LAD Slope: {slope}")
    if slope is not None:
        beta = _slope_to_beta(slope)
        print(f"Beta: {beta}")

if __name__ == "__main__":
    debug_ml1m()
