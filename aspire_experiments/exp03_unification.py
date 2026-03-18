# Usage: uv run python aspire_experiments/exp03_unification.py --datasets ml1m ml100k steam
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import get_loader_and_svd, ensure_dir
from src.models.csar.ASPIRELayer import AspireEngine
from src.models.csar import beta_estimators

def run_unification(dataset_name, seed=42):
    print(f"\n[Unification] Analyzing {dataset_name} (Full Spectrum, seed={seed})...")
    loader, R, S, V, config = get_loader_and_svd(dataset_name, seed=seed)
    
    item_freq = np.array(R.sum(axis=0)).flatten().astype(float)
    s_np = S.cpu().numpy()
    p_tilde = AspireEngine.compute_spp(V, item_freq)
    
    # 1. Projection-based Beta (The standard SPP way)
    # log p_k ~ (2β) log σ_k + C
    beta_proj, r2_proj = beta_estimators.beta_lad(S, p_tilde)
    
    # 2. Direct Slope-Ratio (Theoretical derivation)
    # log σ_k ~ -α log k
    # log n_i ~ -η log i
    # β_direct = η / (2α - η)  [Derived by User]
    def calculate_beta_theory(S_in, pops):
        s_vals = S_in.cpu().numpy()
        def get_slope(v):
            L = len(v)
            lx = np.log(np.arange(1, L + 1))
            ly = np.log(np.clip(v, 1e-12, None))
            return abs(np.polyfit(lx, ly, 1)[0])
        
        alpha_eff = get_slope(s_vals)
        eta_eff = get_slope(np.sort(pops)[::-1])
        denom = 2.0 * alpha_eff - eta_eff
        if denom <= 0: return eta_eff / (2.0 * alpha_eff + 1e-9)
        return eta_eff / denom

    beta_direct = calculate_beta_theory(S, item_freq)
    
    # 3. Component Slopes for Analysis
    def get_abs_slope(vals):
        L = len(vals)
        x = np.log(np.arange(1, L + 1))
        y = np.log(np.clip(vals, 1e-12, None))
        A = np.column_stack([x, np.ones_like(x)])
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return abs(float(slope))

    alpha = get_abs_slope(s_np)
    eta = get_abs_slope(np.sort(item_freq)[::-1])
    
    print(f"  Spectral  α: {alpha:.4f}")
    print(f"  Frequency η: {eta:.4f}")
    print(f"  β_direct (η/2α): {beta_direct:.4f}")
    print(f"  β_proj   (SPP) : {beta_proj:.4f} (R²={r2_proj:.4f})")
    print(f"  Difference     : {abs(beta_direct - beta_proj):.4f}")
    
    return {
        "dataset": dataset_name,
        "alpha": alpha,
        "eta": eta,
        "beta_direct": beta_direct,
        "beta_proj": beta_proj,
        "diff": abs(beta_direct - beta_proj),
        "r2_proj": r2_proj
    }

def main():
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    results = []
    for ds in args.datasets:
        res = run_unification(ds, seed=args.seed)
        results.append(res)
        
    # Visualization
    out_dir = ensure_dir("aspire_experiments/output/theory_unification")
    
    # Comparison Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ds_names = [r["dataset"] for r in results]
    b_direct = [r["beta_direct"] for r in results]
    b_proj = [r["beta_proj"] for r in results]
    
    x = np.arange(len(ds_names))
    width = 0.35
    
    ax.bar(x - width/2, b_direct, width, label='β_direct (η/2α)', color='skyblue', alpha=0.8)
    ax.bar(x + width/2, b_proj, width, label='β_proj (SPP-LAD)', color='salmon', alpha=0.8)
    
    ax.set_ylabel('Beta Value')
    ax.set_title('ASPIRE Unification: Direct vs Projection-based Beta')
    ax.set_xticks(x)
    ax.set_xticklabels(ds_names)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(out_dir, "beta_comparison.png"), dpi=150)
    plt.close()
    
    # Ratio plot: η vs (2α * β_proj)
    plt.figure(figsize=(7, 7))
    etas = [r["eta"] for r in results]
    denoms = [2.0 * r["alpha"] * r["beta_proj"] for r in results]
    
    plt.scatter(denoms, etas, s=100, color='purple', alpha=0.6)
    for i, txt in enumerate(ds_names):
        plt.annotate(txt, (denoms[i], etas[i]), xytext=(5, 5), textcoords='offset points')
        
    line_max = max(max(etas), max(denoms)) * 1.1
    plt.plot([0, line_max], [0, line_max], 'k--', alpha=0.3, label='Theoretical Identity (η = 2αβ_proj)')
    
    plt.xlabel('2 * α * β_proj')
    plt.ylabel('Observed η (Item Freq Decay)')
    plt.title('Theory Validation: Coupling Consistency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "consistency_scatter.png"), dpi=150)
    plt.close()
    
    # [NEW] Save detailed summary items
    import pandas as pd
    pd.DataFrame(results).to_csv(os.path.join(out_dir, "unification_results.csv"), index=False)
    with open(os.path.join(out_dir, "unification_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    
    print(f"\n[Done] Result plots and data saved to {out_dir}")

if __name__ == "__main__":
    main()
