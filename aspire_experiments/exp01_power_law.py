# Usage: uv run python aspire_experiments/exp2_power_law.py --dataset ml1m --energy 1.0
import os
import sys
import json
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from scipy.stats import spearmanr, pearsonr

# Framework root path
sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import get_loader_and_svd, ensure_dir, get_trimmed_data
from src.models.csar.ASPIRELayer import AspireEngine
from src.models.csar import beta_estimators

def run_power_law(dataset_name, seed=42):
    print(f"Running Experiment 2: Power-law Coupling on {dataset_name} (Full Spectrum, seed={seed})...")
    loader, R, S, V, config = get_loader_and_svd(dataset_name, seed=seed)
    
    item_pops = np.array(R.sum(axis=0)).flatten()
    s_np = S.cpu().numpy()
    
    p_tilde = AspireEngine.compute_spp(V, item_pops)
    
    M_full, N_full = R.shape
    Q_val = float(N_full) / float(M_full)
    
    print(f"\n[Estimating Beta with Multiple Methods]")
    estimates = beta_estimators.estimate_all(
        s_np, p_tilde, 
        item_freq=item_pops, 
        n_items=N_full, 
        n_users=M_full,
        trim_tail=0.05
    )
    
    raw_slope = np.linalg.lstsq(np.column_stack([np.log(s_np + 1e-12), np.ones_like(s_np)]), np.log(p_tilde + 1e-12), rcond=None)[0][0]
    
    print(f"\n[Beta Fitting Comparison for {dataset_name}]")
    print(f"  Observed Slope: {raw_slope:.4f}")
    
    s_norm = s_np / (s_np.max() + 1e-12)
    x = np.log(p_tilde + 1e-9)
    y = np.log(s_norm + 1e-9)
    x_trim, y_trim = get_trimmed_data(x, y)
    
    def calculate_r2(beta_val):
        slope = beta_val
        intercept = np.mean(y_trim) - slope * np.mean(x_trim)
        y_pred = slope * x_trim + intercept
        ss_res = np.sum((y_trim - y_pred)**2)
        ss_tot = np.sum((y_trim - np.mean(y_trim))**2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
        return float(r2)

    result_dict = {"dataset": config["dataset_name"]}
    for k, v in estimates.items():
        if isinstance(v, tuple) and len(v) >= 2:
            beta_val = v[0]
            if isinstance(beta_val, np.ndarray):
                b_mean = np.mean(beta_val)
                r2_cent = calculate_r2(b_mean)
                print(f"  {k:15s}: \u03b2(mean)={b_mean:7.4f}, R\u00b2(Cent)={r2_cent:7.4f}")
                result_dict[f"beta_{k}"] = float(b_mean)
                result_dict[f"r2_{k}"] = r2_cent
            else:
                r2_cent = calculate_r2(beta_val)
                print(f"  {k:15s}: \u03b2={beta_val:7.4f}, R\u00b2(Cent)={r2_cent:7.4f}")
                result_dict[f"beta_{k}"] = float(beta_val)
                result_dict[f"r2_{k}"] = r2_cent
            
    out_dir = ensure_dir(f"aspire_experiments/output/powerlaw/{config['dataset_name']}")
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 7))
    plt.scatter(x_trim, y_trim, alpha=0.3, s=10, label="Data points (Trimmed)")
    
    colors = ["blue", "green", "orange", "purple", "red", "cyan", "black", "gray"]
    line_styles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
    
    for i, (method, res) in enumerate(estimates.items()):
        if not isinstance(res, tuple) or len(res) < 2: continue
        b, r_sq = res[0], res[1]
        c = colors[i % len(colors)]
        ls = line_styles[i % len(line_styles)]
        
        if isinstance(b, np.ndarray):
            # Try to plot curve if same len, else use mean
            if len(b) == len(x_trim):
                slope = b
                intercept = np.mean(y_trim) - np.mean(slope) * np.mean(x_trim)
                plt.plot(x_trim, slope * x_trim + intercept, color=c, linestyle=ls, label=f"{method} (Curve)")
            else:
                b_mean = np.mean(b)
                slope = b_mean
                intercept = np.mean(y_trim) - slope * np.mean(x_trim)
                plt.plot(x_trim, slope * x_trim + intercept, color=c, linestyle=ls, label=f"{method} (\u03b2_mean={b_mean:.2f})")
        else:
            # Power-law slope based on beta
            slope = b
            intercept = np.mean(y_trim) - slope * np.mean(x_trim)
            plt.plot(x_trim, slope * x_trim + intercept, color=c, linestyle=ls, label=f"{method} (\u03b2={b:.2f})")
    
    if "rmt_lad" in estimates and len(estimates["rmt_lad"]) >= 3:
        diag = estimates["rmt_lad"][2]
        if "lambda_plus" in diag:
            # Plotting RMT cutoff needs adjustment if axes changed
            # But x is now log(p). We need to find p corresponding to lambda_plus.
            # This is complex inside this script. We can skip or fix later.
            pass
            
    plt.xlabel("log(Spectral Propensity \u1e57_k)")
    plt.ylabel("log(Normalized Singular Value \u03c3_k)")
    plt.title(f"Spectral Power-law Analysis: {config['dataset_name']}\n(RMT & Rank-Index Estimation)\nQ = n_items/n_users = {Q_val:.3f}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "powerlaw_comparison.png"))
    plt.close()

    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=4)
        
    import pandas as pd
    pd.DataFrame({
        "rank": np.arange(1, len(s_np) + 1),
        "singular_value": s_np,
        "log_singular_value": x,
        "spp_ptilde": p_tilde,
        "log_spp_ptilde": y
    }).to_csv(os.path.join(out_dir, "powerlaw_data.csv"), index=False)
        
    print(f"  Result saved to {out_dir}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml100k", help="Dataset name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    dataset = args.dataset.replace("旋", "") # Clean up accidental char
    run_power_law(dataset, seed=args.seed)
