# Usage: uv run python aspire_experiments/exp07_wiener_vs_aspire.py --dataset ml100k
#
# §Theory: Wiener vs. ASPIRE Filter Comparison (with HPO)
#
# 표준 Tikhonov/Wiener 필터(MCAR 가정, beta=0)와 
# ASPIRE 필터(MNAR 가정, beta 추론값)의 최적 성능 및 형상을 비교한다.
# 각각의 방식에 대해 최적의 alpha를 HPO로 탐색한 후 결과를 대조한다.

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import get_loader_and_svd, ensure_dir, AspireHPO
from src.models.csar.ASPIRELayer import AspireEngine
from src.models.csar import beta_estimators
from src.evaluation import get_ndcg

def fast_val_ndcg(XV_val, filter_diag, V_t, val_gt, val_hist, device, k=10):
    # Ensure everything is on the same device
    XV_val = XV_val.to(device)
    filter_diag = filter_diag.to(device)
    V_t = V_t.to(device)
    
    scores = torch.mm(XV_val * filter_diag, V_t.t())
    u_ids = list(val_gt.keys())
    for idx, u_id in enumerate(u_ids):
        excl = list(val_hist.get(u_id, set()) - set(val_gt[u_id]))
        if excl: scores[idx, excl] = -1e9
    _, top_idx = torch.topk(scores, k=k, dim=1)
    top_idx_np = top_idx.cpu().numpy()
    ndcgs = [get_ndcg(top_idx_np[idx].tolist(), val_gt[u_id]) for idx, u_id in enumerate(u_ids)]
    return float(np.mean(ndcgs))

def run_filter_comparison(dataset_name, target_energy=0.95, n_trials=30):
    print(f"\n[Comparison] HPO Comparison of Wiener vs ASPIRE on {dataset_name}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    loader, R, S, V, config = get_loader_and_svd(dataset_name, target_energy=target_energy)
    
    item_pops = np.array(R.sum(axis=0)).flatten()
    s_np = S.cpu().numpy()
    p_tilde = AspireEngine.compute_spp(V, item_pops)
    
    # 1. Beta Estimation (Switch to LAD as requested)
    beta_aspire, _ = beta_estimators.beta_lad(S, p_tilde)
    beta_wiener = 0.0
    
    # Validation Setup
    val_loader = loader.get_validation_loader(batch_size=2048)
    val_users, val_items = [], []
    for u_b, i_b in val_loader:
        val_users.append(u_b.numpy()); val_items.append(i_b.numpy())
    val_users = np.concatenate(val_users); val_items = np.concatenate(val_items)
    val_df = pd.DataFrame({"u": val_users, "i": val_items})
    val_gt = val_df.groupby("u")["i"].apply(list).to_dict()
    val_hist = loader.train_user_history
    
    XV_np = R.dot(V.cpu().numpy())
    XV_t = torch.from_numpy(XV_np).float().to(device)
    val_u_t = torch.LongTensor(list(val_gt.keys())).to(device)
    XV_val = XV_t[val_u_t]
    
    S_t, V_t = S.to(device).float(), V.to(device).float()
    
    results = []
    
    # 2. HPO for both
    for name, beta in [("Wiener", beta_wiener), ("ASPIRE", beta_aspire)]:
        print(f"\n--- Optimizing {name} (Beta={beta:.4f}) ---")
        
        def objective(params):
            h = AspireEngine.apply_filter(S_t, params["alpha"], beta).to(device)
            return fast_val_ndcg(XV_val, h, V_t, val_gt, val_hist, device)
        
        hpo = AspireHPO([{"name": "alpha", "type": "float", "range": "1.0 1000000.0", "log": True}], 
                        n_trials=n_trials, patience=20)
        best_params, best_val = hpo.search(objective, study_name=f"Comp_{name}")
        
        h_best = AspireEngine.apply_filter(s_np, best_params["alpha"], beta)
        
        results.append({
            "name": name,
            "beta": float(beta),
            "best_alpha": float(best_params["alpha"]),
            "ndcg": float(best_val),
            "filter_diag": h_best
        })

    # 3. Analyze and Visualize
    out_dir = ensure_dir(os.path.join("aspire_experiments/output/filter_comp", dataset_name))
    
    # Gain calculation (ASPIRE / Wiener)
    h_w = results[0]["filter_diag"]
    h_a = results[1]["filter_diag"]
    gain = h_a / (h_w + 1e-12)
    
    # Metrics
    head_size = max(1, len(s_np) // 10)
    tail_start = len(s_np) - head_size
    tail_gain = np.mean(gain[tail_start:])
    
    plt.figure(figsize=(15, 6))
    
    # Subplot 1: Filter Shape
    plt.subplot(1, 2, 1)
    plt.plot(s_np, h_w, 'k--', label=f'Wiener (NDCG={results[0]["ndcg"]:.4f})')
    plt.plot(s_np, h_a, 'orange', lw=2, label=f'ASPIRE (NDCG={results[1]["ndcg"]:.4f})')
    plt.gca().invert_xaxis()
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"Optimized Filter Comparison ({dataset_name})")
    plt.xlabel("Singular Value (sigma)")
    plt.ylabel("h(sigma)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    
    # Subplot 2: Relative Gain
    plt.subplot(1, 2, 2)
    plt.plot(s_np, gain, color='green', lw=2, label='Spectral Gain (Aspire/Wiener)')
    plt.axhline(1.0, color='gray', ls=':')
    plt.gca().invert_xaxis()
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"Gain: Tail Lift = {tail_gain:.2f}x")
    plt.xlabel("Singular Value (sigma)")
    plt.ylabel("Gain Ratio")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "optimized_filter_comparison.png"))
    plt.close()
    
    # 4. Filter Similarity
    cos_sim = float(cosine_similarity(h_w.reshape(1, -1), h_a.reshape(1, -1))[0][0])
    corr, _ = spearmanr(h_w, h_a)
    
    # 5. Save Final JSON & CSV
    final_output = {
        "dataset": dataset_name,
        "wiener": {
            "beta": results[0]["beta"],
            "alpha": results[0]["best_alpha"],
            "ndcg": results[0]["ndcg"]
        },
        "aspire": {
            "beta": results[1]["beta"],
            "alpha": results[1]["best_alpha"],
            "ndcg": results[1]["ndcg"]
        },
        "similarity": {
            "cosine": cos_sim,
            "spearman": float(corr)
        },
        "tail_gain": float(tail_gain),
        "ndcg_improvement_pct": float((results[1]["ndcg"] - results[0]["ndcg"]) / results[0]["ndcg"] * 100)
    }
    
    with open(os.path.join(out_dir, "result.json"), 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4)
        
    # Detailed CSV
    pd.DataFrame({
        "rank": np.arange(1, len(s_np) + 1),
        "sigma": s_np,
        "h_wiener": h_w,
        "h_aspire": h_a,
        "gain": gain
    }).to_csv(os.path.join(out_dir, "detailed_filter_data.csv"), index=False)
        
    print(f"  [Done] ASPIRE Improvement: {final_output['ndcg_improvement_pct']:.2f}%")
    print(f"  Result saved to {out_dir}")
    
    return final_output

def run_beta_sensitivity(dataset_name, target_energy=0.95, n_trials=30):
    print(f"\n[Sensitivity] Analyzing Beta Sensitivity on {dataset_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    loader, R, S, V, config = get_loader_and_svd(dataset_name, target_energy=target_energy)
    
    # Simple validation setup using loader's internal data
    val_gt = loader.valid_df.groupby("user_id")["item_id"].apply(list).to_dict()
    val_hist = loader.train_user_history
    
    XV_np = R.dot(V.cpu().numpy())
    XV_t = torch.from_numpy(XV_np).float()
    val_u_t = torch.LongTensor(list(val_gt.keys()))
    XV_val = XV_t[val_u_t]
    S_t, V_t = S.to(device), V.to(device)

    betas = np.linspace(0.0, 1.5, 16)
    sensitivity_results = []
    
    for b in betas:
        print(f"  Testing Beta = {b:.2f}...")
        def objective(params):
            h = AspireEngine.apply_filter(S_t, params["alpha"], b).to(device)
            return fast_val_ndcg(XV_val, h, V_t, val_gt, val_hist, device)
        
        hpo = AspireHPO([{"name": "alpha", "type": "float", "range": "1.0 1000000.0", "log": True}], 
                        n_trials=n_trials, patience=15)
        best_params, best_val = hpo.search(objective, study_name=f"Sens_B{b:.2f}")
        sensitivity_results.append({"beta": b, "ndcg": best_val, "alpha": best_params["alpha"]})

    out_dir = ensure_dir(os.path.join("aspire_experiments/output/sensitivity", dataset_name))
    df = pd.DataFrame(sensitivity_results)
    df.to_csv(os.path.join(out_dir, "beta_sensitivity.csv"), index=False)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df["beta"], df["ndcg"], marker='o', lw=2, color='blue')
    plt.axvline(df.loc[df["ndcg"].idxmax(), "beta"], color='red', ls='--', label=f'Best Beta={df.loc[df["ndcg"].idxmax(), "beta"]:.2f}')
    plt.title(f"Beta Sensitivity Analysis ({dataset_name})")
    plt.xlabel("Beta")
    plt.ylabel("Optimal NDCG@10")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(out_dir, "beta_sensitivity_plot.png"))
    plt.close()
    
    print(f"  Sensitivity plot saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml100k")
    parser.add_argument("--energy", type=float, default=0.95)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--sensitivity", action="store_true", help="Run beta sensitivity analysis")
    args = parser.parse_args()
    
    if args.sensitivity:
        run_beta_sensitivity(args.dataset, target_energy=args.energy, n_trials=args.trials)
    else:
        run_filter_comparison(args.dataset, target_energy=args.energy, n_trials=args.trials)
