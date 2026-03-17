# Usage: uv run python aspire_experiments/exp08_beta_ablation.py --dataset ml1m --energy 0.99
import os
import sys
import json
import yaml
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import get_loader_and_svd, ensure_dir, AspireHPO
from src.models.csar.ASPIRELayer import AspireEngine
from src.models.csar import beta_estimators
from src.evaluation import get_recall, get_ndcg

def fast_eval_obj(S_in, alpha, beta, k, XV_val, V_t, val_gt, val_hist, u_ids):
    device = S_in.device
    h = AspireEngine.apply_filter(S_in, float(alpha), float(beta)).float().to(device)
    XV_val = XV_val.to(device)
    V_t = V_t.to(device)
    scores = torch.mm(XV_val * h, V_t.t())
    for idx, u_id in enumerate(u_ids):
        excl = list(val_hist.get(u_id, set()) - set(val_gt[u_id]))
        if excl: scores[idx, excl] = -1e9
    _, top_idx = torch.topk(scores, k=k, dim=1)
    top_idx_np = top_idx.cpu().numpy()
    
    ndcgs = [get_ndcg(top_idx_np[idx].tolist(), val_gt[u_id]) for idx, u_id in enumerate(u_ids)]
    return float(np.mean(ndcgs))

def full_test_evaluation(XV_test, filter_diag, V_t, test_gt, test_hist, u_ids, k=20):
    device = V_t.device
    XV_test = XV_test.to(device)
    filter_diag = filter_diag.to(device)
    scores = torch.mm(XV_test * filter_diag, V_t.t())
    for idx, u_id in enumerate(u_ids):
        excl = list(test_hist.get(u_id, set()) - set(test_gt[u_id]))
        if excl: scores[idx, excl] = -1e9
        
    _, top_idx = torch.topk(scores, k=k, dim=1)
    top_idx_np = top_idx.cpu().numpy()
    
    recalls = [get_recall(top_idx_np[idx].tolist(), test_gt[u_id]) for idx, u_id in enumerate(u_ids)]
    ndcgs = [get_ndcg(top_idx_np[idx].tolist(), test_gt[u_id]) for idx, u_id in enumerate(u_ids)]
    return float(np.mean(recalls)), float(np.mean(ndcgs))

def run_beta_ablation(dataset_name, target_energy=0.99, n_trials=30):
    print(f"\n[Beta Ablation] Running Beta Range Scan on {dataset_name}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    loader, R, S, V, config = get_loader_and_svd(dataset_name, target_energy=target_energy)
    
    item_pops = np.array(R.sum(axis=0)).flatten()
    s_np = S.cpu().numpy()
    p_tilde = AspireEngine.compute_spp(V, item_pops)
    
    # 1. Estimate theoretical Betas
    beta_lad, _ = beta_estimators.beta_lad(s_np, p_tilde)
    beta_ols, _ = beta_estimators.beta_ols(s_np, p_tilde)
    
    print(f"Theory LAD: {beta_lad:.4f}, OLS: {beta_ols:.4f}")
    
    # Validation Setup
    val_gt = loader.valid_df.groupby("user_id")["item_id"].apply(list).to_dict()
    val_hist = loader.train_user_history
    val_u_t = torch.LongTensor(list(val_gt.keys())).to(device)
    val_u_ids = list(val_gt.keys())
    
    # Test Setup
    test_gt = loader.test_df.groupby("user_id")["item_id"].apply(list).to_dict()
    test_hist = loader.eval_user_history
    test_u_t = torch.LongTensor(list(test_gt.keys())).to(device)
    test_u_ids = list(test_gt.keys())
    
    XV_np = R.dot(V.cpu().numpy())
    XV_t = torch.from_numpy(XV_np).float().to(device)
    
    XV_val = XV_t[val_u_t]
    XV_test = XV_t[test_u_t]
    S_t, V_t = S.to(device).float(), V.to(device).float()

    # Define Beta scan points
    beta_points = list(np.arange(0.0, 1.51, 0.25))
    # Ensure theoretical points are evaluated exactly
    for b in [beta_lad]:
        if not any(np.isclose(b, bp, atol=0.05) for bp in beta_points):
            beta_points.append(float(b))
    beta_points = sorted(list(set(beta_points)))
    
    results = []
    out_dir = ensure_dir(f"aspire_experiments/output/beta_ablation/{dataset_name}")

    for beta in beta_points:
        print(f"\n--- Scanning Beta = {beta:.3f} ---")
        
        def objective(params):
            return fast_eval_obj(S_t, params["alpha"], beta, 20, XV_val, V_t, val_gt, val_hist, val_u_ids)
            
        hpo = AspireHPO([{"name": "alpha", "type": "float", "range": "1.0 1000000.0", "log": True}], 
                        n_trials=n_trials, patience=15)
        best_params, val_ndcg = hpo.search(objective, study_name=f"BetaAblation_{beta:.2f}")
        
        # Test Evaluation
        h_test = AspireEngine.apply_filter(s_np, best_params["alpha"], beta)
        h_test_t = torch.from_numpy(h_test).float().to(device)
        
        test_recall, test_ndcg = full_test_evaluation(XV_test, h_test_t, V_t, test_gt, test_hist, test_u_ids, k=20)
        
        print(f"  Test Recall@20: {test_recall:.4f}, Test NDCG@20: {test_ndcg:.4f}")
        
        results.append({
            "beta": float(beta),
            "opt_alpha": float(best_params["alpha"]),
            "val_ndcg20": float(val_ndcg),
            "test_recall20": float(test_recall),
            "test_ndcg20": float(test_ndcg)
        })

    # Save and Plot
    df = pd.DataFrame(results)
    df = df.sort_values("beta")
    df.to_csv(os.path.join(out_dir, "beta_ablation_results.csv"), index=False)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df["beta"], df["test_recall20"], marker='o', linestyle='-', linewidth=2, color='royalblue', label="Recall@20")
    
    # Mark theoretical points
    colors = {'LAD': ('green', beta_lad)}
    for name, (color, val) in colors.items():
        # Interpolate y value for the line
        y_val = np.interp(val, df["beta"], df["test_recall20"])
        plt.axvline(x=val, color=color, linestyle='--', alpha=0.8, label=rf"$\hat{{\beta}}$ ({name}) = {val:.3f}")
        plt.scatter([val], [y_val], color=color, s=100, zorder=5, marker='D')

    plt.xlabel(r"Filter Exponent Parameter $\beta$")
    plt.ylabel("Recall@20 (Test)")
    plt.title(f"Ablation: Recall vs $\\beta$ ({dataset_name})\n(Optimized $\\alpha$ for each $\\beta$)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "beta_ablation_recall.png"), dpi=150)
    plt.close()
    
    # Also plot NDCG
    plt.figure(figsize=(10, 6))
    plt.plot(df["beta"], df["test_ndcg20"], marker='s', linestyle='-', linewidth=2, color='crimson', label="NDCG@20")
    for name, (color, val) in colors.items():
        y_val = np.interp(val, df["beta"], df["test_ndcg20"])
        plt.axvline(x=val, color=color, linestyle='--', alpha=0.8, label=rf"$\hat{{\beta}}$ ({name}) = {val:.3f}")
        plt.scatter([val], [y_val], color=color, s=100, zorder=5, marker='D')
    plt.xlabel(r"Filter Exponent Parameter $\beta$")
    plt.ylabel("NDCG@20 (Test)")
    plt.title(f"Ablation: NDCG vs $\\beta$ ({dataset_name})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "beta_ablation_ndcg.png"), dpi=150)
    plt.close()

    print(f"\n[Done] Beta ablation curve saved to {out_dir}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml100k")
    parser.add_argument("--energy", type=float, default=0.99)
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    
    run_beta_ablation(args.dataset, args.energy, args.trials)
