# Usage: uv run python aspire_experiments/exp09_alpha_transferability.py --source_dataset ml100k --target_datasets ml1m steam
import os
import sys
import json
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import get_loader_and_svd, ensure_dir, AspireHPO
from src.models.csar.ASPIRELayer import AspireEngine
from src.models.csar import beta_estimators

# Reuse the evaluation functions from exp08
from aspire_experiments.exp08_beta_ablation import fast_eval_obj, full_test_evaluation

def optimize_alpha(dataset_name, energy, n_trials):
    print(f"\n[Alpha Transfer] Optimizing native alpha for {dataset_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    loader, R, S, V, _ = get_loader_and_svd(dataset_name, target_energy=energy)
    
    item_pops = np.array(R.sum(axis=0)).flatten()
    s_np = S.cpu().numpy()
    p_tilde = AspireEngine.compute_spp(V, item_pops)
    
    # Use LAD as the standard theoretical beta for this experiment
    beta_hat, _ = beta_estimators.beta_lad(s_np, p_tilde)
    print(f"  Native estimated Beta (LAD): {beta_hat:.4f}")
    
    val_gt = loader.valid_df.groupby("user_id")["item_id"].apply(list).to_dict()
    val_hist = loader.train_user_history
    val_u_t = torch.LongTensor(list(val_gt.keys())).to(device)
    val_u_ids = list(val_gt.keys())
    
    test_gt = loader.test_df.groupby("user_id")["item_id"].apply(list).to_dict()
    test_hist = loader.eval_user_history
    test_u_t = torch.LongTensor(list(test_gt.keys())).to(device)
    test_u_ids = list(test_gt.keys())
    
    XV_np = R.dot(V.cpu().numpy())
    XV_t = torch.from_numpy(XV_np).float().to(device)
    
    XV_val = XV_t[val_u_t]
    XV_test = XV_t[test_u_t]
    S_t, V_t = S.to(device).float(), V.to(device).float()
    
    def objective(params):
        return fast_eval_obj(S_t, params["alpha"], beta_hat, 20, XV_val, V_t, val_gt, val_hist, val_u_ids)
        
    hpo = AspireHPO([{"name": "alpha", "type": "float", "range": "1.0 1000000.0", "log": True}], 
                    n_trials=n_trials, patience=15)
    best_params, val_ndcg = hpo.search(objective, study_name=f"AlphaNative_{dataset_name}")
    
    opt_alpha = float(best_params["alpha"])
    
    # Eval Native
    h_test = AspireEngine.apply_filter(s_np, opt_alpha, beta_hat)
    h_test_t = torch.from_numpy(h_test).float().to(device)
    test_recall, test_ndcg = full_test_evaluation(XV_test, h_test_t, V_t, test_gt, test_hist, test_u_ids, k=20)
    
    print(f"  Native Opt Alpha: {opt_alpha:.1f} | Test NDCG: {test_ndcg:.4f}")
    
    # Return everything needed to evaluate a transferred alpha later
    ctx = {
        "dataset_name": dataset_name,
        "beta_hat": float(beta_hat),
        "native_alpha": opt_alpha,
        "native_ndcg": test_ndcg,
        "native_recall": test_recall,
        "s_np": s_np,
        "device": device,
        "XV_test": XV_test,
        "V_t": V_t,
        "test_gt": test_gt,
        "test_hist": test_hist,
        "test_u_ids": test_u_ids
    }
    return ctx

def run_alpha_transfer(source_dataset, target_datasets, energy=0.95, n_trials=30):
    print(f"\n{'='*60}\nSelf-Normalizing Alpha Transfer Experiment\nSource: {source_dataset} -> Targets: {target_datasets}\n{'='*60}")
    
    # 1. Get Source Domain Optimal Alpha
    source_ctx = optimize_alpha(source_dataset, energy, n_trials)
    source_alpha = source_ctx["native_alpha"]
    print(f"\n>>> Established Source Alpha ({source_dataset}): {source_alpha:.1f} <<<")
    
    results = []
    out_dir = ensure_dir("aspire_experiments/output/alpha_transfer")
    
    # 2. Evaluate on Target Domains
    for t_ds in target_datasets:
        # Get target native optimal baseline
        t_ctx = optimize_alpha(t_ds, energy, n_trials)
        
        print(f"\nEvaluating Transfer to {t_ds}...")
        # Apply Source Alpha to Target Domain (using Target's own Beta)
        h_tf = AspireEngine.apply_filter(t_ctx["s_np"], source_alpha, t_ctx["beta_hat"])
        h_tf_t = torch.from_numpy(h_tf).float().to(t_ctx["device"])
        tf_recall, tf_ndcg = full_test_evaluation(
            t_ctx["XV_test"], h_tf_t, t_ctx["V_t"], t_ctx["test_gt"], t_ctx["test_hist"], t_ctx["test_u_ids"], k=20
        )
        
        rel_drop_ndcg = ((t_ctx["native_ndcg"] - tf_ndcg) / t_ctx["native_ndcg"]) * 100
        print(f"  Transfer NDCG: {tf_ndcg:.4f} (Drop: {rel_drop_ndcg:.2f}%)")
        
        results.append({
            "Source": source_dataset,
            "Target": t_ds,
            "Target_Beta": t_ctx["beta_hat"],
            "Native_Alpha": t_ctx["native_alpha"],
            "Transferred_Alpha": source_alpha,
            "Native_NDCG@20": t_ctx["native_ndcg"],
            "Transferred_NDCG@20": tf_ndcg,
            "Degradation_%": rel_drop_ndcg
        })
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, f"alpha_transfer_{source_dataset}_to_all.csv"), index=False)
    
    # Vis
    x = np.arange(len(target_datasets))
    width = 0.35
    
    plt.figure(figsize=(8, 6))
    plt.bar(x - width/2, df["Native_NDCG@20"], width, label='Native Opt (Tuned)', color='lightgray', edgecolor='black')
    plt.bar(x + width/2, df["Transferred_NDCG@20"], width, label=rf'Transferred ($\alpha={source_alpha:.0f}$)', color='royalblue', edgecolor='black')
    
    for i in range(len(target_datasets)):
        drop = df["Degradation_%"].iloc[i]
        plt.text(i + width/2, df["Transferred_NDCG@20"].iloc[i] + 0.001, f"{drop:.1f}%", ha='center')
        
    plt.ylabel('NDCG@20 (Test)')
    plt.title(f'Self-Normalizing Validation: Alpha Transfer from {source_dataset}')
    plt.xticks(x, target_datasets)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.savefig(os.path.join(out_dir, f"alpha_transfer_bar.png"), dpi=150)
    plt.close()
    
    print(f"\n[Done] Alpha Transfer Results saved to {out_dir}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dataset", type=str, default="ml100k")
    parser.add_argument("--target_datasets", nargs="+", default=["ml1m", "steam"])
    parser.add_argument("--energy", type=float, default=0.95)
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    
    run_alpha_transfer(args.source_dataset, args.target_datasets, args.energy, args.trials)
