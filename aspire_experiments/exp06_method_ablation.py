# Usage: uv run python aspire_experiments/exp06_method_ablation.py --dataset ml1m
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

def full_evaluation(XV_val, filter_diag, V_t, val_gt, val_hist, item_popularity, eval_config, device):
    top_k_list = eval_config.get('top_k', [10, 20, 50])
    metrics_list = eval_config.get('metrics', ['NDCG', 'Recall', 'HitRate', 'Coverage', 'LongTailCoverage'])
    lt_percent = eval_config.get('long_tail_percent', 0.8)
    
    XV_val = XV_val.to(device)
    h = filter_diag.to(device)
    V_t = V_t.to(device)
    u_ids = list(val_gt.keys())
    
    # 1. Compute Scores: [B, K] * [K] -> [B, K], then [B, K] mm [K, N] -> [B, N]
    scores = torch.mm(XV_val * h, V_t.t())
    for idx, u_id in enumerate(u_ids):
        excl = list(val_hist.get(u_id, set()) - set(val_gt[u_id]))
        if excl:
            scores[idx, excl] = -1e9
            
    # 2. Get Top-K Recommendations
    max_k = max(top_k_list)
    _, top_idx = torch.topk(scores, k=max_k, dim=1)
    top_idx = top_idx.cpu().numpy()
    
    # 3. Preparation for Global Metrics
    n_items = V_t.shape[0]
    from src.evaluation import get_long_tail_item_set, get_ndcg, get_recall, get_hit_rate
    tail_set = get_long_tail_item_set(item_popularity, head_volume_percent=lt_percent)
    
    final_results = {}
    
    for k in top_k_list:
        user_ndcgs = []
        user_recalls = []
        user_hits = []
        all_recs = []
        
        for idx, u_id in enumerate(u_ids):
            recs = top_idx[idx, :k].tolist()
            gt = val_gt[u_id]
            all_recs.extend(recs)
            
            user_ndcgs.append(get_ndcg(recs, gt))
            user_recalls.append(get_recall(recs, gt))
            user_hits.append(get_hit_rate(recs, gt))
            
        final_results[f"NDCG@{k}"] = float(np.mean(user_ndcgs))
        final_results[f"Recall@{k}"] = float(np.mean(user_recalls))
        final_results[f"HitRate@{k}"] = float(np.mean(user_hits))
        
        # Coverage@K
        unique_recs = set(all_recs)
        final_results[f"Coverage@{k}"] = len(unique_recs) / n_items
        
        # LongTailCoverage@K
        if tail_set:
            tail_recs = unique_recs.intersection(tail_set)
            final_results[f"LongTailCoverage@{k}"] = len(tail_recs) / len(tail_set)
            
    return final_results

def run_method_ablation(dataset_name, target_energy=0.99, n_trials=30):
    print(f"\n[Ablation] Comparing methods on {dataset_name} (Multi-Metric)...")
    
    # Load Evaluation Config
    with open("configs/evaluation.yaml", "r", encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f)["evaluation"]
    
    main_metric = eval_cfg.get("main_metric", "NDCG")
    main_k = eval_cfg.get("main_metric_k", 20)
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    loader, R, S, V, config = get_loader_and_svd(dataset_name, target_energy=target_energy)
    
    item_pops = np.array(R.sum(axis=0)).flatten()
    s_np = S.cpu().numpy()
    p_tilde = AspireEngine.compute_spp(V, item_pops)
    
    # Validation Setup
    val_gt = loader.valid_df.groupby("user_id")["item_id"].apply(list).to_dict()
    val_hist = loader.train_user_history
    
    XV_np = R.dot(V.cpu().numpy())
    XV_t = torch.from_numpy(XV_np).float().to(device)
    val_u_t = torch.LongTensor(list(val_gt.keys())).to(device)
    XV_val = XV_t[val_u_t]
    S_t, V_t = S.to(device).float(), V.to(device).float()

    # Estimators to compare
    methods = {
        "OLS": lambda s, pt, pops: beta_estimators.beta_ols(s, pt)[0],
        "LAD": lambda s, pt, pops: beta_estimators.beta_lad(s, pt)[0],
        "Pairwise": lambda s, pt, pops: beta_estimators.beta_pairwise_ratio(s, pt)[0]
    }
    
    detailed_results = []
    out_dir = ensure_dir(f"aspire_experiments/output/method_ablation/{dataset_name}")

    # Helper for simple HPO objective (NDCG@main_k)
    def fast_ndcg_obj(S_in, alpha, beta, k):
        h = AspireEngine.apply_filter(S_in, float(alpha), float(beta)).float()
        # We need a quick ranking calc for the specific K
        u_ids = list(val_gt.keys())
        # [Corrected dimensions]
        scores = torch.mm(XV_val * h, V_t.t())
        for idx, u_id in enumerate(u_ids):
            excl = list(val_hist.get(u_id, set()) - set(val_gt[u_id]))
            if excl: scores[idx, excl] = -1e9
        _, top_idx = torch.topk(scores, k=k, dim=1)
        top_idx_np = top_idx.cpu().numpy()
        
        from src.evaluation import get_ndcg
        ndcgs = [get_ndcg(top_idx_np[idx].tolist(), val_gt[u_id]) for idx, u_id in enumerate(u_ids)]
        return float(np.mean(ndcgs))

    for name, b_fn in methods.items():
        print(f"\n--- Testing Method: {name} ---")
        try:
            beta = b_fn(s_np, p_tilde, item_pops)
            print(f"  Estimated Beta: {beta:.4f}")
            
            def objective(params):
                return fast_ndcg_obj(S_t, params["alpha"], beta, main_k)
            
            hpo = AspireHPO([{"name": "alpha", "type": "float", "range": "1.0 1000000.0", "log": True}], 
                            n_trials=n_trials, patience=15)
            best_params, _ = hpo.search(objective, study_name=f"Ablation_{name}")
            
            # --- Full Evaluation ---
            final_h = AspireEngine.apply_filter(s_np, best_params["alpha"], beta)
            metrics = full_evaluation(XV_t[val_u_t], torch.from_numpy(final_h), V.cpu(), val_gt, val_hist, item_pops, eval_cfg, device)
            
            metrics.update({
                "method": name,
                "beta": float(beta),
                "alpha": float(best_params["alpha"])
            })
            detailed_results.append(metrics)
        except Exception as e:
            print(f"  [Error] Method {name} failed: {e}")

    # --- Beta-HPO ---
    print("\n--- Testing Method: Beta-HPO (Simultaneous) ---")
    try:
        def objective_hpo(params):
            return fast_ndcg_obj(S_t, params["alpha"], params["beta"], main_k)
        
        hpo_spec = [
            {"name": "alpha", "type": "float", "range": "1.0 1000000.0", "log": True},
            {"name": "beta",  "type": "float", "range": "0.0 2.0"}
        ]
        hpo = AspireHPO(hpo_spec, n_trials=n_trials * 2, patience=20)
        best_params, _ = hpo.search(objective_hpo, study_name="Ablation_BetaHPO")
        
        final_h = AspireEngine.apply_filter(s_np, best_params["alpha"], best_params["beta"])
        metrics = full_evaluation(XV_t[val_u_t], torch.from_numpy(final_h), V.cpu(), val_gt, val_hist, item_pops, eval_cfg, device)
        
        metrics.update({
            "method": "Beta-HPO",
            "beta": float(best_params["beta"]),
            "alpha": float(best_params["alpha"])
        })
        detailed_results.append(metrics)
    except Exception as e:
        print(f"  [Error] Beta-HPO failed: {e}")

    # 4. Save and Plot
    df = pd.DataFrame(detailed_results)
    df.to_csv(os.path.join(out_dir, "results_detailed.csv"), index=False)
    with open(os.path.join(out_dir, "results.json"), 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=4)
    
    # Plotting Grouped Bar Charts
    top_ks = eval_cfg.get('top_k', [10, 20, 50])
    plot_metrics = ["NDCG", "Recall", "Coverage", "LongTailCoverage"]
    
    plt.figure(figsize=(18, 12))
    for i, m_name in enumerate(plot_metrics):
        plt.subplot(2, 2, i + 1)
        
        # Prepare data for grouped bar
        data = []
        for k in top_ks:
            col = f"{m_name}@{k}"
            if col in df.columns:
                data.append(df[col].values)
        
        if not data: continue
        
        x = np.arange(len(df["method"]))
        width = 0.8 / len(top_ks)
        
        for j, k in enumerate(top_ks):
            plt.bar(x + j*width, df[f"{m_name}@{k}"], width, label=f"@{k}", alpha=0.8)
            
        plt.title(f"{m_name} Comparison ({dataset_name})")
        plt.ylabel(m_name)
        plt.xticks(x + width*(len(top_ks)-1)/2, df["method"], rotation=45)
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "method_comparison_multi_k.png"), dpi=150)
    plt.close()
    
    print(f"\n[Done] Multi-metric ablation completed. Results saved to {out_dir}")
    return detailed_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml100k")
    parser.add_argument("--energy", type=float, default=0.99)
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    
    run_method_ablation(args.dataset, args.energy, args.trials)
