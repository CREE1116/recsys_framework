import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from scipy.sparse import csr_matrix

# Framework root path
sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import load_config, ensure_dir, get_eval_config
from aspire_experiments.proof_models import ASPIRE_Test
from src.data_loader import DataLoader
from src.evaluation import evaluate_metrics

def run_exp3(dataset_name, override_gamma=None, k=None):
    print(f"Running Exp 3 on {dataset_name}...")
    config = load_config(dataset_name)
    loader = DataLoader(config)
    
    # 1. Determine Gamma (Priority: CLI > Exp 2 results > default 1.0)
    gamma_to_use = 1.0
    if override_gamma is not None:
        gamma_to_use = override_gamma
        print(f"  [CLI] Using provided Gamma: {gamma_to_use:.4f}")
    else:
        exp2_path = f"aspire_experiments/output/exp2/{dataset_name}/results.json"
        if os.path.exists(exp2_path):
            with open(exp2_path, "r") as f:
                res2 = json.load(f)
                gamma_to_use = res2.get("best_gamma", 1.0)
                print(f"  [Exp2 Hit] Using Best Gamma from HPO: {gamma_to_use:.4f}")
        else:
            print(f"  [Default] Using fallback Gamma: 1.0")

    # Define models
    models_to_test = {
        f"ASPIRE (γ={gamma_to_use:.2f})": {"name": "aspire_test", "gamma": gamma_to_use, "k": config.get('k'), "filter_mode": "gamma_only", "target_energy": 1.0},
        "LIRA (Pure Wiener)": {"name": "aspire_test", "gamma": 2.0, "k": config.get('k'), "filter_mode": "gamma_only", "target_energy": 1.0},
        "EASE": {"name": "ease_test", "alpha": 138.76453930062902} 
    }
    
    item_popularity = np.array(loader.train_df.groupby('item_id').size().reindex(range(loader.n_items), fill_value=0))
    # Rank items (0 is most popular)
    ranks = np.argsort(np.argsort(-item_popularity))
    deciles = (ranks * 10 // loader.n_items)
    
    all_stats = {} # name -> {"proportion": [], "hit_rate": []}
    eval_cfg = get_eval_config(loader, {"top_k": [20]})
    
    # Pre-calculate ground truth per decile for all test users
    # test_gt_by_user = {u: set(items)}
    test_gt = loader.test_df.groupby('user_id')['item_id'].agg(set).to_dict()
    test_users_all = sorted(list(test_gt.keys()))
    
    for name, m_cfg in models_to_test.items():
        print(f"  Evaluating {name}...")
        cfg = {**config, 'model': m_cfg, 'device': 'auto'}
        
        if m_cfg["name"] == "ease_test":
            from aspire_experiments.proof_models import EASE_Test
            model = EASE_Test(cfg, loader)
        elif m_cfg["name"] == "ips_lae": # Using the registered ips_lae
            from src.models import get_model
            model = get_model("ips_lae", cfg, loader)
            model.fit(loader)
        else:
            model = ASPIRE_Test(cfg, loader)
        
        # Recommendation Frequency (using 5000 users or all)
        sample_users = test_users_all[:min(5000, len(test_users_all))]
        user_tensor = torch.LongTensor(sample_users).to(model.device)
        scores = model.forward(user_tensor)
        
        # Mask history
        train_rows = loader.train_df[loader.train_df['user_id'].isin(sample_users)]
        R_mask = csr_matrix((np.ones(len(train_rows)), (train_rows['user_id'].values, train_rows['item_id'].values)), 
                            shape=(len(sample_users), loader.n_items))
        # Ensure R_mask rows align with sample_users
        u_to_idx = {u: i for i, u in enumerate(sample_users)}
        rows_mapped = [u_to_idx[u] for u in train_rows['user_id'].values]
        R_mask_aligned = csr_matrix((np.ones(len(train_rows)), (rows_mapped, train_rows['item_id'].values)), 
                                    shape=(len(sample_users), loader.n_items))
        
        mask = torch.from_numpy(R_mask_aligned.toarray().astype(np.float32)).to(model.device)
        scores[mask > 0] = -1e10
        topk_indices = torch.topk(scores, k=20, dim=1).indices.cpu().numpy()
        
        # 1. Recommendation Proportion per Decile
        rec_freq = np.zeros(loader.n_items)
        unique, counts = np.unique(topk_indices, return_counts=True)
        rec_freq[unique] = counts
        decile_freq = [rec_freq[deciles == d].sum() for d in range(10)]
        decile_prop = np.array(decile_freq) / (np.sum(decile_freq) + 1e-12)
        
        # 2. HitRate@20 per Decile
        # Group test interactions by decile
        test_df = loader.test_df[loader.test_df['user_id'].isin(sample_users)]
        decile_hits = [0] * 10
        decile_counts = [0] * 10
        
        for idx, u_id in enumerate(sample_users):
            recs = set(topk_indices[idx])
            gt_items = test_gt.get(u_id, set())
            for item in gt_items:
                d = deciles[item]
                decile_counts[d] += 1
                if item in recs:
                    decile_hits[d] += 1
        
        decile_hr = [decile_hits[d] / (decile_counts[d] + 1e-12) for d in range(10)]
        all_stats[name] = {"proportion": decile_prop.tolist(), "hit_rate": decile_hr}

    # Plotting (1x2 Subplots)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(1, 11)
    width = 0.25
    
    for i, (name, stats) in enumerate(all_stats.items()):
        ax1.bar(x + (i-1)*width, stats["proportion"], width, label=name)
        ax2.plot(x, stats["hit_rate"], marker='o', label=name)
        
    ax1.set_xlabel("Item Popularity Decile (1=Most Popular, 10=Long-tail)")
    ax1.set_ylabel("Recommendation Frequency (Proportion)")
    ax1.set_title(f"Rec Frequency Distribution: {dataset_name}")
    ax1.set_xticks(x)
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)
    
    ax2.set_xlabel("Item Popularity Decile")
    ax2.set_ylabel("HitRate@20")
    ax2.set_title(f"HitRate@20 per Decile: {dataset_name}")
    ax2.set_xticks(x)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_dir = ensure_dir(f"aspire_experiments/output/exp3/{dataset_name}")
    plt.savefig(os.path.join(out_dir, "debiasing_quality_plot.png"), dpi=150)
    plt.close()
    
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(all_stats, f, indent=4)
    
    print(f"Exp 3 on {dataset_name} finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml1m", help="Dataset name")
    parser.add_argument("--gamma", type=float, default=None, help="Explicit gamma for ASPIRE (default: from Exp 2 results)")
    parser.add_argument("--k", type=int, default=None, help="Rank k for ASPIRE")
    args = parser.parse_args()
    
    run_exp3(args.dataset, override_gamma=args.gamma, k=args.k)
