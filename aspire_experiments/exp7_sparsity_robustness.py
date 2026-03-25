import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import pandas as pd
from tqdm import tqdm

# Framework root path
sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import load_config, ensure_dir, get_eval_config
from aspire_experiments.proof_models import ASPIRE_Test, EASE_Test
from src.data_loader import DataLoader
from src.evaluation import evaluate_metrics
from src.models import get_model

def run_exp7(dataset_name, ratios=[0.0, 0.2, 0.4, 0.6], k=None):
    print(f"Running Exp 7: Sparsity Robustness on {dataset_name}...")
    config = load_config(dataset_name)
    loader = DataLoader(config)
    eval_cfg = get_eval_config(loader, {"top_k": [20], "metrics": ["NDCG"]})
    test_loader = loader.get_final_loader(batch_size=1024)
    
    orig_train_df = loader.train_df.copy()
    
    # Model configs with assumed "best" params for ml100k (from prior runs)
    model_configs = {
        "EASE": {"name": "ease_test", "alpha": 100.0},
        "ASPIRE (\u03b3=0.1)": {"name": "aspire_test", "gamma": 0.1, "k": k, "filter_mode": "gamma_only", "target_energy": 1.0},
        "IPS-LAE": {"name": "ips_lae", "backbone": "ease", "wtype": "powerlaw", "wbeta": 0.4, "reg_lambda": 500.0}
    }
    
    results = {name: [] for name in model_configs.keys()}

    for ratio in ratios:
        print(f"\n  [Ratio {ratio:.1f}] Masking {ratio*100:.0f}% of interactions (with Protection)...")
        
        # 1. Protected Masking Logic
        if ratio > 0:
            # First, pick 1 random interaction for each user and item to keep
            u_keep = orig_train_df.groupby('user_id').sample(n=1, random_state=42)
            i_keep = orig_train_df.groupby('item_id').sample(n=1, random_state=42)
            protected_idx = pd.Index(list(set(u_keep.index) | set(i_keep.index)))
            
            # Remaining indices that can be masked
            maskable_df = orig_train_df.drop(protected_idx)
            
            # How many more do we need to mask to reach total ratio?
            total_to_keep = int(len(orig_train_df) * (1.0 - ratio))
            still_needed = max(0, total_to_keep - len(protected_idx))
            
            if still_needed > 0 and len(maskable_df) > 0:
                sampled_maskable = maskable_df.sample(n=min(still_needed, len(maskable_df)), random_state=42)
                masked_df = pd.concat([orig_train_df.loc[protected_idx], sampled_maskable])
            else:
                masked_df = orig_train_df.loc[protected_idx]
        else:
            masked_df = orig_train_df
            
        # 2. Update Loader's train_df (temporarily)
        loader.train_df = masked_df
        
        for name, m_cfg in model_configs.items():
            print(f"    Training {name}...")
            cfg = {**config, 'model': m_cfg, 'device': 'auto'}
            
            if m_cfg["name"] == "ease_test":
                model = EASE_Test(cfg, loader)
            elif m_cfg["name"] == "ips_lae":
                model = get_model("ips_lae", cfg, loader)
                model.fit(loader)
            else:
                model = ASPIRE_Test(cfg, loader)
            
            metrics = evaluate_metrics(model, loader, eval_cfg, model.device, test_loader)
            results[name].append(metrics["NDCG@20"])
            print(f"      NDCG@20: {metrics['NDCG@20']:.4f}")

    # Restore loader
    loader.train_df = orig_train_df

    # 3. Plotting
    plt.figure(figsize=(10, 6))
    for name, scores in results.items():
        plt.plot(ratios, scores, marker='o', linewidth=2, label=name)
        
    plt.xlabel("Masking Ratio (Dropped interactions)")
    plt.ylabel("NDCG@20")
    plt.title(f"Sparsity Robustness: {dataset_name}\n(Performance vs. Data Availability)")
    plt.xticks(ratios)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_dir = ensure_dir(f"aspire_experiments/output/exp7/{dataset_name}")
    plt.savefig(os.path.join(out_dir, "sparsity_robustness_plt.png"), dpi=150)
    plt.close()
    
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"ratios": ratios, "k": k if k is not None else min(1024, loader.n_items), "results": results}, f, indent=4)
        
    print(f"\nExp 7 finished. Results saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml100k", help="Dataset name")
    parser.add_argument("--k", type=int, default=None, help="Rank k for ASPIRE")
    args = parser.parse_args()
    
    run_exp7(args.dataset, k=args.k)
