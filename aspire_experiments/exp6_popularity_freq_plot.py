import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from tqdm import tqdm
from scipy.sparse import csr_matrix

# Framework root path
sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import load_config, ensure_dir, get_eval_config, AspireHPO
from aspire_experiments.proof_models import ASPIRE_Test, EASE_Test
from src.data_loader import DataLoader
from src.evaluation import evaluate_metrics
from src.models import get_model

def get_rec_freq(model, loader, top_k=100):
    model.eval()
    batch_size = 1024
    test_users = loader.test_df['user_id'].unique()
    rec_counts = np.zeros(loader.n_items)
    
    # Masking matrix
    history = loader.eval_user_history
    
    with torch.no_grad():
        for i in tqdm(range(0, len(test_users), batch_size), desc="Inferencing", leave=False):
            u_batch_np = test_users[i:i+batch_size]
            u_batch = torch.LongTensor(u_batch_np).to(model.device)
            scores = model.forward(u_batch)
            
            # Mask history
            for idx, u_id in enumerate(u_batch_np):
                seen = list(history.get(u_id, []))
                if seen:
                    scores[idx, seen] = -1e10
            
            _, top_items = torch.topk(scores, k=top_k, dim=1)
            unique, counts = torch.unique(top_items, return_counts=True)
            rec_counts[unique.cpu().numpy()] += counts.cpu().numpy()
            
    return rec_counts

def run_exp6(dataset_name, n_trials=20):
    print(f"Running Exp 6: Popularity vs. Rec Frequency on {dataset_name}")
    config = load_config(dataset_name)
    loader = DataLoader(config)
    eval_cfg = get_eval_config(loader, {"top_k": [20], "metrics": ["NDCG"]})
    test_loader = loader.get_final_loader(batch_size=1024)
    
    item_pop = loader.item_popularity
    results = {}

    # --- 1. EASE HPO ---
    print("\n[EASE] Running HPO...")
    def objective_ease(params):
        m_cfg = {"name": "ease_test", "alpha": params['reg_lambda']}
        cfg = config.copy(); cfg['model'] = m_cfg; cfg['device'] = 'auto'
        model = EASE_Test(cfg, loader)
        metrics = evaluate_metrics(model, loader, eval_cfg, model.device, test_loader)
        return metrics["NDCG@20"]

    hpo_ease = AspireHPO([{'name': 'reg_lambda', 'type': 'float', 'range': '10.0 100000.0', 'log': True}], n_trials=n_trials)
    best_ease_params, _ = hpo_ease.search(objective_ease)
    
    best_ease_model = EASE_Test({**config, 'model': {'name': 'ease_test', 'alpha': best_ease_params['reg_lambda']}, 'device': 'auto'}, loader)
    results["EASE"] = get_rec_freq(best_ease_model, loader)

    # --- 2. ASPIRE HPO ---
    print("\n[ASPIRE] Running HPO...")
    def objective_aspire(params):
        m_cfg = {"name": "aspire_test", "gamma": params['gamma'], "filter_mode": "gamma_only", "target_energy": 1.0}
        cfg = config.copy(); cfg['model'] = m_cfg; cfg['device'] = 'auto'
        model = ASPIRE_Test(cfg, loader)
        metrics = evaluate_metrics(model, loader, eval_cfg, model.device, test_loader)
        return metrics["NDCG@20"]

    hpo_aspire = AspireHPO([{'name': 'gamma', 'type': 'float', 'range': '0.0 2.0'}], n_trials=n_trials)
    best_asp_params, _ = hpo_aspire.search(objective_aspire)
    
    best_asp_model = ASPIRE_Test({**config, 'model': {'name': 'aspire_test', 'gamma': best_asp_params['gamma'], 'filter_mode': 'gamma_only', 'target_energy': 1.0}, 'device': 'auto'}, loader)
    results["ASPIRE"] = get_rec_freq(best_asp_model, loader)

    # --- 3. IPS-LAE HPO ---
    print("\n[IPS-LAE] Running HPO...")
    def objective_ips(params):
        m_cfg = {"name": "ips_lae", "backbone": "ease", "wtype": "powerlaw", "wbeta": params['wbeta'], "reg_lambda": params['reg_lambda']}
        cfg = config.copy(); cfg['model'] = m_cfg; cfg['device'] = 'auto'
        model = get_model("ips_lae", cfg, loader)
        model.fit(loader)
        metrics = evaluate_metrics(model, loader, eval_cfg, model.device, test_loader)
        return metrics["NDCG@20"]

    hpo_ips = AspireHPO([
        {'name': 'reg_lambda', 'type': 'float', 'range': '10.0 100000.0', 'log': True},
        {'name': 'wbeta', 'type': 'float', 'range': '0.0 1.0'}
    ], n_trials=n_trials)
    best_ips_params, _ = hpo_ips.search(objective_ips)
    
    best_ips_model = get_model("ips_lae", {**config, 'model': {**best_ips_params, 'name': 'ips_lae', 'backbone': 'ease', 'wtype': 'powerlaw'}, 'device': 'auto'}, loader)
    best_ips_model.fit(loader)
    results["IPS-LAE"] = get_rec_freq(best_ips_model, loader)

    # --- 4. Plotting (Separate Files) ---
    out_dir = ensure_dir(f"aspire_experiments/output/exp6/{dataset_name}")
    
    valid_idx = item_pop > 0
    pop_vals = item_pop[valid_idx]
    
    # Create log-spaced bins for popularity
    min_pop, max_pop = pop_vals.min(), pop_vals.max()
    bins = np.logspace(np.log10(min_pop), np.log10(max_pop), 15)
    bin_centers = (bins[:-1] * bins[1:])**0.5 # Geometric center
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    model_names = list(results.keys())
    
    bin_means_all = {}
    
    for idx, name in enumerate(model_names):
        freqs = results[name]
        y = freqs[valid_idx]
        
        # Calculate mean frequency per bin
        bin_means = []
        for i in range(len(bins)-1):
            mask = (pop_vals >= bins[i]) & (pop_vals < bins[1+i])
            bin_means.append(y[mask].mean() if mask.any() else 0)
        bin_means_all[name] = bin_means
        
        # Individual plot
        plt.figure(figsize=(8, 6))
        plt.bar(bin_centers, bin_means, width=np.diff(bins)*0.6, alpha=0.7, color=colors[idx])
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f"Rec Frequency vs Popularity: {name}")
        plt.xlabel("Popularity (Training Frequency)")
        plt.ylabel("Avg. Rec Frequency (Top-100)")
        plt.grid(True, alpha=0.2)
        safe_name = name.split()[0].lower()
        plt.savefig(os.path.join(out_dir, f"pop_freq_{safe_name}.png"), dpi=150)
        plt.close()

    # Combined Plot (Line-only)
    plt.figure(figsize=(10, 6))
    for idx, name in enumerate(model_names):
        plt.plot(bin_centers, bin_means_all[name], marker='o', label=name, color=colors[idx], linewidth=2)
        
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"Popularity vs. Rec Frequency Comparison: {dataset_name}")
    plt.xlabel("Popularity")
    plt.ylabel("Avg. Rec Frequency")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(out_dir, "pop_freq_combined.png"), dpi=150)
    plt.close()
    
    # Save best params
    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump({
            "EASE": best_ease_params,
            "ASPIRE": best_asp_params,
            "IPS-LAE": best_ips_params
        }, f, indent=4)

    print(f"\nExp 6 finished. Plot saved to {out_dir}/popularity_freq_plot.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml100k", help="Dataset name")
    parser.add_argument("--trials", type=int, default=20, help="Number of HPO trials")
    args = parser.parse_args()
    
    run_exp6(args.dataset, n_trials=args.trials)
