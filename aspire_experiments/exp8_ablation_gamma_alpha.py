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

from aspire_experiments.exp_utils import load_config, ensure_dir, AspireHPO, get_eval_config
from aspire_experiments.exp2_gamma_skewness import estimate_zeta
from aspire_experiments.proof_models import ASPIRE_Test
from src.data_loader import DataLoader
from src.evaluation import evaluate_metrics

def run_exp8(dataset_name, n_trials=30, k=None):
    print(f"Running Exp 8: Gamma-Alpha Ablation on {dataset_name}...")
    config = load_config(dataset_name)
    loader = DataLoader(config)
    min_dim = min(loader.n_users, loader.n_items)
    eval_cfg = get_eval_config(loader, {"top_k": [20], "metrics": ["NDCG"]})
    valid_loader = loader.get_validation_loader(batch_size=2048)
    test_loader = loader.get_final_loader(batch_size=2048)
    
    # 1. Theoretical Gamma
    zeta = estimate_zeta(loader)
    gamma_theory = 2.0 - zeta
    eff_k = k if k is not None else min(10000, loader.n_items)
    default_alpha = 1.0
    
    print(f"  [Theory] Gamma: {gamma_theory:.4f}, Alpha (Default): {default_alpha}")

    # Helper for evaluation
    def evaluate(gamma, alpha, loader_set="test", k=None):
        m_cfg = {
            'name': 'aspire_test', 
            'gamma': gamma, 
            'alpha': alpha, 
            'k': k,
            'filter_mode': 'gamma_only', 
            'target_energy': 1.0
        }
        cfg = {**config, 'model': m_cfg, 'device': 'auto'}
        model = ASPIRE_Test(cfg, loader)
        ldr = test_loader if loader_set == "test" else valid_loader
        metrics = evaluate_metrics(model, loader, eval_cfg, model.device, ldr)
        return metrics["NDCG@20"]

    # Determine tail items (Bottom 50% popularity)
    pop = loader.item_popularity
    tail_threshold = np.median(pop)
    tail_items = np.where(pop <= tail_threshold)[0]
    
    eval_cfg_full = get_eval_config(loader, {"top_k": [20], "metrics": ["NDCG", "Coverage"]})

    # Helper for evaluation
    def evaluate(gamma, alpha, loader_set="test", k=None):
        m_cfg = {
            'name': 'aspire_test', 
            'gamma': gamma, 
            'alpha': alpha, 
            'k': k,
            'filter_mode': 'gamma_only', 
            'target_energy': 1.0
        }
        cfg = {**config, 'model': m_cfg, 'device': 'auto'}
        model = ASPIRE_Test(cfg, loader)
        ldr = test_loader if loader_set == "test" else valid_loader
        
        # Standard metrics
        metrics = evaluate_metrics(model, loader, eval_cfg_full, model.device, ldr)
        
        # Tail NDCG calculation (custom)
        # We'll sample 2000 users for speed or use all
        test_users = torch.arange(loader.n_users).to(model.device)
        scores = model.forward(test_users).detach() # Added detach for memory safety
        
        # Mask train history speed/memory efficient way
        rows = torch.from_numpy(loader.train_df['user_id'].values).to(model.device)
        cols = torch.from_numpy(loader.train_df['item_id'].values).to(model.device)
        
        # Safety bound check
        valid_mask = (rows < scores.shape[0]) & (cols < scores.shape[1])
        scores[rows[valid_mask], cols[valid_mask]] = -1e10
        
        topk = torch.topk(scores, k=20, dim=1).indices.cpu().numpy()
        
        # Tail NDCG
        test_gt = loader.test_df.groupby('user_id')['item_id'].apply(list).to_dict()
        tail_ndcgs = []
        for u in range(loader.n_users):
            gt = test_gt.get(u, [])
            if not gt: continue
            rec = topk[u]
            # DCG for tail items only
            gt_tail = [i for i in gt if i in tail_items]
            if not gt_tail: continue
            
            dcg = 0
            idcg = 0
            for rank, item in enumerate(rec):
                if item in gt_tail:
                    dcg += 1.0 / np.log2(rank + 2)
            
            for rank in range(min(len(gt_tail), 20)):
                idcg += 1.0 / np.log2(rank + 2)
            
            tail_ndcgs.append(dcg / idcg)
            
        tail_ndcg = np.mean(tail_ndcgs) if tail_ndcgs else 0.0
        
        return {
            "NDCG@20": metrics["NDCG@20"],
            "Coverage@20": metrics["Coverage@20"],
            "Tail-NDCG@20": tail_ndcg
        }

    modes = [
        ("V1 (Theory G, Def A)", gamma_theory, default_alpha),
        ("V2 (HPO G, Def A)", None, default_alpha), # To be filled
        ("V3 (Theory G, HPO A)", gamma_theory, None), # To be filled
        ("V4 (HPO Both)", None, None) # To be filled
    ]
    
    final_results = {}

    # Sequential HPOs and evaluations
    # V1 (Theory G, Def A)
    print("  Evaluating V1 (Theory G, Def A)...")
    final_results["V1"] = evaluate(gamma_theory, default_alpha)
    final_results["V1"]["gamma"] = float(gamma_theory)
    final_results["V1"]["alpha"] = float(default_alpha)
    final_results["V1"]["k"] = int(eff_k)

    # V2 (HPO G + K, Def A)
    print("  Evaluating V2 (HPO Gamma + Rank)...")
    hpo_g = AspireHPO([
        {'name': 'gamma', 'type': 'float', 'range': '0.0 2.0'},
        {'name': 'k', 'type': 'int_min_dim', 'log': True}
    ], n_trials=n_trials, patience=20, min_dim=min_dim)
    best_g_params, _ = hpo_g.search(lambda p: evaluate(p['gamma'], default_alpha, "valid", k=p['k'])["NDCG@20"], study_name=f"Exp8_G_{dataset_name}")
    final_results["V2"] = evaluate(best_g_params['gamma'], default_alpha, k=best_g_params['k'])
    final_results["V2"]["gamma"] = float(best_g_params['gamma'])
    final_results["V2"]["alpha"] = float(default_alpha)
    final_results["V2"]["k"] = int(best_g_params['k'])

    # V3 (Theory G, HPO A + K)
    print("  Evaluating V3 (HPO Alpha + Rank)...")
    hpo_a = AspireHPO([
        {'name': 'alpha', 'type': 'float', 'range': '1e-3 1e3', 'log': True},
        {'name': 'k', 'type': 'int_min_dim', 'log': True}
    ], n_trials=n_trials, patience=20, min_dim=min_dim)
    best_a_params, _ = hpo_a.search(lambda p: evaluate(gamma_theory, p['alpha'], "valid", k=p['k'])["NDCG@20"], study_name=f"Exp8_A_{dataset_name}")
    final_results["V3"] = evaluate(gamma_theory, best_a_params['alpha'], k=best_a_params['k'])
    final_results["V3"]["gamma"] = float(gamma_theory)
    final_results["V3"]["alpha"] = float(best_a_params['alpha'])
    final_results["V3"]["k"] = int(best_a_params['k'])

    # V4 (Joint HPO Both + Rank k)
    print("  Evaluating V4 (Joint HPO Both + Rank k)...")
    hpo_joint = AspireHPO([
        {'name': 'gamma', 'type': 'float', 'range': '0.0 2.0'},
        {'name': 'alpha', 'type': 'float', 'range': '1e-3 1e3', 'log': True},
        {'name': 'k',     'type': 'int_min_dim', 'log': True}
    ], n_trials=max(n_trials * 2, 30), patience=30, min_dim=min_dim)
    
    def objective_v4(p):
        res = evaluate(p['gamma'], p['alpha'], "valid", k=p['k'])
        return res["NDCG@20"]

    best_v4_params, _ = hpo_joint.search(objective_v4, study_name=f"Exp8_Joint_{dataset_name}")
    
    final_results["V4"] = evaluate(best_v4_params['gamma'], best_v4_params['alpha'], "test", k=best_v4_params['k'])
    final_results["V4"]["gamma"] = float(best_v4_params['gamma'])
    final_results["V4"]["alpha"] = float(best_v4_params['alpha'])
    final_results["V4"]["k"] = int(best_v4_params['k'])

    # --- Plotting ---
    descriptions = {
        "V1": "Theory G, Default A",
        "V2": "HPO G, Default A",
        "V3": "Theory G, HPO A",
        "V4": "Joint HPO Both"
    }
    
    v1_label = f"V1: {descriptions['V1']}\n(G={gamma_theory:.2f}, A=1)"
    v2_label = f"V2: {descriptions['V2']}\n(G={final_results['V2']['gamma']:.2f}, A=1)"
    v3_label = f"V3: {descriptions['V3']}\n(G={gamma_theory:.2f}, A={final_results['V3']['alpha']:.1f})"
    v4_label = f"V4: {descriptions['V4']}\n(G={final_results['V4']['gamma']:.2f}, A={final_results['V4']['alpha']:.1f})"
    
    labels = [v1_label, v2_label, v3_label, v4_label]
    ndcg = [final_results["V1"]["NDCG@20"], final_results["V2"]["NDCG@20"], 
            final_results["V3"]["NDCG@20"], final_results["V4"]["NDCG@20"]]
    cov = [final_results["V1"]["Coverage@20"], final_results["V2"]["Coverage@20"], 
           final_results["V3"]["Coverage@20"], final_results["V4"]["Coverage@20"]]
    tail_ndcg = [final_results["V1"]["Tail-NDCG@20"], final_results["V2"]["Tail-NDCG@20"], 
                 final_results["V3"]["Tail-NDCG@20"], final_results["V4"]["Tail-NDCG@20"]]
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.bar(x - width, ndcg, width, label='Overall NDCG@20', color='#1f77b4')
    ax.bar(x, cov, width, label='Coverage@20', color='#ff7f0e', alpha=0.7)
    ax.bar(x + width, tail_ndcg, width, label='Tail NDCG@20', color='#2ca02c')
    
    ax.set_ylabel('Scores')
    ax.set_title(f'Ablation Study: Gamma & Alpha ({dataset_name})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add explanation text box
    explanation = (
        "Ablation Modes:\n"
        "- V1: No tuning, uses theoretical gamma (Bridge Lemma)\n"
        "- V2: Tune gamma only, keep default alpha=1.0\n"
        "- V3: Keep theoretical gamma, tune alpha only\n"
        "- V4: Jointly tune both gamma and alpha"
    )
    plt.annotate(explanation, xy=(0.02, 0.98), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                 fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    out_dir = ensure_dir(f"aspire_experiments/output/exp8/{dataset_name}")
    plt.savefig(os.path.join(out_dir, "ablation_metrics_plot.png"), dpi=200)
    plt.close()
    
    # Save descriptive results
    final_output = {
        "dataset": dataset_name,
        "gamma_theory": gamma_theory,
        "ablation_modes": descriptions,
        "results": final_results
    }
    
    with open(os.path.join(out_dir, "ablation_results.json"), "w") as f:
        json.dump(final_output, f, indent=4)
        
    print(f"\nExp 8 finished on {dataset_name}. Plot saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml100k")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--k", type=int, default=None, help="Rank k for ASPIRE")
    args = parser.parse_args()
    run_exp8(args.dataset, n_trials=args.trials, k=args.k)
