import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from tqdm import tqdm

# Framework root path
sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import load_config, ensure_dir, get_eval_config
from aspire_experiments.proof_models import ASPIRE_Test
from src.data_loader import DataLoader
from src.evaluation import evaluate_metrics
from src.models import get_model

def run_exp5(dataset_name, quick=False):
    print(f"Running Exp 5 Visualizations on {dataset_name}...")
    config = load_config(dataset_name)
    loader = DataLoader(config)
    
    # 1. Parameter Ranges
    if quick:
        gammas = [0.0, 0.2, 0.5, 1.0, 2.0]
        wbetas = [0.0, 0.4, 0.8]
    else:
        gammas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]
        wbetas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

    results = {"ASPIRE": [], "IPS-LAE": [], "EASE": None}
    metrics_to_request = ["NDCG", "Recall", "Coverage", "HeadNDCG", "LongTailNDCG"]
    eval_cfg = get_eval_config(loader, {"top_k": [20], "metrics": metrics_to_request})
    
    batch_size = config.get('train', {}).get('batch_size', 1024)
    test_loader = loader.get_final_loader(batch_size)

    # --- ASPIRE Sweep ---
    print(f"  Sweeping ASPIRE gammas: {gammas}")
    for g in tqdm(gammas):
        m_cfg = {"name": "aspire_test", "gamma": g, "filter_mode": "gamma_only", "target_energy": 1.0}
        cfg = config.copy()
        cfg['device'] = 'auto'
        cfg['model'] = m_cfg
        model = ASPIRE_Test(cfg, loader)
        metrics = evaluate_metrics(model, loader, eval_cfg, model.device, test_loader)
        results["ASPIRE"].append({
            "param": g,
            "NDCG@20": metrics["NDCG@20"],
            "Coverage@20": metrics["Coverage@20"],
            "LongTailNDCG@20": metrics["LongTailNDCG@20"],
            "HeadNDCG@20": metrics["HeadNDCG@20"]
        })

    # --- IPS-LAE Sweep ---
    print(f"  Sweeping IPS-LAE wbetas: {wbetas}")
    for wb in tqdm(wbetas):
        m_cfg = {"name": "ips_lae", "backbone": "ease", "wtype": "powerlaw", "wbeta": wb, "reg_lambda": 500.0}
        cfg = config.copy()
        cfg['device'] = 'auto'
        cfg['model'] = m_cfg
        model = get_model("ips_lae", cfg, loader)
        model.fit(loader)
        metrics = evaluate_metrics(model, loader, eval_cfg, model.device, test_loader)
        results["IPS-LAE"].append({
            "param": wb,
            "NDCG@20": metrics["NDCG@20"],
            "Coverage@20": metrics["Coverage@20"],
            "LongTailNDCG@20": metrics["LongTailNDCG@20"],
            "HeadNDCG@20": metrics["HeadNDCG@20"]
        })

    # --- EASE Baseline ---
    print("  Evaluating EASE baseline...")
    from aspire_experiments.proof_models import EASE_Test
    m_cfg = {"name": "ease_test", "alpha": 100.0}
    cfg = config.copy()
    cfg['device'] = 'auto'
    cfg['model'] = m_cfg
    model = EASE_Test(cfg, loader)
    metrics = evaluate_metrics(model, loader, eval_cfg, model.device, test_loader)
    results["EASE"] = {
        "NDCG@20": metrics["NDCG@20"],
        "Coverage@20": metrics["Coverage@20"],
        "LongTailNDCG@20": metrics["LongTailNDCG@20"],
        "HeadNDCG@20": metrics["HeadNDCG@20"]
    }

    # --- Save Results ---
    out_dir = ensure_dir(f"aspire_experiments/output/exp5/{dataset_name}")
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # --- Plotting 1: Trade-off Curve ---
    plt.figure(figsize=(8, 6))
    
    # ASPIRE Curve
    asp_cov = [r["Coverage@20"] for r in results["ASPIRE"]]
    asp_ndcg = [r["NDCG@20"] for r in results["ASPIRE"]]
    plt.plot(asp_cov, asp_ndcg, 'o-', label="ASPIRE ($\gamma$ sweep)", markersize=8)
    for i, g in enumerate(gammas):
        plt.annotate(f"{g}", (asp_cov[i], asp_ndcg[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    # IPS-LAE Curve
    ips_cov = [r["Coverage@20"] for r in results["IPS-LAE"]]
    ips_ndcg = [r["NDCG@20"] for r in results["IPS-LAE"]]
    plt.plot(ips_cov, ips_ndcg, 's--', label="IPS-LAE ($w_\\beta$ sweep)", markersize=7, alpha=0.7)
    
    # EASE Point
    plt.scatter([results["EASE"]["Coverage@20"]], [results["EASE"]["NDCG@20"]], c='red', marker='*', s=150, label="EASE baseline", zorder=5)

    plt.xlabel("Coverage@20")
    plt.ylabel("NDCG@20")
    plt.title(f"Accuracy-Coverage Trade-off: {dataset_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "tradeoff_curve.png"), dpi=150)
    plt.close()

    # --- Plotting 2: Head vs Tail Comparison ---
    # Pick a "Balanced" ASPIRE (e.g., gamma=0.5 or 0.2)
    plt.figure(figsize=(10, 6))
    
    labels = ["EASE", "ASPIRE ($\gamma=0.2$)", "IPS-LAE ($w_\\beta=0.4$)"]
    
    # Find closest results
    asp_balanced = next((r for r in results["ASPIRE"] if abs(r["param"] - 0.2) < 0.05), results["ASPIRE"][1])
    ips_balanced = next((r for r in results["IPS-LAE"] if abs(r["param"] - 0.4) < 0.05), results["IPS-LAE"][1])
    
    head_scores = [results["EASE"]["HeadNDCG@20"], asp_balanced["HeadNDCG@20"], ips_balanced["HeadNDCG@20"]]
    tail_scores = [results["EASE"]["LongTailNDCG@20"], asp_balanced["LongTailNDCG@20"], ips_balanced["LongTailNDCG@20"]]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, head_scores, width, label='Head NDCG@20', color='skyblue')
    plt.bar(x + width/2, tail_scores, width, label='Long-Tail NDCG@20', color='salmon')
    
    plt.ylabel('NDCG@20')
    plt.title(f'Popular vs Long-tail Performance: {dataset_name}')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.savefig(os.path.join(out_dir, "head_tail_ndcg.png"), dpi=150)
    plt.close()

    print(f"Exp 5 finished. Plots saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml100k", help="Dataset name")
    parser.add_argument("--quick", action="store_true", help="Quick mode with fewer parameters")
    args = parser.parse_args()
    
    run_exp5(args.dataset, quick=args.quick)
