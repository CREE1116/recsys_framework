import os
import sys
import torch
import optuna
import copy
import json
import numpy as np
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import get_loader_and_svd, ensure_dir
from src.models.csar.ASPIRE import ASPIRE
from src.evaluation import evaluate_metrics

def run_tau_gamma_hpo(dataset_name, n_trials=60):
    print(f"\n{'='*60}")
    print(f"EXP6: Tau-Gamma HPO for {dataset_name}")
    print(f"{'='*60}")
    
    loader, R, S, V, config = get_loader_and_svd(dataset_name, k=None)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    eval_cfg = copy.deepcopy(config.get('evaluation', {}))
    eval_cfg['metrics'] = ['NDCG']
    eval_cfg['top_k'] = [20]
    val_loader = loader.get_validation_loader(2048)
    
    def objective(trial):
        # Expanded ranges for more robust exploration
        gamma = trial.suggest_float("gamma", 0.1, 3.0)
        # Tau: Relative scaling to peak power (s1^gamma)
        # 10.0 means strong regularization (cutoff at very high s)
        # Narrowed lower bound to 0.01 based on user feedback
        tau = trial.suggest_float("tau", 0.01, 10.0, log=True)
        
        cfg = copy.deepcopy(config)
        cfg['device'] = device
        cfg['model']['name'] = 'aspire'
        cfg['model']['gamma'] = gamma
        cfg['model']['tau'] = tau
        cfg['model']['filter_mode'] = 'gamma_only' # Relative peak logic
        cfg['model']['target_energy'] = 1.0
        cfg['model']['visualize'] = False
        
        model = ASPIRE(cfg, loader)
        model.fit(loader)
        res = evaluate_metrics(model, loader, eval_cfg, device=device, test_loader=val_loader, is_final=False)
        return float(res['NDCG@20'])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    # Save results
    output_dir = ensure_dir(f"aspire_experiments/output/exp6/{dataset_name}")
    best_res = {
        "dataset": dataset_name,
        "method": "tau_gamma_relative_peak",
        "best_params": study.best_params,
        "best_val_NDCG@20": study.best_value,
        "hpo_trials": n_trials,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(best_res, f, indent=4)
        
    print(f"\n[DONE] {dataset_name} | Best Gamma: {study.best_params['gamma']:.4f}, Tau: {study.best_params['tau']:.6f}")
    print(f"Validation NDCG@20: {study.best_value:.6f}")
    return best_res

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml100k,ml1m")
    args = parser.parse_args()
    
    datasets = [d.strip() for d in args.dataset.split(",")]
    for ds in datasets:
        try:
            run_tau_gamma_hpo(ds)
        except Exception as e:
            print(f"Error running EXP6 on {ds}: {e}")
