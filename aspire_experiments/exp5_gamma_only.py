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

def run_gamma_only_hpo(dataset_name, n_trials=50):
    print(f"\n{'='*60}")
    print(f"EXP5: Gamma-only HPO for {dataset_name}")
    print(f"{'='*60}")
    
    loader, R, S, V, config = get_loader_and_svd(dataset_name, k=None)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    eval_cfg = copy.deepcopy(config.get('evaluation', {}))
    eval_cfg['metrics'] = ['NDCG', 'GiniIndex']
    eval_cfg['top_k'] = [20]
    val_loader = loader.get_validation_loader(2048)
    
    def objective(trial):
        gamma = trial.suggest_float("gamma", 0.3, 5.0)
        
        cfg = copy.deepcopy(config)
        cfg['device'] = device
        cfg['model']['name'] = 'aspire'
        cfg['model']['gamma'] = gamma
        cfg['model']['filter_mode'] = 'gamma_only'
        cfg['model']['target_energy'] = 1.0 # BEST와 동일 조건
        cfg['model']['visualize'] = False
        
        model = ASPIRE(cfg, loader)
        model.fit(loader)
        res = evaluate_metrics(model, loader, eval_cfg, device=device, test_loader=val_loader, is_final=False)
        return float(res['NDCG@20'])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    # Save results
    output_dir = ensure_dir(f"aspire_experiments/output/exp5/{dataset_name}")
    
    # Final evaluation with best gamma
    trial_best = study.best_trial
    best_res = {
        "dataset": dataset_name,
        "method": "gamma_only_aspire",
        "best_params": study.best_params,
        "best_val_NDCG@20": study.best_value,
        "hpo_trials": n_trials,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(best_res, f, indent=4)
        
    print(f"\n[DONE] Best Gamma for {dataset_name}: {study.best_params['gamma']:.4f}")
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
            run_gamma_only_hpo(ds)
        except Exception as e:
            print(f"Error running EXP5 on {ds}: {e}")
