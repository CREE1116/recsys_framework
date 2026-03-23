import os
import sys
import torch
import optuna
import copy
import time
import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path
sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import get_loader_and_svd
from src.models.csar.ASPIRE import ASPIRE
from src.models.csar.ASPIRELayer import AspireFilter
from src.evaluation import evaluate_metrics

def apply_filter_v1(vals, alpha, gamma, is_gram=False):
    # Method 1: Include s1 in mean
    exp = float(gamma) if not is_gram else float(gamma) / 2.0
    s_gamma = torch.pow(torch.clamp(vals.float(), min=1e-12), exp)
    mean_val = s_gamma.mean().item()
    effective_lambda = float(alpha) * mean_val
    h = s_gamma / (s_gamma + effective_lambda + 1e-10)
    return h.float(), float(alpha), float(effective_lambda)

def apply_filter_v2(vals, alpha, gamma, is_gram=False):
    # Method 2: Exclude s1 in mean (current)
    exp = float(gamma) if not is_gram else float(gamma) / 2.0
    s_gamma = torch.pow(torch.clamp(vals.float(), min=1e-12), exp)
    if len(s_gamma) > 1:
        mean_val = s_gamma[1:].mean().item()
    else:
        mean_val = s_gamma.mean().item()
    effective_lambda = float(alpha) * mean_val
    h = s_gamma / (s_gamma + effective_lambda + 1e-10)
    return h.float(), float(alpha), float(effective_lambda)

def run_multi_seed_test(method_name, apply_filter_fn, n_seeds=5, n_trials=25):
    print(f"\n{'='*60}")
    print(f"RUNNING MULTI-SEED TEST: {method_name}")
    print(f"{'='*60}")
    
    # Monkey patch AspireFilter.apply_filter
    original_apply = AspireFilter.apply_filter
    AspireFilter.apply_filter = staticmethod(apply_filter_fn)
    
    loader, R, S, V, config = get_loader_and_svd("ml100k", k=None)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    eval_cfg = copy.deepcopy(config.get('evaluation', {}))
    eval_cfg['metrics'] = ['NDCG']
    eval_cfg['top_k'] = [20]
    val_loader = loader.get_validation_loader(2048)
    
    all_seed_best_values = []
    all_seed_curves = [] # (n_seeds, n_trials)
    
    for seed in range(n_seeds):
        print(f"\n[Seed {seed}] Optimization starting...")
        trial_scores = []
        
        def objective(trial):
            gamma = trial.suggest_float("gamma", 0.3, 2.0)
            alpha = trial.suggest_float("alpha", 0.0001, 2.0, log=True)
            
            cfg = copy.deepcopy(config)
            cfg['device'] = device
            cfg['model']['name'] = 'aspire'
            cfg['model']['gamma'] = gamma
            cfg['model']['alpha'] = alpha
            cfg['model']['target_energy'] = 1.0
            cfg['model']['visualize'] = False
            
            model = ASPIRE(cfg, loader)
            model.fit(loader)
            res = evaluate_metrics(model, loader, eval_cfg, device=device, test_loader=val_loader, is_final=False)
            score = float(res['NDCG@20'])
            trial_scores.append(score)
            return score

        # Use fixed seed TPE sampler for reproducibility within multi-seed test
        from optuna.samplers import TPESampler
        sampler = TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials)
        
        all_seed_best_values.append(study.best_value)
        all_seed_curves.append(np.maximum.accumulate(trial_scores))
        print(f"[Seed {seed}] Best NDCG: {study.best_value:.6f}")

    # Restore original function
    AspireFilter.apply_filter = original_apply
    
    return np.array(all_seed_curves), np.array(all_seed_best_values)

if __name__ == "__main__":
    n_seeds = 5
    n_trials = 30
    
    curves1, bests1 = run_multi_seed_test("Include s1", apply_filter_v1, n_seeds=n_seeds, n_trials=n_trials)
    curves2, bests2 = run_multi_seed_test("Exclude s1", apply_filter_v2, n_seeds=n_seeds, n_trials=n_trials)
    
    print("\n" + "="*60)
    print(f"FINAL STABILITY SUMMARY (ml100k, {n_seeds} Seeds x {n_trials} Trials)")
    print("="*60)
    print(f"Include s1: Mean Best NDCG = {np.mean(bests1):.6f} (+/- {np.std(bests1):.6f})")
    print(f"Exclude s1: Mean Best NDCG = {np.mean(bests2):.6f} (+/- {np.std(bests2):.6f})")
    print("="*60)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    x = np.arange(1, n_trials + 1)
    
    # Include s1 Mean + Std
    mean1 = np.mean(curves1, axis=0)
    std1 = np.std(curves1, axis=0)
    plt.plot(x, mean1, label="Include s1 (Mean)", color='blue', marker='o', markersize=4, alpha=0.9)
    plt.fill_between(x, mean1 - std1, mean1 + std1, color='blue', alpha=0.1)
    
    # Exclude s1 Mean + Std
    mean2 = np.mean(curves2, axis=0)
    std2 = np.std(curves2, axis=0)
    plt.plot(x, mean2, label="Exclude s1 (Mean)", color='red', marker='s', markersize=4, alpha=0.9)
    plt.fill_between(x, mean2 - std2, mean2 + std2, color='red', alpha=0.1)
    
    plt.title(f"HPO Convergence Stability: Tail Mean (s1 excluded) vs Full Mean\n(Mean over {n_seeds} seeds +/- 1 Std Dev)")
    plt.xlabel("Trial #")
    plt.ylabel("Cumulative Best NDCG@20")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    
    save_path = "aspire_experiments/hpo_stability_comparison_5seed.png"
    plt.savefig(save_path)
    print(f"\n[INFO] Comparison graph (5-seed) saved to {save_path}")
