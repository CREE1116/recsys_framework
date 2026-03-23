import os
import sys
import copy
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aspire_experiments.exp_utils import get_loader_and_svd, ensure_dir
from src.models.csar.ASPIRE import ASPIRE
from src.models.csar.ASPIRELayer import AspireFilter
from src.evaluation import evaluate_metrics

def gini_coefficient(array):
    """지니계수(Gini Coefficient) 계산"""
    array = np.clip(array, 0, None)
    if np.sum(array) == 0:
        return 0.0
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

def run_exp3(datasets):
    output_dir = ensure_dir("aspire_experiments/output/exp3")
    results = {}
    
    # 1. 시각화 및 분석용 Gamma 범위 설정 (0.3 ~ 3.0)
    gamma_list = np.linspace(0.3, 3.0, 28).tolist()
    plot_gammas = [0.3, 0.5, 1.0, 2.0]
    
    for ds in datasets:
        ds_output_dir = ensure_dir(os.path.join(output_dir, ds))
        try:
            print(f"\n{'='*50}")
            print(f"========== Processing {ds} ==========")
            print(f"{'='*50}")
            
            # 1. Load Data & Cache (Always k=None for full spectral info)
            loader, R, S, V, config = get_loader_and_svd(ds, k=None)
            popularity = np.array(R.sum(axis=0)).flatten()
            gini_orig = gini_coefficient(popularity)
            
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            
            hpo_eval_cfg = copy.deepcopy(config.get('evaluation', {}))
            main_metric = hpo_eval_cfg.get('main_metric', 'NDCG')
            main_k = hpo_eval_cfg.get('main_metric_k', 20)
            hpo_eval_cfg['metrics'] = [main_metric]
            hpo_eval_cfg['top_k'] = [main_k]
            metric_key = f"{main_metric}@{main_k}"
            
            trial_records = []
            P_diag = torch.tensor(popularity, dtype=torch.float32, device=V.device)
            p_proj = V.T @ P_diag  # V^T p
            V_sq = V ** 2
            
            # 2. HPO Definition
            def objective(params, trial):
                cfg = copy.deepcopy(config)
                cfg['device'] = device
                cfg['model'] = cfg.get('model', {})
                cfg['model']['name'] = 'aspire'
                cfg['model']['target_energy'] = 1.0 # 전체 스펙트럼 사용 (BEST 모델과 동일 유지)
                g_val = params['gamma']
                a_val = params['alpha']
                cfg['model']['gamma'] = g_val
                cfg['model']['alpha'] = a_val
                cfg['model']['visualize'] = False
                cfg['model']['visualize_heavyweight'] = False
                
                model = ASPIRE(cfg, loader)
                model.fit(loader)
                
                eval_batch_size = cfg.get('evaluation', {}).get('batch_size') or (cfg.get('train', {}).get('size', 512) * 2)
                val_loader = loader.get_validation_loader(eval_batch_size)
                
                res = evaluate_metrics(model, loader, hpo_eval_cfg, device=device, test_loader=val_loader, is_final=False)
                val_score = res.get(metric_key, 0.0)
                
                # 지니계수 계산 (W p)
                h, _, _ = AspireFilter.apply_filter(S, alpha=a_val, gamma=g_val)
                B_p = V @ (h * p_proj)
                B_diag = V_sq @ h
                p_filtered = B_p - (B_diag * P_diag)
                gini_filtered = gini_coefficient(p_filtered.cpu().numpy())
                
                trial.set_user_attr("gini_filtered", float(gini_filtered))
                trial_records.append({
                    "trial_id": trial.number,
                    "gamma": float(g_val),
                    "alpha": float(a_val),
                    "gini": float(gini_filtered),
                    "val_score": float(val_score)
                })
                return val_score
            
            # 3. 최적화 실행
            print(f"Starting Bayesian HPO for {ds} (Gamma: 0.3~2.0, Alpha: 0.0001~2.0, Log Scale)...")
            import optuna
            study = optuna.create_study(direction="maximize", study_name=ds)
            
            def _wrapped_objective(trial):
                gamma = trial.suggest_float("gamma", 0.3, 2.0)
                alpha = trial.suggest_float("alpha", 0.0001, 2.0, log=True)
                return objective({"gamma": gamma, "alpha": alpha}, trial)
                
            study.optimize(_wrapped_objective, n_trials=50)
            
            best_params = study.best_params
            best_val = study.best_value
            best_trial_gini = study.best_trial.user_attrs.get("gini_filtered", -1)
            
            res_dict = {
                "dataset": ds,
                "best_params": best_params,
                f"best_val_{metric_key}": best_val,
                "gini_original": float(gini_orig),
                "best_trial_gini": float(best_trial_gini),
                "trials": trial_records
            }
            results[ds] = res_dict
            
            with open(os.path.join(ds_output_dir, "results.json"), "w") as f:
                json.dump(res_dict, f, indent=4)
                
            # --- 시각화 ---
            gammas = [r["gamma"] for r in trial_records]
            alphas = [r["alpha"] for r in trial_records]
            ginis = [r["gini"] for r in trial_records]
            scores = [r["val_score"] for r in trial_records]
            
            plt.figure(figsize=(9, 6))
            sc = plt.scatter(gammas, ginis, c=scores, cmap='viridis', s=60, alpha=0.8, edgecolors='k')
            plt.axhline(gini_orig, color='red', linestyle='--', label=f'Original Gini ({gini_orig:.4f})')
            plt.scatter([best_params['gamma']], [best_trial_gini], c='red', marker='*', s=300, edgecolors='k', label="BEST")
            plt.colorbar(sc, label=f"Validation {metric_key}")
            plt.title(f"{ds}: Gamma vs Gini (Min $\gamma=0.3$)")
            plt.xlabel("Gamma ($\gamma$)")
            plt.ylabel("Gini Coefficient")
            plt.legend()
            plt.savefig(os.path.join(ds_output_dir, "gamma_vs_gini_scatter.png"), dpi=150)
            plt.close()
            
            plt.figure(figsize=(9, 6))
            sc = plt.scatter(alphas, ginis, c=scores, cmap='viridis', s=60, alpha=0.8, edgecolors='k')
            plt.axhline(gini_orig, color='red', linestyle='--', label=f'Original Gini ({gini_orig:.4f})')
            plt.scatter([best_params['alpha']], [best_trial_gini], c='red', marker='*', s=300, edgecolors='k', label="BEST")
            plt.colorbar(sc, label=f"Validation {metric_key}")
            plt.xscale('log')
            plt.title(f"{ds}: Alpha vs Gini")
            plt.xlabel("Alpha ($\\alpha$)")
            plt.ylabel("Gini Coefficient")
            plt.legend()
            plt.savefig(os.path.join(ds_output_dir, "alpha_vs_gini_scatter.png"), dpi=150)
            plt.close()

            print(f"Exp3 visuals saved for {ds}.")

        except Exception as e:
            print(f"Skipping {ds} due to error: {e}")
            import traceback
            traceback.print_exc()
            
    with open(os.path.join(output_dir, "summary_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nExperiment 3 completed. Results in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASPIRE Exp3: Gamma Impact")
    parser.add_argument("--dataset", nargs='+', default=["ml-100k"], help="Datasets split by space or comma")
    args = parser.parse_args()
    
    dataset_str = " ".join(args.dataset)
    datasets_to_run = [d.strip() for d in dataset_str.replace(',', ' ').split() if d.strip()]
    if 'all' in datasets_to_run:
        datasets_to_run = ['ml-100k', 'ml-1m', 'steam']
        run_exp3(datasets_to_run)
