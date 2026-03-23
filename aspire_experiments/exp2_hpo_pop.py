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

from aspire_experiments.exp_utils import get_loader_and_svd, ensure_dir, AspireHPO, get_eval_config
from src.models.csar.ASPIRE import ASPIRE
from src.models.csar.ASPIRELayer import AspireFilter
from src.evaluation import evaluate_metrics

def gini_coefficient(array):
    """지니계수(Gini Coefficient)를 계산합니다."""
    # 인기도처럼 음수가 될 수 없는 값 가정. 
    # 필터 적용 시 0 이하로 떨어지는 노이즈를 클리핑
    array = np.clip(array, 0, None)
    if np.sum(array) == 0:
        return 0.0
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

def run_exp2(datasets):
    output_dir = ensure_dir("aspire_experiments/output/exp2")
    results = {}
    
    # HPO 설정 (Mean-scaling Alpha 반영: 0.0001 ~ 2.0)
    params_spec = [
        {'name': 'gamma', 'type': 'float', 'range': '0.3 2.0'},
        {'name': 'alpha', 'type': 'float', 'range': '0.0001 2.0', 'log': True}
    ]
    
    for ds in datasets:
        ds_output_dir = ensure_dir(os.path.join(output_dir, ds))
        try:
            print(f"\n{'='*50}")
            print(f"========== Processing {ds} ==========")
            print(f"{'='*50}")
            
            # 1. Load Data & Cache (k=None ensures Full or Maximum Cached EVD)
            loader, R, S, V, config = get_loader_and_svd(ds, k=None)
            
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            
            # Prepare eval_cfg for HPO objective
            # This ensures that the evaluation metrics are explicitly set for the HPO
            hpo_eval_cfg = copy.deepcopy(config.get('evaluation', {}))
            main_metric = hpo_eval_cfg.get('main_metric', 'NDCG')
            main_k = hpo_eval_cfg.get('main_metric_k', 20)
            hpo_eval_cfg['metrics'] = [main_metric]
            hpo_eval_cfg['top_k'] = [main_k]
            metric_key = f"{main_metric}@{main_k}"

            popularity = np.array(R.sum(axis=0)).flatten()
            
            # 2. HPO Definition
            def objective(params):
                cfg = copy.deepcopy(config)
                cfg['device'] = device
                cfg['model'] = cfg.get('model', {})
                cfg['model']['name'] = 'aspire'
                cfg['model']['target_energy'] = 1.0
                cfg['model']['gamma'] = params['gamma']
                cfg['model']['alpha'] = params['alpha']
                
                # 시각화 끄기 (HPO 중 불필요한 I/O, Error 방지)
                cfg['model']['visualize'] = False
                cfg['model']['visualize_heavyweight'] = False
                
                # Model Initialization & Fitting
                model = ASPIRE(cfg, loader)
                model.fit(loader)
                
                # Evaluate (test_loader 생성)
                eval_batch_size = cfg.get('evaluation', {}).get('batch_size') or (cfg.get('train', {}).get('batch_size', 512) * 2)
                val_loader = loader.get_validation_loader(eval_batch_size)
                
                # Validate using the prepared hpo_eval_cfg
                res = evaluate_metrics(model, loader, hpo_eval_cfg, device=device, test_loader=val_loader, is_final=False)
                return res.get(metric_key, 0.0)

            # 3. Run HPO (n_trials=30)
            print(f"Starting HPO for {ds}...")
            hpo = AspireHPO(params_spec, n_trials=30, patience=10, direction='maximize')
            best_params, best_val = hpo.search(objective, study_name=ds, output_dir=ds_output_dir)
            print(f"Optimal Params for {ds}: {best_params} (Val {metric_key}: {best_val:.4f})")
            
            # 4. Filter Evaluation (Apply h to Popularity)
            best_gamma = best_params['gamma']
            best_alpha = best_params['alpha']
            
            # h(σ) = σ^gamma / (σ^gamma + alpha)
            h, _, _ = AspireFilter.apply_filter(S, alpha=best_alpha, gamma=best_gamma)
            
            # 4-1. Spectral Domain Popularity Change
            P_diag = torch.tensor(popularity, dtype=torch.float32, device=V.device)
            V_T_P = V.T @ (P_diag.unsqueeze(1) * V)
            p_k = torch.diag(V_T_P)
            
            p_k_filtered = h * p_k
            
            p_k_np = p_k.cpu().numpy()
            p_k_filtered_np = p_k_filtered.cpu().numpy()
            
            # 4-2. Spatial Domain (Item) Popularity Change
            # 실제 추천 행렬 W = B - diag(B)
            # p_filtered = W p = B p - diag(B) p
            
            p_proj = V.T @ P_diag  # V^T p
            B_p = V @ (h * p_proj) # B p = V diag(h) V^T p
            
            # diag(B) 계산: B_{ii} = sum_k V_{ik}^2 h_k
            V_sq = V ** 2
            B_diag = V_sq @ h
            
            # W p = B p - diag(B) * p
            p_filtered = B_p - (B_diag * P_diag)
            
            p_filtered_np = p_filtered.cpu().numpy()
            
            # Gini Coefficient Calculation
            gini_orig = gini_coefficient(popularity)
            gini_filtered = gini_coefficient(p_filtered_np)
            
            # Store Results
            res_dict = {
                "dataset": ds,
                "best_params": best_params,
                f"best_val_{metric_key}": best_val,
                "gini_original": float(gini_orig),
                "gini_filtered": float(gini_filtered)
            }
            results[ds] = res_dict
            
            with open(os.path.join(ds_output_dir, "results.json"), "w") as f:
                json.dump(res_dict, f, indent=4)
                
            # 5. Visualizations
            # -------------------------------------------------------------
            # Plot 1: Spectral Popularity Change
            plt.figure(figsize=(10, 6))
            k_plot = min(p_k_np.shape[0], 500) # Plot top 500 for visibility
            components = np.arange(k_plot)
            
            # Since p_k is typically positive and large at head, we log scale it.
            plt.plot(components, np.log1p(np.abs(p_k_np[:k_plot])), label='Original $p_k$', color='blue')
            plt.plot(components, np.log1p(np.abs(p_k_filtered_np[:k_plot])), label='Filtered $\hat{p}_k$', color='red', linestyle='--')
            
            plt.title(f"{ds}: Spectral Popularity ($p_k$) Change\n(Opt $\gamma={best_gamma:.2f}, \\alpha={best_alpha:.2f}$)")
            plt.xlabel("Singular Vector Component Index ($k$)")
            plt.ylabel("log(1 + $|p_k|$)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(ds_output_dir, "spectral_pop_change.png"), dpi=150)
            plt.close()
            
            # -------------------------------------------------------------
            # Plot 2: Item Popularity Change (Sorted)
            plt.figure(figsize=(10, 6))
            
            # Sort items by original popularity (descending)
            sort_idx = np.argsort(popularity)[::-1]
            pop_sorted = popularity[sort_idx]
            p_filtered_sorted = p_filtered_np[sort_idx]
            
            x_items = np.arange(len(pop_sorted))
            
            plt.plot(x_items, np.log1p(pop_sorted), label='Original Item Popularity', color='blue', alpha=0.8)
            plt.scatter(x_items, np.log1p(np.clip(p_filtered_sorted, 0, None)), 
                        label='Filtered (Expected) Popularity', color='red', s=2, alpha=0.3)
            
            plt.title(f"{ds}: Item Popularity Distribution Change\nGini Orig: {gini_orig:.4f} -> Gini Filtered: {gini_filtered:.4f}")
            plt.xlabel("Item Rank (by Original Popularity)")
            plt.ylabel("log(1 + Popularity)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(ds_output_dir, "item_pop_change.png"), dpi=150)
            plt.close()
            
            print(f"Evaluation for {ds} Complete. Gini: {gini_orig:.4f} -> {gini_filtered:.4f}")

        except Exception as e:
            print(f"Skipping {ds} due to error: {e}")
            import traceback
            traceback.print_exc()
            
    # 전체 요약 결과를 root output 폴더에 저장
    with open(os.path.join(output_dir, "summary_results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nExperiment 2 completed. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASPIRE Exp2: HPO && Popularity Change Measurement")
    parser.add_argument("--dataset", nargs='+', default=["ml-100k"],
                        help="평가할 데이터셋 이름 (예: ml-100k, yaml 빼고 입력). 쉼표 구분 및 띄어쓰기 포함 다중입력 가능.")
    args = parser.parse_args()
    
    # 리스트로 받아진 인자들을 하나의 문자열로 합친 뒤 쉼표로 분리 (띄어쓰기 대응)
    dataset_str = "".join(args.dataset)
    if dataset_str == 'all':
        datasets_to_run = ['ml-100k', 'ml-1m', 'yahoo_r3', 'gowalla', 'yelp2018', 'amazon-book']
    else:
        datasets_to_run = [d.strip() for d in dataset_str.split(',') if d.strip()]
        
    run_exp2(datasets_to_run)
