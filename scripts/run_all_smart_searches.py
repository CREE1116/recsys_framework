import argparse
import yaml
import os
import sys
import json
import copy
import collections
import numpy as np
from smart_grid_search import SmartGridSearch
from bayesian_opt import BayesianOptimizer
from config_utils import merge_all_configs

# Add project root to sys.path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.gpu_accel import SVDCacheManager, GramEigenCacheManager, GramMatrixCacheManager
from src.models.general.slim import SLIMMatrixCacheManager
from src.models.general.item_knn import ItemKNNSimCacheManager

class Args:
    """Helper class to convert dictionary to object with attributes"""
    def __init__(self, **entries):
        self.__dict__.update(entries)

def run_all_searches(config_path, output_dir_base, cli_args=None):
    print(f"Loading batch configuration from {config_path}...")
    with open(config_path, 'r', encoding='utf-8') as f:
        batch_config = yaml.safe_load(f)

    # Allow legacy config (no top-level datasets) for backward compatibility
    datasets = batch_config.get('datasets', [])
    search_definitions = batch_config.get('searches', [])
    global_seeds = batch_config.get('seeds', None)
    # config 최상위 summary_metrics: 데이터셋 요약 시 표시할 메트릭 리스트
    global_summary_metrics = batch_config.get('summary_metrics', [])

    use_global_datasets = len(datasets) > 0
    dataset_loop_items = datasets if use_global_datasets else [None]

    # Ensure output directory exists
    os.makedirs(output_dir_base, exist_ok=True)

    for dataset_path in dataset_loop_items:
        dataset_results = {}
        dataset_name = "default"
        
        if use_global_datasets:
            # Extract clean dataset name e.g. 'ml100k' from 'configs/dataset/ml100k.yaml'
            dataset_name = os.path.basename(dataset_path).replace('.yaml', '')
            print(f"\n" + "#"*80)
            print(f"Processing Dataset: {dataset_name} ({dataset_path})")
            print("#"*80)

        for search_def_orig in search_definitions:
            search_def = copy.deepcopy(search_def_orig)
            sub_name = search_def.pop('name', 'Unnamed')
            
            # Set dataset config
            if use_global_datasets:
                search_def['dataset_config'] = dataset_path
                full_search_name = f"[{dataset_name}] {sub_name}"
            else:
                full_search_name = sub_name
                if 'dataset_config' not in search_def:
                    print(f"Error: No dataset_config found for search '{sub_name}'")
                    continue
                # Try to guess dataset name from config path if not global
                dataset_name = os.path.basename(search_def['dataset_config']).replace('.yaml', '')

            print(f"\n" + "="*60)
            print(f"Starting Search: {full_search_name}")
            print("="*60)
            
            if global_seeds is not None:
                search_def['seeds'] = global_seeds

            method = search_def.pop('method')
            
            # Defaults
            if 'device' not in search_def: search_def['device'] = None
            if 'direction' not in search_def: search_def['direction'] = 'max'
            
            if method == 'grid':
                if 'metric' not in search_def: search_def['metric'] = 'NDCG@10'
            elif method == 'bayesian':
                # Priority: CLI > YAML > Default(20)
                if 'metric' not in search_def: search_def['metric'] = 'NDCG@10'
                
                # Apply YAML values first
                if 'n_trials' not in search_def: search_def['n_trials'] = 20
                if 'patience' not in search_def: search_def['patience'] = 20
                
                # Override with CLI if provided
                if getattr(cli_args, 'n_trials', None) is not None:
                    search_def['n_trials'] = cli_args.n_trials
                if getattr(cli_args, 'patience', None) is not None:
                    search_def['patience'] = cli_args.patience
                
                # Handle legacy single param -> params list
                if 'params' not in search_def and 'param' in search_def:
                    search_def['params'] = [{
                        'name': search_def.pop('param'),
                        'type': search_def.pop('type', 'float'),
                        'range': search_def.pop('range'),
                        'log': search_def.pop('log', False)
                    }]

            args = Args(**search_def)
            
            try:
                model_name = os.path.basename(args.model_config).replace('.yaml', '')
                search_output_dir = os.path.join(output_dir_base, dataset_name, sub_name)
                
                seeds_to_run = global_seeds if global_seeds is not None else [42]
                all_seed_metrics = collections.defaultdict(list)
                best_metrics_per_seed = {}
                best_params_per_seed = {}
                best_dirs_per_seed = {}
                
                print(f"  -> Base Output directory: {search_output_dir}")
                print(f"  -> Seeds to evaluate independently: {seeds_to_run}")

                for seed in seeds_to_run:
                    # [Resumption] Check if BEST result for this seed already exists
                    dataset_name_clean = dataset_name # already cleaned above
                    best_dir_candidate = os.path.join('trained_model', dataset_name_clean, f"BEST_{model_name}_seed_{seed}")
                    metrics_path = os.path.join(best_dir_candidate, "final_metrics.json")
                    
                    if os.path.exists(metrics_path):
                        print(f"\n   [SKIP] BEST result for Seed {seed} already exists at {best_dir_candidate}. Skipping search.")
                        # Load existing metrics for aggregation
                        try:
                            with open(metrics_path, 'r', encoding='utf-8') as f:
                                final_metrics = json.load(f)
                            best_metrics_per_seed[seed] = final_metrics.get(getattr(args, 'metric', 'NDCG@10'), None)
                            best_params_per_seed[seed] = {"status": "reused_existing"}
                            best_dirs_per_seed[seed] = best_dir_candidate
                            for k, v in final_metrics.items():
                                all_seed_metrics[k].append(v)
                        except Exception as e:
                            print(f"   [Warning] Could not load existing metrics: {e}. Proceeding with search.")
                        continue

                    print(f"\n   --- Starting Independent Search for Seed: {seed} ---")
                    # Update config for this specific run
                    args.seeds = seed 
                    
                    seed_output_dir = os.path.join(search_output_dir, f"seed_{seed}")
                    
                    best_metric = None
                    best_params = {}
                    best_dir = None
                    
                    if method == 'grid':
                        searcher = SmartGridSearch(args)
                        searcher.search()
                        best_metric = searcher.best_global_metric
                        best_dir = searcher.best_global_dir
                        best_params = {args.param: 'See best_dir'} 
                    elif method == 'bayesian':
                        searcher = BayesianOptimizer(args)
                        try:
                            searcher.search(output_dir=seed_output_dir)
                        finally:
                            searcher.cleanup()
                            # BEST 모델로 test 셋 평가 1회 재실행
                            try:
                                searcher.rerun_best_with_test()
                            except Exception as te:
                                print(f"[HPO] Test rerun exception: {te}")

                        
                        best_metric = searcher.study.best_value if hasattr(searcher, 'study') and searcher.study.trials else None
                        best_dir = searcher.best_global_dir
                        best_params = searcher.study.best_params if hasattr(searcher, 'study') and searcher.study.trials else {}
                    
                    # Try to load final_metrics.json from best_dir
                    final_metrics = {}
                    if best_dir and os.path.exists(os.path.join(best_dir, "final_metrics.json")):
                        with open(os.path.join(best_dir, "final_metrics.json"), 'r', encoding='utf-8') as f:
                            final_metrics = json.load(f)
                            
                    target_metric = getattr(args, 'metric', 'NDCG@10')
                    if target_metric.startswith('val_'):
                        target_metric = target_metric[4:]
                    
                    best_metrics_per_seed[seed] = final_metrics.get(target_metric, best_metric)
                    best_params_per_seed[seed] = best_params
                    best_dirs_per_seed[seed] = best_dir
                    
                    # Accumulate for average
                    for k, v in final_metrics.items():
                        all_seed_metrics[k].append(v)
                        
                    # ── 캐시 정리 (메모리 누수 방지) ──
                    print("   --- Cleaning up in-memory global caches to prevent OOM ---")
                    try:
                        # NOTE: SVDCacheManager().invalidate()는 디스크의 파일을 삭제하므로 호출하지 않음
                        GramEigenCacheManager.clear()
                        GramMatrixCacheManager.clear()
                        SLIMMatrixCacheManager.clear()
                        ItemKNNSimCacheManager.clear()
                    except Exception as e:
                        print(f"   [Warning] Cache cleanup error: {e}")
                
                # After all seeds, compute average
                aggregated_metrics = {}
                for k, v_list in all_seed_metrics.items():
                    valid = [v for v in v_list if v is not None and not (isinstance(v, float) and np.isnan(v))]
                    if valid:
                        aggregated_metrics[k] = {
                            "mean": float(np.mean(valid)),
                            "std": float(np.std(valid))
                        }

                avg_target_metric = aggregated_metrics.get(target_metric, {}).get("mean", None)

                # ── 1. mean-only 요약 (가장 깔끔) ─────────────────────────
                summary_mean = {k: v["mean"] for k, v in aggregated_metrics.items()}

                # ── 2. mean±std 요약 (기존 final_metric_average.json) ─────
                # (aggregated_metrics 그대로)

                # ── 3. 전체 기록 (seed별 raw metrics + params + dirs) ──────
                result_entry = {
                    'dataset': dataset_name,
                    'model': model_name,
                    'search_name': sub_name,
                    'target_metric': target_metric,
                    'avg_best_metric': avg_target_metric,
                    'best_metrics_per_seed': best_metrics_per_seed,
                    'best_params_per_seed': best_params_per_seed,
                    'best_dirs_per_seed': best_dirs_per_seed,
                    'full_metrics_per_seed': {
                        str(seed): dict(zip(all_seed_metrics.keys(),
                                           [all_seed_metrics[k][i] for k in all_seed_metrics]))
                        for i, seed in enumerate(seeds_to_run)
                    },
                    'final_metric_average': aggregated_metrics
                }

                dataset_results[model_name] = result_entry

                os.makedirs(search_output_dir, exist_ok=True)

                # 파일 1: mean only
                mean_file = os.path.join(search_output_dir, "summary_mean.json")
                with open(mean_file, 'w', encoding='utf-8') as f:
                    json.dump(summary_mean, f, indent=4)

                # 파일 2: mean ± std
                avg_res_file = os.path.join(search_output_dir, "final_metric_average.json")
                with open(avg_res_file, 'w', encoding='utf-8') as f:
                    json.dump(aggregated_metrics, f, indent=4)

                # 파일 3: 전체 기록
                model_res_file = os.path.join(search_output_dir, f"result_{dataset_name}_{model_name}.json")
                with open(model_res_file, 'w', encoding='utf-8') as f:
                    json.dump(result_entry, f, indent=4)

                print(f"  [요약] {model_name} @ {dataset_name}")
                print(f"    ① mean-only       → {mean_file}")
                print(f"    ② mean ± std      → {avg_res_file}")
                print(f"    ③ 전체 기록        → {model_res_file}")
                if avg_target_metric is not None:
                    std_val = aggregated_metrics.get(target_metric, {}).get("std", 0)
                    print(f"    {target_metric}: {avg_target_metric:.4f} ± {std_val:.4f}")

            except Exception as e:
                print(f"Error executing search '{full_search_name}': {e}")
                import traceback
                traceback.print_exc()
            
            # --- Memory & Disk Cache Cleanup ---
            # To prevent OOM and disk bloat across different models/datasets
            try:
                print("\n[Resource Cleanup] Clearing GPU/Eigen memory and SVD disk caches...")
                GramEigenCacheManager.clear()
                GramMatrixCacheManager.clear()
                SLIMMatrixCacheManager.clear()
                # SVDCacheManager clears the default 'data_cache' directory svd_*.pt files
                SVDCacheManager().clear_cache(dataset_name=dataset_name)
            except Exception as clean_e:
                print(f"Warning: Failed to clear cache: {clean_e}")

        # End of dataset loop: Save aggregated best params for this dataset
        if use_global_datasets:
            agg_file = os.path.join(output_dir_base, dataset_name, f"best_hyperparameters_{dataset_name}.json")
            print(f"\nScanning and aggregating ALL best hyperparameters for {dataset_name} to {agg_file}...")
            
            # Read all existing result files across all subdirectories
            dataset_dir = os.path.join(output_dir_base, dataset_name)
            all_results = {}
            if os.path.exists(dataset_dir):
                for sub_dir in os.listdir(dataset_dir):
                    sub_dir_path = os.path.join(dataset_dir, sub_dir)
                    if os.path.isdir(sub_dir_path):
                        for file in os.listdir(sub_dir_path):
                            if file.startswith(f"result_{dataset_name}_") and file.endswith(".json"):
                                try:
                                    with open(os.path.join(sub_dir_path, file), 'r', encoding='utf-8') as f:
                                        res = json.load(f)
                                        model_name = res.get('model', 'unknown')
                                        all_results[model_name] = res
                                except Exception as e:
                                    print(f"Error reading {file}: {e}")
            
            if all_results:
                os.makedirs(os.path.dirname(agg_file), exist_ok=True)
                with open(agg_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=4)

            # ── 데이터셋별 요약 (target metric 기준 정렬) ────────────────────
            # CLI: --summary_metrics "NDCG@20,Recall@20,Coverage@20"
            extra_metrics = global_summary_metrics  # config 최상위에서 읽음

            dataset_dir = os.path.join(output_dir_base, dataset_name)
            # model → {metric_key: {mean, std, display}, sort_val}
            summary_data = {}  # mname → {'target': float, 'target_metric': str, 'metrics': {k: {mean,std}}}
            if os.path.exists(dataset_dir):
                for sub_dir in sorted(os.listdir(dataset_dir)):
                    sub_dir_path = os.path.join(dataset_dir, sub_dir)
                    if not os.path.isdir(sub_dir_path):
                        continue
                    for fname in os.listdir(sub_dir_path):
                        if fname.startswith(f"result_{dataset_name}_") and fname.endswith(".json"):
                            try:
                                with open(os.path.join(sub_dir_path, fname), encoding='utf-8') as f:
                                    res = json.load(f)
                                mname = res.get('search_name') or res.get('model', sub_dir)
                                avg = res.get('final_metric_average', {})
                                tgt = res.get('target_metric', 'NDCG@20')
                                display_metrics = ([tgt] + extra_metrics) if extra_metrics else [tgt]
                                row = {}
                                for km in display_metrics:
                                    if km in avg:
                                        row[km] = {'mean': avg[km].get('mean'), 'std': avg[km].get('std')}
                                sort_val = avg.get(tgt, {}).get('mean') or res.get('avg_best_metric') or -1
                                summary_data[mname] = {
                                    'target_metric': tgt,
                                    'sort_val': float(sort_val) if sort_val is not None else -1,
                                    'metrics': row
                                }
                            except Exception:
                                pass

            if summary_data:
                # target metric 기준 내림차순 정렬
                sorted_models = sorted(summary_data.keys(),
                                       key=lambda m: summary_data[m]['sort_val'], reverse=True)

                # JSON 저장: 정렬된 순서, raw mean/std
                ds_summary_out = {}
                for rank, mname in enumerate(sorted_models, 1):
                    d = summary_data[mname]
                    entry = {'rank': rank, 'target_metric': d['target_metric']}
                    for km, vals in d['metrics'].items():
                        entry[km] = vals  # {mean, std}
                    ds_summary_out[mname] = entry

                ds_summary_file = os.path.join(dataset_dir, f"dataset_summary_{dataset_name}.json")
                with open(ds_summary_file, 'w', encoding='utf-8') as f:
                    json.dump(ds_summary_out, f, indent=4, ensure_ascii=False)

                # 콘솔 테이블
                tgt_sample = summary_data[sorted_models[0]]['target_metric'] if sorted_models else 'NDCG@20'
                all_km = list(dict.fromkeys(
                    [summary_data[m]['target_metric'] for m in sorted_models] + extra_metrics
                ))
                col_w = 18
                header = f"{'Rank':<5}{'Model':<22}" + "".join(f"{km:>{col_w}}" for km in all_km)
                sep = "-" * len(header)
                print(f"\n{'='*70}")
                print(f"  Dataset Summary: {dataset_name}  (sorted by {tgt_sample})")
                print(f"{'='*70}")
                print(header)
                print(sep)
                for rank, mname in enumerate(sorted_models, 1):
                    d = summary_data[mname]
                    line = f"{rank:<5}{mname:<22}"
                    for km in all_km:
                        if km in d['metrics']:
                            m = d['metrics'][km].get('mean')
                            s = d['metrics'][km].get('std')
                            cell = f"{m:.4f}±{s:.4f}" if (m is not None and s is not None) else "-"
                        else:
                            cell = "-"
                        line += f"{cell:>{col_w}}"
                    print(line)
                print(sep)
                print(f"  Saved → {ds_summary_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple smart searches from a config file")
    parser.add_argument('--config', type=str, required=True, help='Path to batch config YAML')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save results')
    parser.add_argument('--n_trials', type=int, default=None, help='Override number of trials for bayesian search')
    parser.add_argument('--patience', type=int, default=None, help='Override patience for bayesian search')

    cli_args = parser.parse_args()

    run_all_searches(cli_args.config, cli_args.output_dir, cli_args)
