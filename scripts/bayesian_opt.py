import argparse
import yaml
import os
import sys
import shutil
import glob
import json
import copy
import optuna
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import main as run_single_experiment
from src.data_loader import DataLoader

class BayesianOptimizer:
    def __init__(self, args):
        self.args = args
        self.dataset_config = self.load_yaml(args.dataset_config)
        self.model_config = self.load_yaml(args.model_config)
        self.base_config = self.merge_configs(self.dataset_config, self.model_config)
        
        if args.device:
            self.base_config['device'] = args.device
        elif 'device' not in self.base_config:
            self.base_config['device'] = 'auto'

        if hasattr(args, 'seeds') and args.seeds is not None:
            single_seed = args.seeds[0] if isinstance(args.seeds, list) else args.seeds
            self.base_config['seed'] = int(single_seed)
            self.optuna_seed = int(single_seed)
        else:
            self.optuna_seed = 42 # Default fallback



        # Check dataset dimensions to clamp K/Rank
        try:
            print(f"[BayesianOptimizer] Loading dataset to check dimensions...")
            # Create a shallow copy and ensure 'evaluation' exists for DataLoader safety
            check_config = copy.deepcopy(self.dataset_config)
            if 'evaluation' not in check_config:
                check_config['evaluation'] = {'validation_method': 'holdout', 'final_method': 'holdout'}
            check_config['seed'] = self.optuna_seed
            
            temp_loader = DataLoader(check_config)
            n_users = temp_loader.n_users
            n_items = temp_loader.n_items
            max_dim = min(n_users, n_items)
            print(f"[BayesianOptimizer] Dataset dimensions: Users={n_users}, Items={n_items}. Max Rank/K = {max_dim}")
            if not np.isfinite(max_dim):
                print(f"[BayesianOptimizer] Warning: Infinite max_dim found. Defaulting to 2000.")
                max_dim = 2000
            else:
                max_dim = int(max_dim)
        except Exception as e:
            print(f"[BayesianOptimizer] Warning: Failed to load dataset dimensions ({e}). K clamping disabled.")
            import traceback
            traceback.print_exc()
            max_dim = 2000 # Safe default fallback instead of inf

        # Handle multiple parameters
        # self.args.params should be a list of dicts: [{'name': '...', 'type': '...', 'range': '...', 'log': bool}, ...]
        # If legacy single param args are present, convert to list
        if hasattr(args, 'params') and args.params:
            self.params_list = args.params
        else:
            self.params_list = [{
                'name': args.param,
                'type': args.type,
                'range': args.range,
                'log': args.log
            }]
            
        # Clamp Rank/K parameters if type is 'int_min_dim'
        for param in self.params_list:
            if param['type'] == 'int_min_dim':
                # 하드 캡: SVD/임베딩 차원은 너무 높으면 계산 비용 급증
                MAX_DIM_CAP = 12288
                effective_max = min(max_dim, MAX_DIM_CAP)
                
                if 'range' not in param:
                    print(f"[BayesianOptimizer] Auto-setting range for '{param['name']}' to 1~{effective_max}")
                    param['range'] = f"1 {effective_max}"
                else:
                    vals = sorted(map(int, param['range'].split()))
                    low, high = vals[0], vals[1]
                    clamped_high = min(high, effective_max)
                    if clamped_high != high:
                        print(f"[BayesianOptimizer] Clamping '{param['name']}' max: {high} → {clamped_high} (max_dim={max_dim}, cap={MAX_DIM_CAP})")
                    if low > clamped_high:
                        low = max(1, clamped_high // 2)
                        print(f"[BayesianOptimizer] Adjusted low bound to {low}")
                    param['range'] = f"{low} {clamped_high}"
                
                # Convert type back to 'int' for Optuna compatibility
                param['type'] = 'int'
                
                # [추가] 차원 관련 파라미터는 기본적으로 로그 스케일 탐색이 효율적임
                if 'log' not in param:
                    param['log'] = True

        self.metric_key = args.metric
        self.maximize = (args.direction == 'max')
        self.best_global_metric = -float('inf') if self.maximize else float('inf')
        self.best_global_dir = None
        self.all_experiment_dirs = []

    def load_yaml(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def merge_configs(self, dataset_conf, model_conf):
        from config_utils import merge_all_configs
        return merge_all_configs(dataset_conf, model_conf)

    def set_nested_value(self, config, path_str, value):
        path = path_str.split('.')
        temp = config
        for key in path[:-1]:
            temp = temp.setdefault(key, {})
        temp[path[-1]] = value

    def get_experiment_dir(self, config):
        model_name = config['model']['name']
        dataset_name = config['dataset_name']
        run_name = config.get('run_name')
        
        base_path = os.path.join('trained_model', dataset_name)
        if run_name and run_name != 'default':
            folder_name = f"{model_name}__{run_name}"
        else:
            folder_name = model_name
            
        return os.path.join(base_path, folder_name)

    def run_experiment(self, param_values):
        # param_values is a dict {param_name: value}
        config = copy.deepcopy(self.base_config)

        # [HPO] Skip test set evaluation — only use validation metrics
        config['hpo_mode'] = True

        run_name_parts = []
        for p_def in self.params_list:
            p_name = p_def['name']
            val = param_values[p_name]

            # Type casting if needed (though optuna usually handles it)
            if p_def.get('type') == 'int':
                val = int(val)

            self.set_nested_value(config, p_name, val)

            # Format run name
            short_name = p_name.split('.')[-1]
            if isinstance(val, float):
                run_name_parts.append(f"{short_name}={val:.6g}")
            else:
                run_name_parts.append(f"{short_name}={val}")

        config['run_name'] = "_".join(run_name_parts)
        print(f"\n>>> Running experiment: {config['run_name']} [hpo_mode]")

        exp_dir = self.get_experiment_dir(config)
        metrics_file = os.path.join(exp_dir, 'final_metrics.json')

        if os.path.exists(metrics_file):
            print(f"Results already exist at {exp_dir}. Loading from {os.path.basename(metrics_file)}...")
            try:
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
            except:
                print("Failed to load existing metrics. Re-running...")
                metrics = None
        else:
            metrics = None

        if metrics is None:
            try:
                run_single_experiment(config)
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
            except Exception as e:
                print(f"Experiment failed with config: {config.get('run_name')}")
                print(f"Error details: {e}")
                import traceback
                traceback.print_exc()
                # Return penalty for failure
                return -float('inf') if self.maximize else float('inf'), None

        self.all_experiment_dirs.append(exp_dir)

        metric_val = metrics.get(self.metric_key)

        # 키가 없으면 부분 일치로 탐색 (방어적 fallback)
        if metric_val is None:
            for k, v in metrics.items():
                if self.metric_key in k:
                    metric_val = v
                    break

        if metric_val is None:
            print(f"Warning: Metric {self.metric_key} not found.")
            return -float('inf') if self.maximize else float('inf'), exp_dir

        return metric_val, exp_dir


    def cleanup(self):
        """HPO 완료 후 모든 trial 디렉토리 삭제. BEST 재실행은 rerun_best_with_test()에서 처리."""
        print("\n=== Cleaning up all trial checkpoints ===")
        model_name = self.base_config['model']['name']
        dataset_name = self.base_config['dataset_name']
        base_path = os.path.join('trained_model', dataset_name)

        stray_dirs = glob.glob(os.path.join(base_path, f"{model_name}__*"))
        if os.path.exists(os.path.join(base_path, model_name)):
            stray_dirs.append(os.path.join(base_path, model_name))

        # Filter out None and Normalize paths
        all_to_check = set()
        for p in stray_dirs:
            if p: all_to_check.add(os.path.abspath(os.path.normpath(p)))
        for p in self.all_experiment_dirs:
            if p: all_to_check.add(os.path.abspath(os.path.normpath(p)))

        best_dir_abs = os.path.abspath(os.path.normpath(self.best_global_dir)) if self.best_global_dir else None

        deleted_count = 0
        for path in all_to_check:
            if not os.path.exists(path):
                continue
            
            basename = os.path.basename(path)
            # BEST_ 폴더는 절대 삭제하지 않음 (이전 시드 결과물 등 보호)
            if basename.startswith("BEST_"):
                continue
            
            # 현재 HPO의 BEST trial 디렉토리는 보존 (rerun_best_with_test에서 재사용 위함)
            if best_dir_abs and path == best_dir_abs:
                print(f"Skipping deletion of the best trial directory (for reuse): {path}")
                continue
                
            try:
                shutil.rmtree(path)
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {path}: {e}")
        print(f"Cleanup complete. Deleted {deleted_count} trial dirs.")

    def rerun_best_with_test(self):
        """
        HPO 완료 후 메모리의 최적 하이퍼파라미터로 test 셋 평가를 1회 실행.
        결과를 BEST_{model}_{seed} 디렉토리에 직접 저장합니다.
        """
        if not hasattr(self, 'study') or not self.study.trials:
            print("[HPO] No completed trials, skipping test rerun.")
            return

        best_params = self.study.best_params
        print(f"\n{'='*60}")
        print(f"[HPO] Re-running BEST config with full TEST evaluation...")
        print(f"[HPO] Best params: {best_params}")
        print(f"{'='*60}")

        config = copy.deepcopy(self.base_config)
        config.pop('hpo_mode', None)  # test mode (hpo_mode=False)

        for p_def in self.params_list:
            p_name = p_def['name']
            val = best_params.get(p_name)
            if val is None:
                continue
            if p_def.get('type') == 'int':
                val = int(val)
            self.set_nested_value(config, p_name, val)

        # run_name은 제거 — output_path_override로 직접 경로 지정
        config.pop('run_name', None)
        model_name = self.base_config['model']['name']
        dataset_name = self.base_config['dataset_name']
        best_dir = os.path.join('trained_model', dataset_name,
                                f"BEST_{model_name}_seed_{self.optuna_seed}")

        # 기존 BEST_ 폴더가 있으면 제거
        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        os.makedirs(best_dir, exist_ok=True)

        # [최적화] 이미 학습된 best_model.pt가 있다면 재학습 없이 복사해서 TEST만 수행
        reused = False
        if self.best_global_dir and os.path.exists(self.best_global_dir):
            source_best_model = os.path.join(self.best_global_dir, "best_model.pt")
            if os.path.exists(source_best_model):
                print(f"[HPO] Found existing best model at {source_best_model}. Reusing for TEST.")
                shutil.copy(source_best_model, os.path.join(best_dir, "best_model.pt"))
                # config에 'train' 블록이 있어도 epoch 0으로 만들어 학습 건너뛰게 함
                if 'train' in config:
                    config['train']['epochs'] = 0
                config['skip_fit'] = True  # non-trainable (fit 방식) 모델도 재생성 생략
                reused = True

        old_best_dir = self.best_global_dir
        config['output_path_override'] = best_dir
        self.best_global_dir = best_dir

        try:
            run_single_experiment(config)
            print(f"[HPO] Test evaluation saved to {best_dir}")
            
            # [최적화] BEST 폴더로 모델 복사 및 평가가 성공했다면, 원본 임시 trial 폴더는 이제 삭제해도 됨
            # reused 여부와 상관없이, BEST_ 폴더로 모든 결과가 옮겨졌으므로 구 폴더는 삭제합니다.
            if old_best_dir and os.path.exists(old_best_dir):
                # old_best_dir이 BEST_ 폴더가 아닐 때만 삭제 (무한 루프 및 중복 삭제 방지)
                if "BEST_" not in os.path.basename(old_best_dir):
                    print(f"[HPO] Final evaluation complete. Deleting intermediate dir: {old_best_dir}")
                    try:
                        shutil.rmtree(old_best_dir)
                    except Exception as e:
                        print(f"Warning: Failed to delete intermediate dir {old_best_dir}: {e}")
        except Exception as e:
            print(f"[HPO] Test rerun failed: {e}")
            import traceback
            traceback.print_exc()

    def objective(self, trial):
        current_params = {}
        for p_def in self.params_list:
            name = p_def['name']
            p_type = p_def.get('type', 'float')
            p_range = p_def.get('range') or p_def.get('choices')
            if p_range is None:
                raise KeyError(f"Parameter '{name}' must have either 'range' or 'choices' defined.")
            p_log = p_def.get('log', False)

            if p_type == 'float':
                vals = sorted(map(float, p_range.split()))
                low, high = vals[0], vals[1]
                val = trial.suggest_float(name, low, high, log=p_log)
            elif p_type == 'int':
                vals = sorted(map(int, p_range.split()))
                low, high = vals[0], vals[1]
                val = trial.suggest_int(name, low, high, log=p_log)
            elif p_type == 'categorical':
                if isinstance(p_range, str):
                    choices = p_range.split()
                else:
                    choices = p_range
                # Try to convert safely without precision loss
                new_choices = []
                for c in choices:
                    try:
                        f_val = float(c)
                        # Only convert to int if it's identical (e.g. 1.0 or "1")
                        if f_val == int(f_val):
                            new_choices.append(int(f_val))
                        else:
                            new_choices.append(f_val)
                    except (ValueError, TypeError):
                        new_choices.append(c)
                choices = new_choices
                val = trial.suggest_categorical(name, choices)
            else:
                raise ValueError(f"Unknown type: {p_type}")
            
            current_params[name] = val
            
        print(f"\n[Trial {trial.number}] Params: {current_params}")

        metric, exp_dir = self.run_experiment(current_params)

        # Force memory cleanup between HPO trials
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Update global best
        if self.is_better(metric, self.best_global_metric):
            self.best_global_metric = metric
            self.best_global_dir = exp_dir
            print(f"  -> NEW BEST: {metric}")
        else:
            print(f"  -> Metric: {metric} (Best: {self.best_global_metric})")
        
        return metric

    def is_better(self, current, best):
        if self.maximize:
            return current > best
        else:
            return current < best

    def save_results(self, output_dir):
        """Save study results: CSV and visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save Trials Dataframe
        try:
            df = self.study.trials_dataframe()
            csv_path = os.path.join(output_dir, 'trials.csv')
            df.to_csv(csv_path, index=False)
            print(f"Saved trials to {csv_path}")
        except Exception as e:
            print(f"Error saving CSV: {e}")

        # 2. Visualizations
        try:
            import matplotlib
            matplotlib.use('Agg') # Force headless backend
            import matplotlib.pyplot as plt
            
            # Optimization History
            plt.figure(figsize=(10, 6))
            trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if len(trials) > 0:
                values = [t.value for t in trials]
                best_values = [np.max(values[:i+1]) if self.maximize else np.min(values[:i+1]) for i in range(len(values))]
                
                plt.plot(values, marker='o', alpha=0.5, label='Objective Value')
                plt.plot(best_values, color='red', linewidth=2, label='Best Value')
                plt.xlabel('Trial')
                plt.ylabel('Metric Value')
                plt.title('Optimization History')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(output_dir, 'optimization_history.png'))
                plt.close()

            # Parameter Importance (if more than 1 param)
            if len(self.params_list) > 1:
                try:
                    importance = optuna.importance.get_param_importances(self.study)
                    names = list(importance.keys())
                    values = list(importance.values())
                    
                    plt.figure(figsize=(10, 6))
                    plt.barh(names, values)
                    plt.xlabel('Importance')
                    plt.title('Hyperparameter Importance')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'param_importance.png'))
                    plt.close()
                except Exception as e:
                    print(f"Could not calculate importance: {e}")

            # Slice Plots (one per param)
            for p_def in self.params_list:
                p_name = p_def['name']
                try:
                    plt.figure(figsize=(8, 6))
                    x_values = [t.params[p_name] for t in trials if p_name in t.params]
                    y_values = [t.value for t in trials if p_name in t.params]
                    
                    plt.scatter(x_values, y_values, alpha=0.6)
                    plt.xlabel(p_name)
                    plt.ylabel('Metric Value')
                    plt.title(f'Slice Plot: {p_name}')
                    if p_def.get('log', False):
                        plt.xscale('log')
                    plt.grid(True, alpha=0.3)
                    plt.savefig(os.path.join(output_dir, f'slice_{p_name}.png'))
                    plt.close()
                except Exception as e:
                    print(f"Error plotting slice for {p_name}: {e}")

            print(f"Visualizations saved to {output_dir}")

        except ImportError:
            print("matplotlib or pandas not installed. Skipping visualization.")
        except Exception as e:
            print(f"Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()

    def search(self, output_dir=None):
        sampler = optuna.samplers.TPESampler(seed=self.optuna_seed)
        print(f"Setting Optuna TPE Sampler seed to: {self.optuna_seed}")

        self.study = optuna.create_study(
            direction="maximize" if self.maximize else "minimize",
            sampler=sampler
        )
        
        patience = getattr(self.args, 'patience', 20)
        print(f"Starting Bayesian Optimization with {self.args.n_trials} trials (Patience: {patience}).")
        
        class EarlyStoppingCallback:
            def __init__(self, patience, maximize):
                self.patience = patience
                self.maximize = maximize
                self.best_score = -float('inf') if maximize else float('inf')
                self.no_improve_count = 0
            
            def __call__(self, study, trial):
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    current_score = trial.value
                    
                    is_better = current_score > self.best_score if self.maximize else current_score < self.best_score
                    
                    if is_better:
                        self.best_score = current_score
                        self.no_improve_count = 0
                    else:
                        self.no_improve_count += 1
                        
                    if self.patience is not None and self.no_improve_count >= self.patience:
                        print(f"\n[Early Stopping] Stopping optimization. No improvement for {self.patience} trials.")
                        study.stop()

        callbacks = [EarlyStoppingCallback(patience, self.maximize)]
        
        self.study.optimize(self.objective, n_trials=self.args.n_trials, callbacks=callbacks)

        print("\n" + "="*50)
        print("Optimization Finished")
        if len(self.study.trials) > 0:
            print(f"Best Value: {self.study.best_value}")
            print(f"Best Params: {self.study.best_params}")
        else:
            print("No trials completed.")
        print("="*50)
        
        self.cleanup()
        
        if output_dir:
            self.save_results(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Optimization with Optuna")
    parser.add_argument('--dataset_config', type=str, required=True)
    parser.add_argument('--model_config', type=str, required=True)
    
    # Legacy args for backward compatibility (optional, but good for testing)
    parser.add_argument('--param', type=str, required=False)
    parser.add_argument('--type', type=str, default='float', choices=['float', 'int', 'categorical'])
    parser.add_argument('--range', type=str, required=False)
    parser.add_argument('--log', action='store_true')
    
    # Common args
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--metric', type=str, default='NDCG@10')
    parser.add_argument('--direction', type=str, default='max', choices=['max', 'min'])
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--patience', type=int, default=20)
    
    # Note: Complex 'params' list usually passed via python API in run_all_smart_searches.py
    # If called from CLI with multi-params, we'd need a JSON argument or similar.
    # For now, CLI primarily supports single param or simple legacy mode.
    # The run_all_smart_searches.py constructs the Args object directly with list.
    
    args = parser.parse_args()
    
    # Check if we are running in legacy CLI mode
    if args.param and args.range and not hasattr(args, 'params'):
        optimizer = BayesianOptimizer(args)
        optimizer.search()
    else:
        print("For multi-parameter optimization, please use scripts/run_all_smart_searches.py or pass 'params' list in API.")
