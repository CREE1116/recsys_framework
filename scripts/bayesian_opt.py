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

        # Check dataset dimensions to clamp K/Rank
        try:
            print(f"[BayesianOptimizer] Loading dataset to check dimensions...")
            # Create a shallow copy and ensure 'evaluation' exists for DataLoader safety
            check_config = copy.deepcopy(self.dataset_config)
            if 'evaluation' not in check_config:
                check_config['evaluation'] = {'validation_method': 'holdout', 'final_method': 'holdout'}
            
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
                if 'range' not in param:
                    # Auto range: 1 to max_dim
                    print(f"[BayesianOptimizer] Auto-setting range for '{param['name']}' to 1~{max_dim}")
                    param['range'] = f"1 {max_dim}"
                else:
                    # Existing range: clamp upper bound
                    low, high = map(int, param['range'].split())
                    if high > max_dim:
                        print(f"[BayesianOptimizer] Clamping parameter '{param['name']}' max from {high} to {max_dim}")
                        new_high = max_dim
                        if low >= new_high:
                            low = max(1, new_high // 2)
                            print(f"[BayesianOptimizer] Adjusted low bound to {low}")
                        param['range'] = f"{low} {new_high}"
                
                # Convert type back to 'int' for Optuna compatibility
                param['type'] = 'int'

        self.metric_key = args.metric
        self.maximize = (args.direction == 'max')
        self.best_global_metric = -float('inf') if self.maximize else float('inf')
        self.best_global_dir = None
        self.all_experiment_dirs = []

    def load_yaml(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def merge_configs(self, dataset_conf, model_conf):
        config = copy.deepcopy(dataset_conf)
        for key, value in model_conf.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
        return config

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
        print(f"\n>>> Running experiment: {config['run_name']}")
        
        exp_dir = self.get_experiment_dir(config)
        metrics_file = os.path.join(exp_dir, 'final_metrics.json')
        
        if os.path.exists(metrics_file):
            print(f"Results already exist at {exp_dir}. Loading...")
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
            except:
                print("Failed to load existing metrics. Re-running...")
                metrics = None
        else:
            metrics = None

        if metrics is None:
            try:
                run_single_experiment(config)
                with open(metrics_file, 'r') as f:
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
        if metric_val is None:
             for k, v in metrics.items():
                if self.metric_key in k:
                    metric_val = v
                    break
        
        if metric_val is None:
            print(f"Warning: Metric {self.metric_key} not found.")
            return -float('inf') if self.maximize else float('inf'), exp_dir
            
        return metric_val, exp_dir

    def cleanup(self, keep_best=True):
        print("\n=== Cleaning up checkpoints ===")
        saved_count = 0
        deleted_count = 0
        for path in self.all_experiment_dirs:
            if keep_best and path == self.best_global_dir:
                print(f"KEEPING Best: {path}")
                saved_count += 1
            elif os.path.exists(path):
                # print(f"DELETING: {path}") # Reduce noise
                try:
                    shutil.rmtree(path)
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {path}: {e}")
        print(f"Cleanup complete. Kept {saved_count}, Deleted {deleted_count}.")

    def objective(self, trial):
        current_params = {}
        for p_def in self.params_list:
            name = p_def['name']
            p_type = p_def.get('type', 'float')
            p_range = p_def['range']
            p_log = p_def.get('log', False)

            if p_type == 'float':
                low, high = map(float, p_range.split())
                val = trial.suggest_float(name, low, high, log=p_log)
            elif p_type == 'int':
                low, high = map(int, p_range.split())
                val = trial.suggest_int(name, low, high, log=p_log)
            elif p_type == 'categorical':
                if isinstance(p_range, str):
                    choices = p_range.split()
                else:
                    choices = p_range
                # Try to convert
                try:
                    choices = [int(c) for c in choices]
                except:
                    try:
                        choices = [float(c) for c in choices]
                    except:
                        pass
                val = trial.suggest_categorical(name, choices)
            else:
                raise ValueError(f"Unknown type: {p_type}")
            
            current_params[name] = val
            
        print(f"\n[Trial {trial.number}] Params: {current_params}")

        metric, exp_dir = self.run_experiment(current_params)

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
        self.study = optuna.create_study(direction="maximize" if self.maximize else "minimize")
        
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
                        
                    if self.no_improve_count >= self.patience:
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
