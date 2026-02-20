import argparse
import yaml
import os
import sys
import shutil
import glob
import json
import copy
import math
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import main as run_single_experiment

class SmartGridSearch:
    def __init__(self, args):
        self.args = args
        self.dataset_config = self.load_yaml(args.dataset_config)
        self.model_config = self.load_yaml(args.model_config)
        self.base_config = self.merge_configs(self.dataset_config, self.model_config)
        
        # Override specific args
        if args.device:
            self.base_config['device'] = args.device

        self.param_path = args.param.split('.')
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

    def set_nested_value(self, config, path, value):
        temp = config
        for key in path[:-1]:
            temp = temp.setdefault(key, {})
        temp[path[-1]] = value

    def get_experiment_dir(self, config):
        # Replicates the logic in Trainer.__init__ to find where the model was saved
        model_name = config['model']['name']
        dataset_name = config['dataset_name']
        run_name = config.get('run_name')
        
        base_path = os.path.join('trained_model', dataset_name)
        if run_name and run_name != 'default':
            folder_name = f"{model_name}__{run_name}"
        else:
            folder_name = model_name
            
        return os.path.join(base_path, folder_name)

    def run_experiment(self, value):
        print(f"\n>>> Running experiment with {self.args.param} = {value}")
        config = copy.deepcopy(self.base_config)
        
        # Determine strict type for the parameter if possible (default to float/int based on value)
        if isinstance(value, int):
            val_to_set = int(value)
        elif isinstance(value, float):
            val_to_set = float(value)
        else:
            val_to_set = value
            
        self.set_nested_value(config, self.param_path, val_to_set)
        
        # Create a unique run name
        param_name = self.param_path[-1]
        run_name = f"{param_name}={val_to_set:.6g}" # Use general format to avoid trailing zeros
        config['run_name'] = run_name
        
        # Prevent re-running if results exist (optional, but good for restartability)
        exp_dir = self.get_experiment_dir(config)
        metrics_file = os.path.join(exp_dir, 'final_metrics.json')
        
        if os.path.exists(metrics_file):
            print(f"Results already exist at {exp_dir}. Loading...")
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        else:
            try:
                # Capture standard output to reduce clutter if needed, but for now let it stream
                run_single_experiment(config)
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
            except Exception as e:
                print(f"Experiment failed: {e}")
                import traceback
                traceback.print_exc()
                return -float('inf') if self.maximize else float('inf'), None

        self.all_experiment_dirs.append(exp_dir)
        
        # Extract metric
        # Handle @k keys (e.g. NDCG@10)
        metric_val = metrics.get(self.metric_key)
        if metric_val is None:
            # Try to find a loose match
            for k, v in metrics.items():
                if self.metric_key in k:
                    metric_val = v
                    print(f"Identified metric {k} as target for {self.metric_key}")
                    break
        
        if metric_val is None:
            print(f"Warning: Metric {self.metric_key} not found in results.")
            return -float('inf') if self.maximize else float('inf'), exp_dir
            
        return metric_val, exp_dir

    def is_better(self, current, best):
        if self.maximize:
            return current > best
        else:
            return current < best

    def cleanup(self):
        print("\n=== Cleaning up inferior checkpoints ===")
        saved_count = 0
        deleted_count = 0
        for path in self.all_experiment_dirs:
            if path == self.best_global_dir:
                print(f"KEEPING Best: {path}")
                saved_count += 1
            elif os.path.exists(path):
                print(f"DELETING: {path}")
                try:
                    shutil.rmtree(path)
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {path}: {e}")
        print(f"Cleanup complete. Kept {saved_count}, Deleted {deleted_count}.")

    def search(self):
        # Phase 1: Log Scale Search
        print("\n=== Phase 1: Log Scale Search ===")
        start_exp, end_exp = map(int, self.args.range.split())
        
        best_exp = start_exp
        phase1_best_metric = -float('inf') if self.maximize else float('inf')
        
        # Iterate exponents
        # e.g. -3, -2, -1, 0, 1, 2, 3
        exponents = list(range(start_exp, end_exp + 1))
        metrics = []
        
        for exp in exponents:
            val = 10 ** exp
            metric, exp_dir = self.run_experiment(val)
            metrics.append(metric)
            
            if self.is_better(metric, self.best_global_metric):
                self.best_global_metric = metric
                self.best_global_dir = exp_dir
                phase1_best_metric = metric
                best_exp = exp
                print(f"New Best found: {metric} at 10^{exp}")
            
            # Simple hill climbing check during log phase? 
            # User said "Scan log scale, then go to that scale".
            # It's safer to scan the whole specified log range first to find the global peak region.
            
        print(f"Phase 1 Best: 10^{best_exp} (Metric: {phase1_best_metric})")
        
        # Determine refinement range
        # User: "In that scale 1~9"
        # If 10^0 (1) is best, we search 1, 2, ..., 9.
        # If 10^1 (10) is best, we search 10, 20, ..., 90.
        # But wait, what if 10^2 is best? Range 100-900.
        # What if 10^-1 is best? Range 0.1, 0.2... 0.9.
        
        # So Base Step = 10^best_exp.
        # But we really want to check the *leading digit*.
        # The "Log search" checked 1 * 10^k.
        # Now we check d * 10^k for d in 1..9.
        # Note: We already checked d=1 (which is 1*10^k).
        # We also checked 10 * 10^k (which is 1*10^{k+1}).
        
        # Let's search [2, 3, ..., 9] * 10^best_exp AND [2, 3, ..., 9] * 10^{best_exp-1}?
        # Actually proper "significant digit" refinement usually implies looking around the best log point.
        # If 100 was best (better than 10 and 1000), the optimal is likely in [10, 100] or [100, 1000].
        # User said: "In that scale 1~9". This usually means 100, 200, ... 900 if 100 was best?
        # Or does it mean if 100 is best, we go down to 10s?
        
        # Let's assume the user means: Find order of magnitude 10^k.
        # Then search d * 10^k for d=1..9. (e.g., if k=2, search 100, 200...900).
        # We stop when metric drops.
        
        # Phase 2: Digit Search
        print(f"\n=== Phase 2: Digit Search for scale 10^{best_exp} ===")
        scale = 10 ** best_exp
        best_digit = 1
        current_best_metric = self.best_global_metric 
        
        # We already have result for digit=1 (it is 1 * 10^best_exp).
        # We need to decide if we go UP (1, 2, 3...) or if we should check if we missed something between 10^(best-1) and 10^best.
        # If 100 > 1000 and 100 > 10, then peak is around 100.
        # It could be 80, 90, 100, 110, 120...
        
        # Let's follow the "1~9" instruction literally first.
        # Search 2*scale, 3*scale ... 9*scale.
        # If 2*scale > 1*scale, success. Continue.
        # If 3*scale < 2*scale, STOP. Fix at 2.
        
        # However, what if the optimal is e.g. 0.8 * scale?
        # The Log search found 10^k. Matches 1, 10, 100.
        # If 100 is best, we check 200, 300...
        # We rely on the log phase to have found the right "starting 1".
        
        for d in range(2, 10):
            val = d * scale
            metric, exp_dir = self.run_experiment(val)

            if self.is_better(metric, current_best_metric):
                print(f"Improvement: {val} (Metric: {metric} > {current_best_metric})")
                current_best_metric = metric
                best_digit = d
                self.best_global_metric = metric
                self.best_global_dir = exp_dir
            else:
                print(f"Drop detected at {val} (Metric: {metric} <= {current_best_metric}). Stopping.")
                break
        
        print(f"Phase 2 Best: {best_digit * scale} (Digit: {best_digit})")
        
        # Phase 3: Refinement (One level deeper)
        # "Go to lower scale and repeat"
        # Current best is V = best_digit * 10^k.
        # Lower scale -> 10^{k-1}.
        # Search V + 1*10^{k-1}, V + 2*10^{k-1}...
        
        # Example: Phase 2 found 300 (3 * 10^2).
        # Next scale is 10^1 = 10.
        # Search 310, 320, 330...
        
        base_val = best_digit * scale
        sub_scale = scale / 10.0
        
        # If sub_scale is too small (e.g. float precision issues or meaningless), maybe stop?
        # But for regularizers (0.001 etc), it's fine.
        
        print(f"\n=== Phase 3: Fine Refinement (Scale {sub_scale}) ===")
        # Search base + d * sub_scale
        
        best_sub_digit = 0 # 0 means we stay at base_val
        current_best_metric = self.best_global_metric
        
        for d in range(1, 10):
            val = base_val + (d * sub_scale)
            # floating point correction
            if isinstance(sub_scale, float) or isinstance(base_val, float):
                 val = float(f"{val:.10g}")

            metric, exp_dir = self.run_experiment(val)
            
            if self.is_better(metric, current_best_metric):
                print(f"Improvement: {val} (Metric: {metric} > {current_best_metric})")
                current_best_metric = metric
                best_sub_digit = d
                self.best_global_metric = metric
                self.best_global_dir = exp_dir
            else:
                 print(f"Drop detected at {val} (Metric: {metric} <= {current_best_metric}). Stopping.")
                 break

        final_val = base_val + (best_sub_digit * sub_scale)
        print(f"\nFinal Best Parameter Found: {final_val}")
        print(f"Final Metric: {self.best_global_metric}")
        print(f"Location: {self.best_global_dir}")
        
        self.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Grid Search")
    parser.add_argument('--dataset_config', type=str, required=True, help='Path to dataset yaml')
    parser.add_argument('--model_config', type=str, required=True, help='Path to model yaml')
    parser.add_argument('--param', type=str, required=True, help='Parameter path (e.g., model.reg_lambda)')
    parser.add_argument('--range', type=str, required=True, help='Log scale range "start end" (e.g., "-3 3")')
    parser.add_argument('--metric', type=str, default='NDCG@10', help='Target metric')
    parser.add_argument('--direction', type=str, default='max', choices=['max', 'min'], help='Optimization direction')
    parser.add_argument('--device', type=str, default=None, help='Device override')
    
    args = parser.parse_args()
    
    searcher = SmartGridSearch(args)
    searcher.search()
