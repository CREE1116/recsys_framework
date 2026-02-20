import argparse
import yaml
import os
import sys
import json
import copy
from smart_grid_search import SmartGridSearch
from bayesian_opt import BayesianOptimizer

class Args:
    """Helper class to convert dictionary to object with attributes"""
    def __init__(self, **entries):
        self.__dict__.update(entries)

def run_all_searches(config_path, output_dir_base, cli_args=None):
    print(f"Loading batch configuration from {config_path}...")
    with open(config_path, 'r') as f:
        batch_config = yaml.safe_load(f)

    # Allow legacy config (no top-level datasets) for backward compatibility
    datasets = batch_config.get('datasets', [])
    search_definitions = batch_config.get('searches', [])
    
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
                # Define output directory for this search
                search_output_dir = os.path.join(output_dir_base, dataset_name, sub_name)
                print(f"  -> Output directory: {search_output_dir}")

                best_metric = None
                best_params = {}
                best_dir = None
                
                if method == 'grid':
                    # Grid search currently only supports single param in args.param
                    # But we might need to parse it if it returns complex struct. 
                    searcher = SmartGridSearch(args)
                    searcher.search()
                    best_metric = searcher.best_global_metric
                    best_dir = searcher.best_global_dir
                    best_params = {args.param: 'See best_dir'} 
                elif method == 'bayesian':
                    searcher = BayesianOptimizer(args)
                    # Pass output_dir to search
                    try:
                        searcher.search(output_dir=search_output_dir)
                    finally:
                        searcher.cleanup(keep_best=False)
                    
                    best_metric = searcher.study.best_value if hasattr(searcher, 'study') and searcher.study.trials else None
                    best_dir = searcher.best_global_dir
                    best_params = searcher.study.best_params if hasattr(searcher, 'study') and searcher.study.trials else {}
                
                # Store result
                model_name = os.path.basename(args.model_config).replace('.yaml', '')
                
                result_entry = {
                    'dataset': dataset_name,
                    'model': model_name,
                    'search_name': sub_name,
                    'best_metric': best_metric,
                    'best_params': best_params,
                    'best_dir': best_dir,
                    'details_dir': search_output_dir
                }
                
                dataset_results[model_name] = result_entry

                # Save individual model result immediately
                model_res_file = os.path.join(search_output_dir, f"result_{dataset_name}_{model_name}.json")
                with open(model_res_file, 'w') as f:
                    json.dump(result_entry, f, indent=4)
                print(f"Saved result for {model_name} to {model_res_file}")

                # --- NEW: Run Final Visualization for Best Model ---
                if best_params:
                    print(f"\n>>> Running Final Visualization for BEST {model_name} (Params: {best_params})")
                    try:
                        # 1. Load base config again to be clean
                        # We need to reconstruct the config used for search but with best_params
                        best_config = copy.deepcopy(args.__dict__)
                        # remove args-specific keys that are not part of model config
                        # actually args is derived from search_def which IS the config structure we need (mostly)
                        
                        # Better approach: 
                        # We used SmartGridSearch/BayesianOptimizer. 
                        # They merge dataset_config and model_config.
                        # We should do the same manually or use a helper.
                        
                        # Load raw configs
                        with open(search_def['dataset_config'], 'r') as f:
                             ds_conf = yaml.safe_load(f)
                        with open(search_def['model_config'], 'r') as f:
                             md_conf = yaml.safe_load(f)
                        
                        # Merge (Model overrides Dataset)
                        final_config = copy.deepcopy(ds_conf)
                        for k, v in md_conf.items():
                             if k in final_config and isinstance(final_config[k], dict) and isinstance(v, dict):
                                 final_config[k].update(v)
                             else:
                                 final_config[k] = v
                                 
                        # Apply Best Params
                        # params are like "model.reg_lambda"
                        for param_key, param_val in best_params.items():
                            keys = param_key.split('.')
                            target = final_config
                            for k in keys[:-1]:
                                target = target.setdefault(k, {})
                            target[keys[-1]] = param_val
                            
                        # Set Visualization Flags
                        if 'model' not in final_config: final_config['model'] = {}
                        final_config['model']['visualize'] = True
                        final_config['model']['save_model'] = True # Save the best model
                        
                        # Set Run Name
                        final_config['run_name'] = f"BEST_{model_name}_{sub_name}"
                        # Output dir for this specific run will be constructed by main/Trainer
                        
                        # Device
                        if args.device:
                            final_config['device'] = args.device
                        elif 'device' not in final_config or final_config['device'] is None:
                            final_config['device'] = 'cpu' # Default fallback

                        # Run!
                        from main import main as run_single_experiment # Ensure import
                        run_single_experiment(final_config)
                        print(f">>> Final Visualization Completed for {model_name}")
                        
                    except Exception as e:
                        print(f"Failed to run final visualization for {model_name}: {e}")
                        import traceback
                        traceback.print_exc()

            except Exception as e:
                print(f"Error executing search '{full_search_name}': {e}")
                import traceback
                traceback.print_exc()

        # End of dataset loop: Save aggregated best params for this dataset
        if use_global_datasets and dataset_results:
            agg_file = os.path.join(output_dir_base, dataset_name, f"best_hyperparameters_{dataset_name}.json")
            print(f"\nSaving aggregated best hyperparameters for {dataset_name} to {agg_file}...")
            os.makedirs(os.path.dirname(agg_file), exist_ok=True)
            with open(agg_file, 'w') as f:
                json.dump(dataset_results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple smart searches from a config file")
    parser.add_argument('--config', type=str, required=True, help='Path to batch config YAML')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save results')
    parser.add_argument('--n_trials', type=int, default=None, help='Override number of trials for bayesian search')
    parser.add_argument('--patience', type=int, default=None, help='Override patience for bayesian search')
    
    cli_args = parser.parse_args()
    
    run_all_searches(cli_args.config, cli_args.output_dir, cli_args)
