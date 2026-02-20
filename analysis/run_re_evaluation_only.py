
import os
import argparse
import glob
import torch
import yaml
import json
import pandas as pd
from tqdm import tqdm
from src.data_loader import DataLoader
from src.evaluation import evaluate_metrics
from src.models import get_model
from src.models import get_model
# from src.utils import seed_everything # Not available in src.utils

def evaluate_single_model_dir(model_dir, data_loader, device):
    """
    Evaluates a single model directory.
    """
    config_path = os.path.join(model_dir, "config.yaml")
    metrics_path = os.path.join(model_dir, "final_metrics.json")
    
    if not os.path.exists(config_path):
        print(f"[SKIP] Missing config in {model_dir}")
        return None
        
    # Load config early
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Proceed to re-evaluate regardless of mtime to ensure consistency
    
    # Checkpoint candidates
    ckpt_candidates = ["best_model.pt", "saved_model.pth"]
    model_path = None
    for ckpt in ckpt_candidates:
        p = os.path.join(model_dir, ckpt)
        if os.path.exists(p):
            model_path = p
            break

    print(f"\n[EVAL] Evaluating {os.path.basename(model_dir)}...")
    
    # Update device
    config['device'] = device
    
    # Initialize Model
    # Note: Some models might require specific init args from config
    try:
        model_name = config['model']['name']
        model = get_model(model_name, config, data_loader).to(device)
        
        if model_path:
            print(f"Loading checkpoint from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"No checkpoint found. Attempting to fit (One-Shot)...")
            if hasattr(model, 'fit'):
                # Check if fit accepts data_loader (most do in this framework)
                import inspect
                sig = inspect.signature(model.fit)
                if 'data_loader' in sig.parameters:
                    model.fit(data_loader)
                else:
                    model.fit()
            elif model_name in ['random-rec', 'most-popular']:
                # These might not have fit or fit is empty
                pass
            else:
                print(f"[WARN] No checkpoint and no fit method for {model_name}. Skipping.")
                return None
                
        model.eval()
    except Exception as e:
        print(f"[ERROR] Failed to load model in {model_dir}: {e}")
        return None

    # Run Evaluation
    eval_config = config['evaluation']
    # Force 'long_tail_percent' to be updated if passed/defaulted in recent code
    # Actually, current evaluation.py uses default if not 0.8 in config.
    # But files on disk might still have 0.2. 
    # We should override it here to ensure consistency with current experiment goal?
    # User said they want to re-eval. Current code defaults to 0.8? Or did we revert to 0.2?
    # In step 873 we implemented Volume-Based Cutoff, but config param is head_volume_percent/long_tail_percent.
    # It's safest to rely on evaluation.py's internal logic which we just updated.
    
    # However, create test loader
    # Assuming batch size from config
    batch_size = eval_config.get('batch_size', 2048)
    # data_loader.config needs to be consistent with model evaluation config?
    # Actually get_final_loader uses self.config (data_loader.config).
    # We initialized data_loader with 'full' in main().
    # If model config specifies something else, we should update data_loader.config temporarily?
    # But usually all models use 'full'.
    
    # Optional: Update data_loader config to match model config's evaluation method
    original_eval_config = data_loader.config.get('evaluation', {})
    data_loader.config['evaluation'] = eval_config
    
    test_loader = data_loader.get_final_loader(batch_size=batch_size)
    
    metrics = evaluate_metrics(model, data_loader, eval_config, device, test_loader, is_final=True)
    
    # Save metrics
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"[DONE] Metrics saved to {metrics_path}")
    
    # Return metrics with model name
    metrics['model'] = config['model']['name']
    metrics['Run'] = os.path.basename(model_dir)
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Re-evaluate all models in a directory.")
    parser.add_argument('models_dir', type=str, help="Directory containing model subdirectories (e.g. trained_model/ml-1m)")
    parser.add_argument('--dataset_config', type=str, default='configs/dataset/ml1m.yaml', help="Path to dataset config used for training")
    parser.add_argument('--device', type=str, default='auto', help="Device to use")
    parser.add_argument('--output', type=str, default='re_evaluation_results.csv', help="Output CSV summary")

    args = parser.parse_args()
    
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load Dataset ONCE
    print(f"Loading dataset from {args.dataset_config}...")
    with open(args.dataset_config, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Add dummy evaluation config for DataLoader init
    if 'evaluation' not in data_config:
        data_config['evaluation'] = {'validation_method': 'full', 'final_method': 'full'}
    
    data_loader = DataLoader(data_config)
    
    # Find all model directories
    # Assuming any subdir in models_dir that has config.yaml
    subdirs = sorted([d for d in glob.glob(os.path.join(args.models_dir, "*")) if os.path.isdir(d)])
    
    all_results = []
    
    for subdir in tqdm(subdirs, desc="Evaluating Models"):
        try:
            res = evaluate_single_model_dir(subdir, data_loader, device)
            if res:
                all_results.append(res)
        except Exception as e:
            print(f"[ERROR] Failed to evaluate {os.path.basename(subdir)}: {e}")
            import traceback
            traceback.print_exc()
            
    # Save combined CSV
    if all_results:
        df = pd.DataFrame(all_results)
        # Reorder columns
        cols = ['Run', 'model'] + [c for c in df.columns if c not in ['Run', 'model']]
        df = df[cols]
        df.to_csv(args.output, index=False)
        print(f"\n[SUCCESS] Aggregated results saved to {args.output}")
    else:
        print("\n[WARN] No results generated.")

if __name__ == "__main__":
    main()
