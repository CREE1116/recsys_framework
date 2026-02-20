import os
import json
import yaml
import csv
import argparse
from pathlib import Path

def summarize_results(root_dir, output_file):
    """
    Summarizes experiment results from a directory into a CSV file.
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"Error: Directory {root_dir} does not exist.")
        return

    results = []
    all_keys = set()
    
    # Iterate through each model folder
    for model_dir in root_path.iterdir():
        if not model_dir.is_dir():
            continue
            
        config_path = model_dir / "config.yaml"
        metrics_path = model_dir / "final_metrics.json"
        
        if not (config_path.exists() and metrics_path.exists()):
            continue
            
        print(f"Processing {model_dir.name}...")
        
        row = {
            "folder_name": model_dir.name,
        }
        
        # 1. Load Config
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                # Prefix model params with 'param_'
                model_params = config.get('model', {})
                for k, v in model_params.items():
                    row[f"param_{k}"] = v
        except Exception as e:
            print(f"  Warning: Failed to load config in {model_dir.name}: {e}")

        # 2. Load Metrics
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                for k, v in metrics.items():
                    row[k] = v
        except Exception as e:
            print(f"  Warning: Failed to load metrics in {model_dir.name}: {e}")
            
        results.append(row)
        all_keys.update(row.keys())

    if not results:
        print("No results found to summarize.")
        return

    # Sort keys: folder_name first, then params, then metrics
    sorted_keys = ["folder_name"]
    param_keys = sorted([k for k in all_keys if k.startswith("param_")])
    metric_keys = sorted([k for k in all_keys if k not in sorted_keys and k not in param_keys])
    
    fieldnames = sorted_keys + param_keys + metric_keys

    # Write to CSV
    try:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"\nSuccessfully saved summary to: {output_file}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize experiment results into CSV")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the dataset result folder (e.g. trained_model/ml-100k)")
    parser.add_argument("--output", type=str, default="experiment_summary.csv", help="Output CSV filename")
    
    args = parser.parse_args()
    summarize_results(args.input_dir, args.output)
