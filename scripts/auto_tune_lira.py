import os
import yaml
import json
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from src.utils.lambda_optimizer import optimize_from_dict, visualize_joint, find_intersection

def run_experiment(dataset_config, model_config_path, reg_lambdas):
    """
    Temporary override LIRA config and run grid search.
    """
    # 1. Load original LIRA config
    with open(model_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Update reg_lambda to the anchor points
    config['model']['reg_lambda'] = reg_lambdas
    config['model']['normalize'] = [True] # Ensure normalization is on for consistency
    
    temp_config_path = 'configs/model/csar/lira_auto_tune_temp.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"[AutoTuner] Running anchor experiments for lambda: {reg_lambdas}...")
    
    # 3. Call grid_search.py
    cmd = [
        "python", "grid_search.py",
        "--dataset_config", dataset_config,
        "--model_config", temp_config_path
    ]
    subprocess.run(cmd, check=True)
    
    # Cleanup temp config
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)

def collect_results(results_base_dir, reg_lambdas):
    """
    Find NDCG@10 and Coverage@10 from the result directories.
    Handles various naming conventions (with/without float decimals, with/without normalize suffix).
    """
    ndcg_data = {}
    coverage_data = {}
    
    if not os.path.exists(results_base_dir):
        # Try dash vs no-dash or other common patterns
        if "-" in results_base_dir:
            alt_dir = results_base_dir.replace("-", "")
        else:
            if "ml100k" in results_base_dir: alt_dir = results_base_dir.replace("ml100k", "ml-100k")
            elif "ml1m" in results_base_dir: alt_dir = results_base_dir.replace("ml1m", "ml-1m")
            else: alt_dir = results_base_dir
        
        if os.path.exists(alt_dir):
            results_base_dir = alt_dir

    print(f"[AutoTuner] Searching for results in: {results_base_dir}")

    for l in reg_lambdas:
        l_int = int(l) if l == int(l) else l
        
        candidates = [
            f"lira__reg_lambda={l}",
            f"lira__reg_lambda={l_int}",
            f"lira__reg_lambda={l}_normalize=True",
            f"lira__reg_lambda={l_int}_normalize=True"
        ]
        
        found = False
        for cand in candidates:
            path = os.path.join(results_base_dir, cand, "final_metrics.json")
            if os.path.exists(path):
                with open(path, 'r') as f:
                    metrics = json.load(f)
                    ndcg_data[l] = metrics.get('NDCG@10', 0.0)
                    coverage_data[l] = metrics.get('Coverage@10', 0.0)
                print(f" [Found] {cand}")
                found = True
                break
        
        if not found:
            print(f" [Missing] No result found for lambda={l} after checking candidates.")
            
    return ndcg_data, coverage_data

def main():
    parser = argparse.ArgumentParser(description="Auto-Tune LIRA Lambda for NDCG and Coverage Peaks.")
    parser.add_argument('--dataset_config', type=str, required=True, help='Path to dataset config (e.g., configs/dataset/ml100k.yaml)')
    parser.add_argument('--model_config', type=str, default='configs/model/csar/lira.yaml', help='Path to LIRA config.')
    parser.add_argument('--anchors', type=float, nargs='+', default=[100, 1000, 2500,5000,10000, 100000], help='Lambda anchor points.')
    args = parser.parse_args()

    # Smart dataset name extraction from the YAML itself
    with open(args.dataset_config, 'r') as f:
        d_conf = yaml.safe_load(f)
    dataset_name_from_conf = d_conf.get('dataset_name', os.path.basename(args.dataset_config).replace('.yaml', ''))
    
    dataset_base = dataset_name_from_conf
    results_base_dir = f"trained_model/{dataset_base}"
    
    # 1. Run experiments if not already present
    run_experiment(args.dataset_config, args.model_config, args.anchors)
    
    # 2. Collect
    ndcg_dict, cov_dict = collect_results(results_base_dir, args.anchors)
    
    if len(ndcg_dict) < 3:
        print(f"[AutoTuner] Error: Could not collect at least 3 points (found {len(ndcg_dict)}). Optimization aborted.")
        return

    # 4. Save visualization
    save_dir = os.path.join(results_base_dir, "auto_tuning_analysis")
    os.makedirs(save_dir, exist_ok=True)
    
    opt_ndcg = optimize_from_dict(ndcg_dict, "NDCG@10")
    opt_cov = optimize_from_dict(cov_dict, "Coverage@10")
    sweet_lambda, prod_lambda = visualize_joint(opt_ndcg, opt_cov, save_path=os.path.join(save_dir, "tune_joint.png"))
    
    print(f"\n[NDCG Prediction]")
    # Handle possible a >= 0 in log-space correctly displayed
    best_l = max(ndcg_dict, key=ndcg_dict.get)
    print(f" - Best Sample: Lambda={best_l}, Val={ndcg_dict[best_l]:.4f}")
    print(f" - Predicted Peak λ: {opt_ndcg.peak_lambda:.2f} (Estimated NDCG: {opt_ndcg.peak_val:.4f})")

    print(f"\n[Coverage Prediction]")
    best_l_cov = max(cov_dict, key=cov_dict.get)
    print(f" - Best Sample: Lambda={best_l_cov}, Val={cov_dict[best_l_cov]:.4f}")
    print(f" - Predicted Peak λ: {opt_cov.peak_lambda:.2f} (Estimated Coverage: {opt_cov.peak_val:.4f})")

    print(f"\n[Joint Optimization]")
    sweet_log_l = np.log10(sweet_lambda)
    sweet_ndcg = opt_ndcg.predict(sweet_log_l)
    sweet_cov = opt_cov.predict(sweet_log_l)
    print(f" - Balanced Sweet Spot λ: {sweet_lambda:.2f} (Normalized Intersection)")
    print(f"   > Est. NDCG: {sweet_ndcg:.4f}, Est. Coverage: {sweet_cov:.4f}")

    prod_log_l = np.log10(prod_lambda)
    prod_ndcg = opt_ndcg.predict(prod_log_l)
    prod_cov = opt_cov.predict(prod_log_l)
    print(f" - Max Product (NDCG * Cov) λ: {prod_lambda:.2f}")
    print(f"   > Est. NDCG: {prod_ndcg:.4f}, Est. Coverage: {prod_cov:.4f}")

    opt_ndcg.visualize(save_path=os.path.join(save_dir, "tune_ndcg.png"))
    opt_cov.visualize(save_path=os.path.join(save_dir, "tune_coverage.png"))
    
    # 5. Final Report
    report = {
        "dataset": dataset_base,
        "anchors": args.anchors,
        "measured_ndcg": {str(k): v for k, v in ndcg_dict.items()},
        "measured_coverage": {str(k): v for k, v in cov_dict.items()},
        "predicted_ndcg_peak_lambda": float(opt_ndcg.peak_lambda),
        "predicted_coverage_peak_lambda": float(opt_cov.peak_lambda),
        "balanced_sweet_spot_lambda": float(sweet_lambda),
        "max_product_lambda": float(prod_lambda)
    }
    
    with open(os.path.join(save_dir, "tuning_report.json"), 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"\n[AutoTuner] Optimization Complete. Report saved to {save_dir}/tuning_report.json")
    print("="*60)

if __name__ == "__main__":
    main()
