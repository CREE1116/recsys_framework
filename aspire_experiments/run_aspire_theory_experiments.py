# Usage: uv run python aspire_experiments/run_aspire_theory_experiments.py --datasets ml100k ml1m steam
#
# ASPIRE Theory Experiments — Master Runner (Cleaned)
#
# exp01: Power-law Coupling (Beta Estimation)
# exp02: Method Ablation (NDCG Comparison for Estimators)
# exp03: Beta Ablation (Recall Curve vs Beta Scale)
# exp04: Popularity Restoration Visualization
#

import os
import sys
import argparse
import pandas as pd
import json

sys.path.append(os.getcwd())

from aspire_experiments.exp01_power_law import run_power_law
from aspire_experiments.exp02_method_ablation import run_method_ablation
from aspire_experiments.exp03_beta_ablation import run_beta_ablation
from aspire_experiments.exp04_popularity_restoration import run_popularity_restoration
from aspire_experiments.exp_utils import ensure_dir

def main():
    parser = argparse.ArgumentParser(description="Run ASPIRE Theory Experiments (Cleaned)")
    parser.add_argument("--datasets", nargs="+", default=["ml1m"], help="Dataset names")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Random seeds")
    parser.add_argument("--skip", nargs="*", default=[], choices=["exp01", "exp02", "exp03", "exp04"], help="Skip specific experiments")
    parser.add_argument("--trials", type=int, default=30, help="HPO trials for exp02")
    args = parser.parse_args()

    summary_data = []

    for dataset in args.datasets:
        print(f"\n{'='*60}")
        print(f"  ASPIRE Theory Space  |  dataset={dataset}")
        print(f"{'='*60}")
        
        # 01. Power-law Coupling
        if "exp01" not in args.skip:
            print("\n[Exp 01] Power-law Coupling & Beta Estimation")
            run_power_law(dataset, seed=args.seeds[0])

        # 04. Popularity Restoration
        if "exp04" not in args.skip:
            print("\n[Exp 04] Popularity-Spectral Restoration Visualization")
            run_popularity_restoration(dataset)

        for seed in args.seeds:
            # 02. Method Ablation
            if "exp02" not in args.skip:
                print(f"\n[Exp 02] Method Ablation (NDCG Comparison, seed={seed})")
                run_method_ablation(dataset, n_trials=args.trials, seed=seed)

            # 03. Beta Ablation
            if "exp03" not in args.skip:
                print(f"\n[Exp 03] Beta Ablation (Recall Scan, seed={seed})")
                run_beta_ablation(dataset, n_trials=args.trials, seed=seed)
        
        try:
            def _load(exp_path, ds, filename):
                for ds_name in [ds, ds.replace("ml", "ml-")]:
                    p = os.path.join(f"aspire_experiments/output/{exp_path}", ds_name, filename)
                    if os.path.exists(p):
                        with open(p, encoding="utf-8") as f:
                            return json.load(f)
                return None

            r1 = _load("powerlaw", dataset, "result.json")
            r2 = _load("method_ablation", dataset, "results.json")
            
            row = {"Dataset": dataset}
            if r1: 
                row["Beta_LAD"] = r1.get("beta_lad")
                row["Beta_OLS"] = r1.get("beta_ols")
                row["Vector_Opt"] = r1.get("beta_vector_opt")
            if r2:
                best = sorted(r2, key=lambda x: x['NDCG@10'], reverse=True)[0]
                row["Best_Method"] = best["method"]
                row["Max_NDCG"] = best["NDCG@10"]

            summary_data.append(row)
        except Exception as e:
            print(f"  [Warning] Summary failed ({dataset}): {e}")

    if summary_data:
        summary_dir = ensure_dir("aspire_experiments/output/summary")
        df = pd.DataFrame(summary_data)
        csv_path = os.path.join(summary_dir, "theory_cleaned_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n{'#'*60}\nMASTER SUMMARY\n{'#'*60}")
        print(df.to_string(index=False))

    print("\nAll Core ASPIRE experiments completed.")

if __name__ == "__main__":
    main()
