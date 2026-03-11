import os
import sys
import argparse
import pandas as pd
import json

# Framework root path
sys.path.append(os.getcwd())

from aspire_experiments.exp1_slp import run_slp
from aspire_experiments.exp2_power_law import run_power_law
from aspire_experiments.exp3_beta_tracking import run_beta_tracking
from aspire_experiments.exp4_targeted_subsampling import run_beta_tracking_v2
from aspire_experiments.exp5_ablation import run_ablation
from aspire_experiments.exp6_corollary2 import run_corollary2
from aspire_experiments.exp_utils import ensure_dir

def main():
    parser = argparse.ArgumentParser(description="Run ASPIRE Theory Experiments (Master Script)")
    parser.add_argument("--datasets", nargs='+', default=["ml1m"], help="Dataset names (e.g., ml1m ml100k)")
    parser.add_argument("--energy", type=float, default=0.95, help="Target energy for SVD rank")
    parser.add_argument("--seeds", type=int, nargs='+', default=[42], help="Random seeds (for Exp 3, 4)")
    args = parser.parse_args()

    summary_data = []

    for dataset in args.datasets:
        print(f"\n{'='*60}")
        print(f"PROCESS DATASET: {dataset}")
        print(f"{'='*60}")

        # Exp 1: SLP Verification
        run_slp(dataset, target_energy=args.energy)

        # Exp 2: Power-law Coupling
        run_power_law(dataset, target_energy=args.energy)

        # Exp 3: Beta Tracking (MCAR Injection)
        # v3 implementation supports multiple datasets and has its own aggregation
        run_beta_tracking(dataset, target_energy=args.energy)

        # Exp 4: Targeted Subsampling
        # v2 implementation supports seeds and has its own aggregation
        run_beta_tracking_v2(dataset, target_energy=args.energy, n_seeds=len(args.seeds))

        # Exp 5: Beta Estimation Ablation
        # New implementation compares 5 methods with NDCG@10
        run_ablation(dataset, target_energy=args.energy)

        # Exp 6: Corollary 2 Verification
        run_corollary2([dataset], target_energy=args.energy)

        # Collect summary metrics (if result files exist)
        try:
            # Result folders might have hyphens even if argument doesn't (or vice versa)
            def find_latest_res(exp_type, ds_name):
                base = f"aspire_experiments/output/{exp_type}"
                search_params = [ds_name, ds_name.replace("ml", "ml-")]
                # For ablation, check both versions
                filenames = ["result_per_method.json", "result.json"] if exp_type == "ablation" else ["result.json"]
                for p in search_params:
                    for f_name in filenames:
                        target = os.path.join(base, p, f_name)
                        if os.path.exists(target): return target
                return None

            res1_path = find_latest_res("slp", dataset)
            res2_path = find_latest_res("powerlaw", dataset)
            # ablation uses result.json in the standalone script
            res5_path = find_latest_res("ablation", dataset)
            res6_path = find_latest_res("corollary2", dataset)
            
            if not res1_path or not res2_path or not res5_path or not res6_path:
                raise FileNotFoundError(f"Missing results for {dataset}")

            with open(res1_path, 'r', encoding='utf-8') as f: r1 = json.load(f)
            with open(res2_path, 'r', encoding='utf-8') as f: r2 = json.load(f)
            with open(res5_path, 'r', encoding='utf-8') as f: r5 = json.load(f)
            with open(res6_path, 'r', encoding='utf-8') as f: r6 = json.load(f)["experiments"][0]
            
            # Huber is [0], HPO is [6], Direct is [4], Damped is [5] in the 7-method list
            huber_res = next(m for m in r5["methods"] if "(1)" in m["method"])
            hpo_res   = next(m for m in r5["methods"] if "(5)" in m["method"])
            direct_res = next(m for m in r5["methods"] if "(6)" in m["method"])
            damped_res = next(m for m in r5["methods"] if "(7)" in m["method"])
            
            summary_data.append({
                "Dataset": dataset,
                "SLP_Epsilon": r1["epsilon"],
                "PowerLaw_Beta": r2["beta"],
                "PowerLaw_R2": r2["r2"],
                "Exp6_Beta_Theory": r6["beta_theory"],
                "Exp6_Beta_Meas": r6["beta_measured"],
                "Exp6_Rel_Err": r6["rel_error"],
                "Ablation_Huber_NDCG": huber_res["ndcg_all"],
                "Ablation_Huber_Tail": huber_res["ndcg_tail"],
                "Ablation_Huber_Cov": huber_res["coverage"],
                "Ablation_Damped_NDCG": damped_res["ndcg_all"],
                "Ablation_Direct_NDCG": direct_res["ndcg_all"],
                "Ablation_HPO_NDCG": hpo_res["ndcg_all"],
            })
        except Exception as e:
            print(f"  Warning: Could not collect summary for {dataset}: {e}")

    # Final Summary Table
    if summary_data:
        summary_dir = ensure_dir("aspire_experiments/output/summary")
        df = pd.DataFrame(summary_data)
        df.to_csv(os.path.join(summary_dir, "master_summary.csv"), index=False)
        print(f"\n{'#'*60}")
        print(f"MASTER SUMMARY")
        print(f"{'#'*60}")
        print(df.to_string(index=False))
        print(f"\nFinal summary saved in {summary_dir}")

    print("\nAll integrated experiments completed.")

if __name__ == "__main__":
    main()
