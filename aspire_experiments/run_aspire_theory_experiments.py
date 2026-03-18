# Usage: uv run python aspire_experiments/run_aspire_theory_experiments.py --datasets ml100k ml1m steam
#
# ASPIRE Theory Experiments — Unified Master Runner v3 (Deterministic Beta)
#
# exp01: SLP Verification       (Spectral Popularity Matrix 대각 우세 검증)
# exp02: Power-law Coupling     (SPP vs Singular Value 멱법칙 시각화)
# exp03: Theory Unification     (Direct Slope-Ratio vs Projection-based Beta 일치성)
# exp04: Spectral Restoration   (필터 적용 후 스텍트럼 Power-law 회복 실증)
# exp06: Method Ablation         (베타 추론 방법별 NDCG 성능 비교 어블레이션)
# exp07: Wiener vs ASPIRE        (Wiener 필터와 ASPIRE 필터의 추천 성능 비교)
# exp08: Beta Ablation           (β값 스캔 및 Recall 성능 변화 검증)
# exp10: Popularity Restoration  (특이값-인기도 상관관계 복원 시각화)

import os
import sys
import argparse
import pandas as pd
import json

sys.path.append(os.getcwd())

from aspire_experiments.exp01_slp               import run_slp
from aspire_experiments.exp02_power_law         import run_power_law
from aspire_experiments.exp03_unification       import run_unification
from aspire_experiments.exp04_restoration       import run_restoration
from aspire_experiments.exp06_method_ablation   import run_method_ablation
from aspire_experiments.exp07_wiener_vs_aspire  import run_filter_comparison
from aspire_experiments.exp08_beta_ablation     import run_beta_ablation
from aspire_experiments.exp10_popularity_restoration import run_popularity_restoration
from aspire_experiments.exp_utils import ensure_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run ASPIRE Theory Experiments (Master Script v3)"
    )
    parser.add_argument("--datasets",  nargs="+",  default=["ml1m"],
                        help="Dataset names (e.g. ml1m ml100k)")

    parser.add_argument("--seeds",     type=int,   nargs="+", default=[42],
                        help="Random seeds for sampling/reproducibility")
    parser.add_argument("--skip",      nargs="*",  default=[],
                        choices=["exp01","exp02","exp03","exp04","exp06","exp07","exp08","exp10"],
                        help="건너뛸 실험 목록")
    parser.add_argument("--trials",    type=int,   default=30,
                        help="exp06용 HPO trial 수")
    args = parser.parse_args()

    summary_data = []

    for dataset in args.datasets:
        print(f"\n{'='*60}")
        print(f"  ASPIRE Theory Experiments v3  |  dataset={dataset}")
        print(f"{'='*60}")

        seed = args.seeds[0]
        
        # 01. SLP
        if "exp01" not in args.skip:
            print("\n[Exp 01] SLP Verification")
            run_slp(dataset) # Keeping simple as per user request (uses default 42)

        # 02. Power-law Coupling
        if "exp02" not in args.skip:
            print("\n[Exp 02] Power-law Coupling (LAD/Ratio)")
            run_power_law(dataset)

        # 03. Theory Unification
        if "exp03" not in args.skip:
            print("\n[Exp 03] Theory Unification (Direct vs Projection)")
            run_unification(dataset)

        # 04. Spectral Restoration
        if "exp04" not in args.skip:
            print("\n[Exp 04] Spectral Restoration Analysis")
            run_restoration(dataset)

        # HPO-based experiments (support multiple seeds)
        for seed in args.seeds:
            # 06. Method Ablation
            if "exp06" not in args.skip:
                print(f"\n[Exp 06] Method Ablation (NDCG Comparison, seed={seed})")
                run_method_ablation(dataset, n_trials=args.trials, seed=seed)

            # 07. Filter Comparison (Wiener vs ASPIRE)
            if "exp07" not in args.skip:
                print(f"\n[Exp 07] Wiener vs. ASPIRE Filter Comparison (seed={seed})")
                run_filter_comparison(dataset, n_trials=args.trials, seed=seed)

            # 08. Beta Ablation
            if "exp08" not in args.skip:
                print(f"\n[Exp 08] Beta Ablation (Recall Scan, seed={seed})")
                run_beta_ablation(dataset, n_trials=args.trials, seed=seed)

        # 10. Popularity Restoration Visualization
        if "exp10" not in args.skip:
            print("\n[Exp 10] Popularity-Spectral Restoration Visualization")
            run_popularity_restoration(dataset)

        # 요약 데이터 수집
        try:
            def _load(exp_path, ds, filename):
                for ds_name in [ds, ds.replace("ml", "ml-")]:
                    p = os.path.join(f"aspire_experiments/output/{exp_path}", ds_name, filename)
                    if os.path.exists(p):
                        with open(p, encoding="utf-8") as f:
                            return json.load(f)
                return None

            r1 = _load("slp", dataset, "result.json")
            r2 = _load("powerlaw", dataset, "result.json")
            r4 = _load("spectral_restoration", dataset, "result.json")
            r6 = _load("method_ablation", dataset, "results.json")
            r7 = _load("filter_comp", dataset, "result.json")
            # Currently ignoring r8 because its mapping is complex

            row = {"Dataset": dataset}
            if r1: 
                row["SLP_MeanRatio"] = r1.get("mean_ratio")
                row["SLP_Rho"] = r1.get("rho_frob")
            if r2: 
                row["Beta_LAD"] = r2.get("beta_lad")
                row["Beta_OLS"] = r2.get("beta_ols")
            if r4:
                row["Gamma"] = r4.get("gamma")
                row["Restored_R2"] = r4.get("r2_restored")
            if r6:
                # Get best method and its NDCG (r6 is a list of results)
                best = sorted(r6, key=lambda x: x['NDCG@10'], reverse=True)[0]
                row["Best_Method"] = best["method"]
                row["Max_NDCG"] = best["NDCG@10"]
            if r7:
                row["Tail_Gain"] = r7.get("tail_gain")

            summary_data.append(row)
        except Exception as e:
            print(f"  [Warning] 요약 수집 실패 ({dataset}): {e}")

    if summary_data:
        summary_dir = ensure_dir("aspire_experiments/output/summary")
        df = pd.DataFrame(summary_data)
        csv_path = os.path.join(summary_dir, "theory_v3_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n{'#'*60}\nMASTER SUMMARY (Theory v3)\n{'#'*60}")
        print(df.to_string(index=False))

    print("\nAll Slimmed ASPIRE theory experiments completed.")

if __name__ == "__main__":
    main()
