"""
exp21_rho_gamma_mapping.py
==========================
ρ → γ 매핑 가능성 검증 실험.

각 데이터셋에서:
1. τ=auto (MP 추정) 고정
2. γ ∈ [0.1, 2.0] 스윕 → 각각 NDCG@10 측정
3. 최적 γ*를 기록
4. ρ vs γ* 관계 시각화

가설: γ* ≈ 2 - c * ρ (선형 관계)
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aspire_experiments.exp_utils import get_loader_and_svd
from src.models.csar.ASPIRE import ASPIRE
from src.models.csar.ASPIRELayer import AspireFilter
from src.utils.gpu_accel import EVDCacheManager
from src.evaluation import evaluate_metrics

# ── 데이터셋 목록 ──────────────────────────────────────────────
DATASETS = ["ml100k", "ml1m", "steam"]

# ── 실험 설정 ──────────────────────────────────────────────────
GAMMA_SWEEP = [round(x, 2) for x in np.arange(0.1, 2.01, 0.2)]  # 10 points

def run_single(dataset: str, tau_method: str = "spectral_gap"):
    print(f"\n{'='*60}")
    print(f"[Exp 21] Dataset: {dataset}  tau_method={tau_method}")
    print('='*60)

    loader, R_train, _, _, config = get_loader_and_svd(dataset)
    test_loader = loader.get_final_loader(batch_size=1024)

    # ── EVD & ρ 계산 ──────────────────────────────────────────
    dev = torch.device("cpu")
    manager = EVDCacheManager(device=dev)
    _, s, v, _ = manager.get_evd(R_train, dataset_name=dataset)

    rho = AspireFilter.compute_rho(s)
    tau_auto = AspireFilter.estimate_tau(s, X_sparse=R_train, method=tau_method)

    print(f"ρ={rho:.4f} | τ_auto({tau_method})={tau_auto:.4f}")

    # ── γ 스윕 ────────────────────────────────────────────────
    results = []
    for gamma in GAMMA_SWEEP:
        config_copy = dict(config)
        config_copy['model'] = dict(config.get('model', {}))
        config_copy['model']['tau'] = tau_auto
        config_copy['model']['gamma'] = gamma
        config_copy['model']['target_energy'] = 1.0
        config_copy['model']['visualize'] = False
        config_copy['device'] = 'cpu'           # BaseModel 필수 키
        config_copy['dataset_name'] = dataset

        try:
            model = ASPIRE(config_copy, loader)
            model.eval()

            eval_cfg = {'top_k': [10, 20], 'metrics': ['NDCG', 'Recall'], 'method': 'full'}
            metrics = evaluate_metrics(
                model, loader, eval_cfg,
                device=torch.device("cpu"),
                test_loader=test_loader
            )
            ndcg10   = metrics.get('NDCG@10', 0.0)
            ndcg20   = metrics.get('NDCG@20', 0.0)
            recall10 = metrics.get('Recall@10', 0.0)
            results.append({
                "dataset": dataset,
                "rho": rho,
                "tau_auto": tau_auto,
                "gamma": gamma,
                "ndcg@10": ndcg10,
                "ndcg@20": ndcg20,
                "recall@10": recall10,
            })
            print(f"  γ={gamma:.2f} → NDCG@10={ndcg10:.4f}")
        except Exception as e:
            print(f"  γ={gamma:.2f} → FAILED: {e}")

    if not results:
        return None

    df = pd.DataFrame(results)
    best_idx = df['ndcg@10'].idxmax()
    best_gamma = df.loc[best_idx, 'gamma']
    best_ndcg = df.loc[best_idx, 'ndcg@10']
    print(f"\n  → Optimal γ*={best_gamma:.2f} (NDCG@10={best_ndcg:.4f}) | ρ={rho:.4f}")

    return df, rho, tau_auto, best_gamma


def run_all(datasets, tau_method="spectral_gap"):
    res_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results",
        f"exp21_rho_gamma_{tau_method}"
    )
    os.makedirs(res_dir, exist_ok=True)

    all_rows = []
    summary = []

    for ds in datasets:
        out = run_single(ds, tau_method=tau_method)
        if out is None:
            continue
        df, rho, tau_auto, best_gamma = out
        all_rows.append(df)
        summary.append({
            "dataset": ds,
            "rho": rho,
            "tau_auto": tau_auto,
            "best_gamma": best_gamma,
            # linear pred: gamma_pred = 2 - 1.5*rho
            "gamma_pred_c1.5": round(2 - 1.5 * rho, 4),
            "gamma_pred_c2.0": round(2 - 2.0 * rho, 4),
        })

    if not all_rows:
        print("No results!")
        return

    full_df = pd.concat(all_rows, ignore_index=True)
    full_df.to_csv(os.path.join(res_dir, "full_results.csv"), index=False)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(res_dir, "summary.csv"), index=False)
    print(f"\n{'='*60}")
    print(summary_df.to_string(index=False))

    # ── 시각화 ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 1) ρ vs γ* scatter
    ax = axes[0]
    rhos = summary_df['rho'].values
    bests = summary_df['best_gamma'].values
    ax.scatter(rhos, bests, s=120, zorder=5, label='Observed γ*')
    for _, row in summary_df.iterrows():
        ax.annotate(row['dataset'], (row['rho'], row['best_gamma']),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)

    # 선형 예측
    rho_line = np.linspace(0, 1, 100)
    ax.plot(rho_line, 2 - 1.5 * rho_line, 'r--', label='γ=2-1.5ρ')
    ax.plot(rho_line, 2 - 2.0 * rho_line, 'b--', label='γ=2-2.0ρ')
    ax.set_xlabel('ρ (SPP)', fontsize=12)
    ax.set_ylabel('Optimal γ*', fontsize=12)
    ax.set_title('ρ → γ* Mapping Feasibility', fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 2) γ sweep curves per dataset
    ax2 = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(summary)))
    for (_, row), color in zip(summary_df.iterrows(), colors):
        ds = row['dataset']
        sub = full_df[full_df['dataset'] == ds]
        ax2.plot(sub['gamma'], sub['ndcg@10'], marker='o', label=ds, color=color)
        ax2.axvline(row['best_gamma'], color=color, linestyle=':', alpha=0.5)

    ax2.set_xlabel('γ', fontsize=12)
    ax2.set_ylabel('NDCG@10', fontsize=12)
    ax2.set_title('γ Sweep per Dataset (τ=auto)', fontsize=13, fontweight='bold')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(res_dir, "rho_gamma_mapping.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    print(f"Results saved to {res_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--tau_method", default="spectral_gap",
                        choices=["mp", "spectral_gap", "median"],
                        help="tau auto-estimation method")
    args = parser.parse_args()
    run_all(args.datasets, tau_method=args.tau_method)
