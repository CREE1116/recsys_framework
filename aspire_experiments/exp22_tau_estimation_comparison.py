"""
exp22_tau_estimation_comparison.py
===================================
Auto-τ 추정 방법론 비교 실험.

각 데이터셋에서 다음을 비교:
  - tau_mp      : Marchenko-Pastur noise edge (√density)
  - tau_gap     : log-scale 특이값 변곡점 (spectral_gap)
  - tau_median  : 정규화 특이값 중간값
  - tau_hpo     : (존재 시) HPO로 찾은 최적 tau (ground truth)

각 추정 방법별로 γ ∈ GAMMA_SWEEP을 스윕하여
  NDCG@10, NDCG@20, Recall@10 측정 후 시각화.
"""

import argparse
import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aspire_experiments.exp_utils import get_loader_and_svd
from src.models.csar.ASPIRE import ASPIRE
from src.models.csar.ASPIRELayer import AspireFilter
from src.utils.gpu_accel import EVDCacheManager
from src.evaluation import evaluate_metrics

# ─── 설정 ─────────────────────────────────────────────────────
DATASETS   = ["ml100k", "ml1m", "steam"]
GAMMA_SWEEP = [round(x, 2) for x in np.arange(0.1, 2.01, 0.25)]   # 8 points
TAU_METHODS = ["mp", "spectral_gap", "median"]
BEST_DIR    = "trained_model"   # HPO 최적 모델 저장 위치


# ─── HPO best tau 읽기 ─────────────────────────────────────────
def load_hpo_best_tau(dataset: str) -> float | None:
    """trained_model/<dataset>/BEST_aspire_seed_*/config.yaml 에서 tau 읽기"""
    patterns = [
        f"{BEST_DIR}/{dataset}/BEST_aspire_seed_*/config.yaml",
        f"{BEST_DIR}/{dataset.replace('100k', '-100k').replace('1m', '-1m')}/BEST_aspire_seed_*/config.yaml",
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            import yaml
            with open(hits[0]) as f:
                cfg = yaml.safe_load(f)
            tau = cfg.get('model', {}).get('tau', None)
            if tau is not None and isinstance(tau, (int, float)):
                return float(tau)
    return None


# ─── 단일 (dataset, tau_method) 실험 ──────────────────────────
def run_sweep(dataset: str, tau_val: float, tau_label: str,
              loader, R_train, config, test_loader) -> pd.DataFrame:
    rows = []
    for gamma in GAMMA_SWEEP:
        cfg = dict(config)
        cfg['model'] = dict(config.get('model', {}))
        cfg['model']['tau'] = tau_val
        cfg['model']['gamma'] = gamma
        cfg['model']['target_energy'] = 1.0
        cfg['model']['visualize'] = False
        cfg['device'] = 'cpu'
        cfg['dataset_name'] = dataset

        try:
            model = ASPIRE(cfg, loader)
            model.eval()
            eval_cfg = {'top_k': [10, 20], 'metrics': ['NDCG', 'Recall'], 'method': 'full'}
            metrics = evaluate_metrics(model, loader, eval_cfg,
                                       device=torch.device("cpu"),
                                       test_loader=test_loader)
            rows.append({
                "dataset": dataset,
                "tau_method": tau_label,
                "tau_val": tau_val,
                "gamma": gamma,
                "ndcg@10": metrics.get('NDCG@10', 0.0),
                "ndcg@20": metrics.get('NDCG@20', 0.0),
                "recall@10": metrics.get('Recall@10', 0.0),
            })
            print(f"    γ={gamma:.2f} → NDCG@10={rows[-1]['ndcg@10']:.4f}")
        except Exception as e:
            print(f"    γ={gamma:.2f} → FAILED: {e}")
    return pd.DataFrame(rows)


# ─── 단일 데이터셋 전체 실험 ─────────────────────────────────
def run_dataset(dataset: str) -> pd.DataFrame:
    print(f"\n{'='*65}")
    print(f"[Exp 22] Dataset: {dataset}")
    print('='*65)

    loader, R_train, _, _, config = get_loader_and_svd(dataset)
    test_loader = loader.get_final_loader(batch_size=2048)

    dev = torch.device("cpu")
    manager = EVDCacheManager(device=dev)
    _, s, _, _ = manager.get_evd(R_train, dataset_name=dataset)

    rho = AspireFilter.compute_rho(s)
    print(f"ρ = {rho:.4f}")

    all_frames = []

    # Auto methods
    for method in TAU_METHODS:
        tau_val = AspireFilter.estimate_tau(s, X_sparse=R_train, method=method)
        print(f"\n  [{method.upper()}] τ = {tau_val:.4f}")
        df = run_sweep(dataset, tau_val, method, loader, R_train, config, test_loader)
        all_frames.append(df)

    # HPO ground truth (있는 경우만)
    hpo_tau = load_hpo_best_tau(dataset)
    if hpo_tau is not None:
        print(f"\n  [HPO] τ = {hpo_tau:.4f}")
        df = run_sweep(dataset, hpo_tau, "hpo", loader, R_train, config, test_loader)
        all_frames.append(df)
    else:
        print(f"\n  [HPO] 저장된 best config 없음. 스킵.")

    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()


# ─── 시각화 ──────────────────────────────────────────────────
def plot_results(full_df: pd.DataFrame, res_dir: str):
    datasets = full_df['dataset'].unique()
    methods  = full_df['tau_method'].unique()
    colors   = {m: c for m, c in zip(methods, cm.tab10(np.linspace(0, 1, len(methods))))}
    styles   = {"mp": "-", "spectral_gap": "--", "median": ":", "hpo": "-"}
    markers  = {"mp": "o", "spectral_gap": "s", "median": "^", "hpo": "*"}

    fig, axes = plt.subplots(len(datasets), 2, figsize=(13, 4*len(datasets)))
    if len(datasets) == 1:
        axes = axes[None, :]

    for row_i, ds in enumerate(datasets):
        sub_ds = full_df[full_df['dataset'] == ds]

        # ── τ 값 막대그래프 ──────────────────────────────────
        ax_bar = axes[row_i, 0]
        tau_summary = sub_ds.groupby('tau_method')['tau_val'].first()
        bars = ax_bar.bar(tau_summary.index, tau_summary.values,
                          color=[colors[m] for m in tau_summary.index])
        for bar, (m, v) in zip(bars, tau_summary.items()):
            ax_bar.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        ax_bar.set_title(f'{ds} — Estimated τ per Method', fontweight='bold')
        ax_bar.set_ylabel('τ value')
        ax_bar.set_ylim(0, max(tau_summary.values) * 1.25)
        ax_bar.grid(True, axis='y', alpha=0.3)

        # ── γ 스윕 NDCG 곡선 ─────────────────────────────────
        ax_ndcg = axes[row_i, 1]
        for method in methods:
            sub_m = sub_ds[sub_ds['tau_method'] == method]
            if sub_m.empty:
                continue
            ax_ndcg.plot(sub_m['gamma'], sub_m['ndcg@10'],
                         label=f"{method} (τ={sub_m['tau_val'].iloc[0]:.3f})",
                         color=colors[method],
                         linestyle=styles.get(method, '-'),
                         marker=markers.get(method, 'o'))
        ax_ndcg.set_xlabel('γ')
        ax_ndcg.set_ylabel('NDCG@10')
        ax_ndcg.set_title(f'{ds} — NDCG@10 vs γ (by τ method)', fontweight='bold')
        ax_ndcg.legend(fontsize=8)
        ax_ndcg.grid(True, alpha=0.3)

    plt.suptitle('Auto-τ Estimation Method Comparison (ASPIRE)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(res_dir, "tau_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\n[Exp 22] Plot saved → {path}")


# ─── Summary 테이블 ──────────────────────────────────────────
def print_summary(full_df: pd.DataFrame):
    print(f"\n{'='*65}")
    print("SUMMARY: Best NDCG@10 per (dataset, tau_method)")
    print('='*65)
    summary = (full_df.groupby(['dataset', 'tau_method', 'tau_val'])
               ['ndcg@10'].max()
               .reset_index()
               .rename(columns={'ndcg@10': 'best_ndcg@10'}))
    print(summary.to_string(index=False))


# ─── 메인 ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    args = parser.parse_args()

    res_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "exp22_tau_comparison"
    )
    os.makedirs(res_dir, exist_ok=True)

    all_frames = []
    for ds in args.datasets:
        df = run_dataset(ds)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        print("결과 없음.")
        return

    full_df = pd.concat(all_frames, ignore_index=True)
    full_df.to_csv(os.path.join(res_dir, "results.csv"), index=False)

    print_summary(full_df)
    plot_results(full_df, res_dir)
    print(f"\n[Exp 22] 완료. 결과 저장: {res_dir}")


if __name__ == "__main__":
    main()
