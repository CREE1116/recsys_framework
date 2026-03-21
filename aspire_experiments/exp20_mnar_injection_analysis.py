"""
exp20_mnar_injection_analysis.py

엄밀한 MNAR 편향 실험:
1. Popularity-Reweighted Resampling (Density 고정)
2. Inner-Loop Alpha HPO (공정한 비교)
3. Isolated Test Set (편향 주입 = train만)
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import optuna
from datetime import datetime
from scipy.sparse import csr_matrix

sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import load_config, get_loader_and_svd
from src.models.csar.ASPIRE import ASPIRE
from src.evaluation import evaluate_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# 핵심 1: Constant-Density Resampling
# ─────────────────────────────────────────────────────────────────────────────

def popularity_reweighted_resample(X, gamma_bias=1.0, seed=42):
    """
    총 관측수(N)를 기대값으로 유지하면서 인기도 분포만 편향시킵니다.

    각 관측 (u,i)에 독립 Bernoulli 샘플링 적용:
        keep_prob(u,i) = min(w_i * N / Σ w_j, 1.0)
        where  w_i = n_i^gamma_bias

    gamma_bias=0  → 균일 샘플링 (MCAR)
    gamma_bias=1  → 원본 분포 유지
    gamma_bias>1  → 인기 아이템 과다 대표 (강한 MNAR)
    """
    rng = np.random.default_rng(seed)
    X_coo = X.tocoo()
    N = float(X_coo.nnz)

    item_counts = np.array(X.sum(axis=0)).flatten().astype(float)
    item_counts = np.maximum(item_counts, 1)

    w = item_counts ** gamma_bias          # (n_items,)
    obs_weights = w[X_coo.col].astype(float)  # 각 엔트리의 가중치

    # keep_prob ∝ w_i, E[kept] = N
    keep_prob = obs_weights * N / obs_weights.sum()
    keep_prob = np.minimum(keep_prob, 1.0)

    mask = rng.random(int(N)) < keep_prob

    R_biased = csr_matrix(
        (X_coo.data[mask], (X_coo.row[mask], X_coo.col[mask])),
        shape=X.shape
    )
    print(f"  [Resample] N={int(N)} → Kept={R_biased.nnz} "
          f"({R_biased.nnz/N*100:.1f}%), "
          f"density={R_biased.nnz/(X.shape[0]*X.shape[1])*100:.4f}%")
    return R_biased

# ─────────────────────────────────────────────────────────────────────────────
# 핵심 2: Inner-Loop Alpha HPO
# ─────────────────────────────────────────────────────────────────────────────

def find_best_alpha(R_biased, gamma_model, loader, config,
                    n_trials=15, alpha_range=(0.01, 100.0)):
    """
    주어진 (R_biased, gamma)에 대해 최적 alpha를 탐색합니다.
    Validation set으로 평가합니다.
    """
    valid_loader = loader.get_validation_loader(batch_size=1024)

    def objective(trial):
        alpha = trial.suggest_float("alpha", *alpha_range, log=True)
        cfg = config.copy()
        cfg['model'] = dict(config.get('model', {}))
        cfg['model']['gamma'] = gamma_model
        cfg['model']['alpha'] = alpha
        cfg['model']['visualize'] = False
        cfg['device'] = 'cpu'

        model = ASPIRE(cfg, loader)
        model.train_matrix_csr = R_biased
        model.lira_layer.build(R_biased, verbose=False)

        eval_cfg = {'top_k': [10], 'metrics': ['NDCG'], 'method': 'full'}
        metrics = evaluate_metrics(model, loader, eval_cfg,
                                   device=torch.device("cpu"),
                                   test_loader=valid_loader)
        return metrics.get('NDCG@10', 0.0)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params['alpha'], study.best_value

# ─────────────────────────────────────────────────────────────────────────────
# 실험 메인
# ─────────────────────────────────────────────────────────────────────────────

def run_mnar_experiment(dataset="ml100k", n_alpha_trials=15):
    print(f"\n--- [Exp 20v2] Rigorous MNAR Analysis on {dataset} ---")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    res_dir = os.path.join(base_dir, "results", f"exp20v2_mnar_{dataset}_{timestamp}")
    plot_dir = os.path.join(base_dir, "plots")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    loader, R_train, _, _, config = get_loader_and_svd(dataset)
    test_loader = loader.get_final_loader(batch_size=1024)

    bias_strengths = [round(x, 1) for x in np.arange(0.0, 5.1, 0.5)]
    gamma_values   = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    data_log = []
    results_map = {}

    for gb in bias_strengths:
        print(f"\n[Bias γ={gb}] Resampling train data...")
        R_biased = popularity_reweighted_resample(R_train, gamma_bias=gb)
        density  = R_biased.nnz / (R_biased.shape[0] * R_biased.shape[1])
        orig_density = R_train.nnz / (R_train.shape[0] * R_train.shape[1])
        print(f"  Density: {density*100:.4f}% | Original: {orig_density*100:.4f}%")

        for gm in gamma_values:
            print(f"  ASPIRE(γ={gm}) | HPO alpha [{n_alpha_trials} trials]...", end='', flush=True)

            best_alpha, best_val_ndcg = find_best_alpha(
                R_biased, gm, loader, config, n_trials=n_alpha_trials)

            # 최적 alpha로 재학습 후 테스트 평가
            cfg = config.copy()
            cfg['model'] = dict(config.get('model', {}))
            cfg['model']['gamma'] = gm
            cfg['model']['alpha'] = best_alpha
            cfg['model']['visualize'] = False
            cfg['device'] = 'cpu'

            model = ASPIRE(cfg, loader)
            model.train_matrix_csr = R_biased
            model.lira_layer.build(R_biased, verbose=False)

            eval_cfg = {'top_k': [10, 20], 'metrics': ['NDCG', 'Recall'], 'method': 'full'}
            metrics  = evaluate_metrics(model, loader, eval_cfg,
                                        device=torch.device("cpu"),
                                        test_loader=test_loader)

            diag = model.diagnostics()
            row = {
                "gamma_bias":     gb,
                "gamma_model":    gm,
                "best_alpha":     best_alpha,
                "effective_alpha": diag['effective_alpha'],
                "rho":            diag['rho'],
                "val_ndcg@10":    best_val_ndcg,
                "ndcg@10":        metrics.get('NDCG@10', 0),
                "ndcg@20":        metrics.get('NDCG@20', 0),
                "recall@10":      metrics.get('Recall@10', 0),
                "density":        density,
            }
            data_log.append(row)
            results_map[(gb, gm)] = row['ndcg@10']

            print(f" α*={best_alpha:.4f} | Test NDCG@10={row['ndcg@10']:.4f}")

    # ── Save ──
    df = pd.DataFrame(data_log)
    csv_path = os.path.join(res_dir, "results.csv")
    os.makedirs(res_dir, exist_ok=True)  # 이중 보호
    df.to_csv(csv_path, index=False)

    meta = {
        "dataset": dataset,
        "timestamp": timestamp,
        "bias_strengths": bias_strengths,
        "gamma_values": gamma_values,
        "n_alpha_trials": n_alpha_trials,
        "note": "Constant-density resampling + per-combo alpha HPO"
    }
    with open(os.path.join(res_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=4)
    print(f"\nData saved to {res_dir}")

    # ── Plot ──
    markers = ['o', 's', '^', 'D', 'x']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for i, gm in enumerate(gamma_values):
        y = [results_map[(gb, gm)] for gb in bias_strengths]
        label = f"γ_model={gm}" + (" [EASE]" if gm == 2.0 else "")
        ax.plot(bias_strengths, y, marker=markers[i % len(markers)], label=label)
    ax.set_title(f"NDCG@10 vs MNAR Bias ({dataset})")
    ax.set_xlabel("Bias Strength (γ_bias)")
    ax.set_ylabel("NDCG@10")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    # 각 gamma_bias에서 optimal gamma_model 추적
    opt_gammas = []
    for gb in bias_strengths:
        best_gm = max(gamma_values, key=lambda gm: results_map[(gb, gm)])
        opt_gammas.append(best_gm)
    ax.plot(bias_strengths, opt_gammas, marker='o', color='firebrick', linewidth=2)
    ax.set_title("Optimal γ_model vs Bias Strength")
    ax.set_xlabel("Bias Strength (γ_bias)")
    ax.set_ylabel("Optimal γ_model")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(plot_dir, f"exp20v2_mnar_{dataset}.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)  # 이중 보호
    plt.savefig(plot_path, dpi=150)
    plt.savefig(os.path.join(res_dir, "analysis_plot.png"), dpi=150)
    print(f"Plot saved to {plot_path}")

    # ── Optimal γ Analysis ──
    print("\n=== Optimal γ_model per Bias Strength ===")
    for gb, gm in zip(bias_strengths, opt_gammas):
        print(f"  γ_bias={gb:.1f} → optimal γ_model={gm:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml100k")
    parser.add_argument("--n_alpha_trials", type=int, default=15,
                        help="알파 HPO 탐색 횟수 (속도-정확도 트레이드오프)")
    args = parser.parse_args()

    run_mnar_experiment(args.dataset, n_alpha_trials=args.n_alpha_trials)
