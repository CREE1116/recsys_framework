# Usage: uv run python aspire_experiments/exp4_targeted_subsampling.py --dataset ml1m --energy 0.95
import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# Framework root path
sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import get_loader_and_svd, ensure_dir
from src.models.csar.ASPIRELayer import AspireEngine

# ─── 핵심: Targeted Subsampling ──────────────────────────────────────────────

def targeted_subsample(R_orig, eta_target, target_density=0.5, seed=42):
    """
    아이템별 인기도 분포를 η_target으로 제어한 서브샘플링.
    
    Args:
        R_orig:          원본 CSR 행렬
        eta_target:      목표 인기도 멱법칙 지수 (0=MCAR, ↑=강한 MNAR)
        target_density:  원본 밀도 대비 유지 비율 (기본 50%)
        seed:            재현성
    
    Returns:
        R_sub: 서브샘플링된 CSR 행렬
        keep_probs: 아이템별 실제 keep probability
    """
    rng = np.random.default_rng(seed)
    n_users, n_items = R_orig.shape
    R_csc = R_orig.tocsc()

    # 아이템 인기도 및 rank
    pop = np.array(R_orig.sum(axis=0)).flatten()          # raw count
    item_rank = np.argsort(np.argsort(-pop)) + 1           # 1-indexed, 높은 인기 = rank 1

    if eta_target == 0.0:
        # MCAR: 모든 아이템에 동일한 keep probability
        total_keep = int(R_orig.nnz * target_density)
        uniform_prob = total_keep / R_orig.nnz
        keep_probs = np.full(n_items, uniform_prob)
    else:
        # MNAR: rank^{-eta_target} 에 비례하는 keep probability
        target_pop = item_rank.astype(float) ** (-eta_target)

        # C 결정: 전체 기대 상호작용 수 = R_orig.nnz * target_density
        # E[keep] = Σ_i n_i · min(1, C·target_pop[i]/n_i)
        # C를 이진탐색으로 결정
        def expected_nnz(C):
            p = np.minimum(1.0, C * target_pop / (pop + 1e-9))
            return (pop * p).sum()

        lo, hi = 0.0, 1e6
        target_nnz = R_orig.nnz * target_density
        for _ in range(60):
            mid = (lo + hi) / 2
            if expected_nnz(mid) < target_nnz:
                lo = mid
            else:
                hi = mid
        C_opt = (lo + hi) / 2
        keep_probs = np.minimum(1.0, C_opt * target_pop / (pop + 1e-9))

    # 아이템별 Bernoulli 서브샘플링
    rows, cols = [], []
    for item in range(n_items):
        col = R_csc.getcol(item)
        user_indices = col.nonzero()[0]
        if len(user_indices) == 0:
            continue
        keep = rng.random(len(user_indices)) < keep_probs[item]
        kept_users = user_indices[keep]
        rows.extend(kept_users.tolist())
        cols.extend([item] * len(kept_users))

    data = np.ones(len(rows))
    R_sub = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    return R_sub, keep_probs


# ─── β 추정 ──────────────────────────────────────────────────────────────────

def quick_beta(R_mod, k_val):
    k_val = min(k_val, min(R_mod.shape) - 1, R_mod.nnz - 1)
    if k_val < 5:
        return None, None
    u, s, vt = svds(R_mod.astype(float), k=k_val)
    idx = np.argsort(s)[::-1]
    s, vt = s[idx], vt[idx, :]
    V_mod = torch.from_numpy(vt.T.copy()).float()
    S_mod = torch.from_numpy(s.copy()).float()
    item_pops = np.array(R_mod.sum(axis=0)).flatten()
    p_tilde = AspireEngine.compute_spp(V_mod, item_pops)
    beta, r2 = AspireEngine.estimate_beta(S_mod, p_tilde, verbose=False)
    return float(beta), float(r2)


# ─── 메인 실험 ───────────────────────────────────────────────────────────────

def run_beta_tracking_v2(
    dataset_name,
    target_energy=0.95,
    eta_targets=None,
    target_density=0.5,
    n_seeds=3,
):
    """
    Args:
        eta_targets:     테스트할 η_target 값 목록
        target_density:  원본 대비 유지 밀도 (0.5 = 50%)
        n_seeds:         평균 낼 시드 수 (오차 막대용)
    """
    if eta_targets is None:
        eta_targets = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]

    print(f"Running Experiment 4: Targeted Subsampling on {dataset_name}")
    print(f"  η_targets={eta_targets}, density={target_density}, seeds={n_seeds}")

    loader, R_orig, S_orig, V_orig, config = get_loader_and_svd(
        dataset_name, target_energy=target_energy
    )
    k_ref = V_orig.shape[1]
    dataset_label = config['dataset_name']

    betas_mean, betas_std = [], []
    r2s_mean = []

    for eta in eta_targets:
        seed_betas, seed_r2s = [], []
        for seed in range(n_seeds):
            R_sub, keep_probs = targeted_subsample(
                R_orig, eta, target_density=target_density, seed=seed
            )
            beta, r2 = quick_beta(R_sub, k_ref)
            if beta is not None:
                seed_betas.append(beta)
                seed_r2s.append(r2)

        mu_b = float(np.mean(seed_betas))
        sd_b = float(np.std(seed_betas)) if len(seed_betas) > 1 else 0.0
        mu_r = float(np.mean(seed_r2s))
        betas_mean.append(mu_b)
        betas_std.append(sd_b)
        r2s_mean.append(mu_r)
        print(f"  η={eta:.1f}  β={mu_b:.4f} ± {sd_b:.4f}  R²={mu_r:.4f}")

    out_dir = ensure_dir(f"aspire_experiments/output/tracking_v2/{dataset_label}")

    # ── Figure 1: β vs η_target (메인 그림) ─────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(8, 5))

    eta_arr = np.array(eta_targets)
    beta_arr = np.array(betas_mean)
    std_arr = np.array(betas_std)

    ax1.errorbar(eta_arr, beta_arr, yerr=std_arr,
                 fmt='o-', color='royalblue', capsize=4,
                 label='β̂ (estimated)', linewidth=2, markersize=6)
    ax1.set_xlabel("η_target  (0 = MCAR,  ↑ = stronger MNAR)", fontsize=12)
    ax1.set_ylabel("Estimated β", color='royalblue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='royalblue')
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='MCAR baseline')

    ax2 = ax1.twinx()
    ax2.plot(eta_arr, r2s_mean, 's--', color='tomato',
             alpha=0.7, label='R² (fit quality)', linewidth=1.5, markersize=5)
    ax2.set_ylabel("R² (power-law fit)", color='tomato', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='tomato')
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    plt.title(
        f"β Tracks MNAR Intensity (Targeted Subsampling)\n"
        f"Dataset: {dataset_label}  |  density={target_density:.0%}",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "beta_vs_eta.png"), dpi=150)
    plt.close()

    # ── Figure 2: β vs η_target 선형성 확인 (scatter + OLS) ─────────────────
    from sklearn.linear_model import LinearRegression
    lm = LinearRegression()
    lm.fit(eta_arr.reshape(-1, 1), beta_arr)
    beta_pred = lm.predict(eta_arr.reshape(-1, 1))
    slope = lm.coef_[0]
    intercept = lm.intercept_
    r2_lin = lm.score(eta_arr.reshape(-1, 1), beta_arr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(eta_arr, beta_arr, color='royalblue', s=60, zorder=3, label='β̂')
    ax.plot(eta_arr, beta_pred, 'r--', linewidth=1.5,
            label=f'Linear fit  (slope={slope:.3f}, R²={r2_lin:.3f})')
    ax.set_xlabel("η_target", fontsize=12)
    ax.set_ylabel("Estimated β", fontsize=12)
    ax.set_title(f"Linearity: β̂ ∝ η_target\n{dataset_label}", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "beta_linearity.png"), dpi=150)
    plt.close()

    # ── 결과 저장 ────────────────────────────────────────────────────────────
    result = {
        "dataset": dataset_label,
        "eta_targets": eta_targets,
        "target_density": target_density,
        "n_seeds": n_seeds,
        "betas_mean": betas_mean,
        "betas_std": betas_std,
        "r2s_mean": r2s_mean,
        "linearity": {
            "slope": float(slope),
            "intercept": float(intercept),
            "r2_linear": float(r2_lin),
        }
    }
    with open(os.path.join(out_dir, "result.json"), 'w') as f:
        json.dump(result, f, indent=4)

    print(f"  Linearity: slope={slope:.4f}, R²_linear={r2_lin:.4f}")
    print(f"  Saved to {out_dir}")
    return result


# ─── 멀티 데이터셋 요약 그림 ─────────────────────────────────────────────────

def plot_summary(results_list, out_dir):
    """여러 데이터셋 결과를 한 그림에"""
    ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(9, 5))

    colors = ['royalblue', 'seagreen', 'darkorange', 'purple']
    for i, res in enumerate(results_list):
        eta_arr = np.array(res['eta_targets'])
        beta_arr = np.array(res['betas_mean'])
        std_arr = np.array(res['betas_std'])
        label = res['dataset']
        ax.errorbar(eta_arr, beta_arr, yerr=std_arr,
                    fmt='o-', color=colors[i % len(colors)],
                    capsize=3, label=label, linewidth=2, markersize=5)

    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.4)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
    ax.set_xlabel("η_target  (0 = MCAR,  ↑ = stronger MNAR)", fontsize=12)
    ax.set_ylabel("Estimated β", fontsize=12)
    ax.set_title("β Consistently Tracks MNAR Intensity Across Datasets", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "beta_summary.png"), dpi=150)
    plt.close()
    print(f"  Summary figure saved to {out_dir}")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, nargs="+", default=["ml1m"],
                        help="하나 이상의 데이터셋 이름 (공백 구분)")
    parser.add_argument("--energy",  type=float, default=0.95)
    parser.add_argument("--eta",     type=float, nargs="+",
                        default=[0.0, 0.3, 0.6, 0.9, 1.2, 1.5],
                        help="테스트할 η_target 값 목록")
    parser.add_argument("--density", type=float, default=0.5,
                        help="원본 대비 유지 밀도 (0~1)")
    parser.add_argument("--seeds",   type=int, default=3,
                        help="오차 막대용 시드 수")
    args = parser.parse_args()

    all_results = []
    for ds in args.dataset:
        res = run_beta_tracking_v2(
            ds,
            target_energy=args.energy,
            eta_targets=args.eta,
            target_density=args.density,
            n_seeds=args.seeds,
        )
        all_results.append(res)

    if len(all_results) > 1:
        plot_summary(all_results, "aspire_experiments/output/tracking_v2/summary")
