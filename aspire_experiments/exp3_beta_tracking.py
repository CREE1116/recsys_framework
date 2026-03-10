# Usage: uv run python aspire_experiments/exp3_beta_tracking.py --dataset ml1m --energy 0.95
import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.sparse import csr_matrix

# Framework root path
sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import get_loader_and_svd, ensure_dir
from src.models.csar.ASPIRELayer import AspireEngine

def inject_mcar_noise(R_orig, noise_ratio):
    """
    원본 행렬에 균일 랜덤 상호작용을 주입.
    noise_ratio: 주입할 상호작용 수 / 원본 상호작용 수
    
    noise_ratio=0.0 → 원본 (pure MNAR)
    noise_ratio=1.0 → 원본만큼의 MCAR 노이즈 추가 (50:50 혼합)
    noise_ratio=4.0 → 원본의 4배 MCAR 추가 (80% MCAR)
    """
    n_users, n_items = R_orig.shape
    n_orig = R_orig.nnz
    n_inject = int(n_orig * noise_ratio)

    if n_inject == 0:
        return R_orig.copy()

    rng = np.random.default_rng(42)

    # 랜덤 (user, item) 쌍 샘플링 — 중복 허용 (implicit 1로 처리)
    users = rng.integers(0, n_users, size=n_inject)
    items = rng.integers(0, n_items, size=n_inject)
    noise = csr_matrix(
        (np.ones(n_inject), (users, items)),
        shape=(n_users, n_items)
    )

    # 원본 + 노이즈, 값은 binary (>0 → 1)
    R_mixed = R_orig + noise
    R_mixed.data[:] = 1.0
    R_mixed.eliminate_zeros()
    return R_mixed


def quick_beta(R_mod, k_val):
    from scipy.sparse.linalg import svds
    # R_mod.nnz - 1로 k_val 제한
    k_val = min(k_val, min(R_mod.shape) - 1, R_mod.nnz - 1)
    if k_val < 2:
        return 0.0, 0.0
    
    u, s, vt = svds(R_mod.astype(float), k=k_val)
    idx = np.argsort(s)[::-1]
    s, vt = s[idx], vt[idx, :]
    V_mod = torch.from_numpy(vt.T.copy()).float()
    S_mod = torch.from_numpy(s.copy()).float()
    item_pops = np.array(R_mod.sum(axis=0)).flatten()
    p_tilde = AspireEngine.compute_spp(V_mod, item_pops)
    beta, r2 = AspireEngine.estimate_beta(S_mod, p_tilde, verbose=False)
    return beta, r2


def run_beta_tracking(dataset_name, target_energy=0.95,
                      noise_levels=None):
    """
    noise_levels: 원본 대비 MCAR 노이즈 주입 비율
      0.0 = 원본 (pure MNAR)
      0.5 = 원본의 50% 랜덤 추가
      1.0 = 원본의 100% 추가 (50:50 혼합)
      2.0 = 원본의 200% 추가 (33% MNAR + 67% MCAR)
      4.0 = 원본의 400% 추가 (20% MNAR + 80% MCAR)
    """
    if noise_levels is None:
        noise_levels = [float(x) for x in np.arange(0.0, 10.5, 0.5)]

    print(f"Running Experiment 3 (v2): MCAR Noise Injection on {dataset_name}...")
    loader, R_orig, S_orig, V_orig, config = get_loader_and_svd(
        dataset_name, target_energy=target_energy
    )
    k_ref = V_orig.shape[1]

    betas, r2s, mcar_fracs = [], [], []

    for nr in noise_levels:
        R_mixed = inject_mcar_noise(R_orig, nr)

        # MCAR 비율: 주입된 노이즈 / 전체
        mcar_frac = nr / (1.0 + nr)
        mcar_fracs.append(mcar_frac)

        beta, r2 = quick_beta(R_mixed, k_ref)
        betas.append(beta)
        r2s.append(r2)

        print(f"  noise_ratio={nr:.1f}  MCAR_frac={mcar_frac:.2%} "
              f" β={beta:.4f}  R²={r2:.4f}")

    out_dir = ensure_dir(
        f"aspire_experiments/output/tracking/{config['dataset_name']}"
    )

    # ── 그림 1: β vs MCAR 비율 ─────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(mcar_fracs, betas, 'o-', color='royalblue', label='β')
    ax1.set_xlabel("MCAR Fraction (injected / total interactions)")
    ax1.set_ylabel("Estimated β", color='royalblue')
    ax1.tick_params(axis='y', labelcolor='royalblue')

    ax2 = ax1.twinx()
    ax2.plot(mcar_fracs, r2s, 's--', color='tomato', alpha=0.7, label='R²')
    ax2.set_ylabel("R² (power-law fit)", color='tomato')
    ax2.tick_params(axis='y', labelcolor='tomato')
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title(f"β Decreases as Data Becomes More MCAR\n"
              f"Dataset: {config['dataset_name']}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "beta_mcar_injection.png"), dpi=150)
    plt.close()

    result = {
        "dataset": config['dataset_name'],
        "noise_levels": noise_levels,
        "mcar_fracs": [round(f, 4) for f in mcar_fracs],
        "betas": [float(b) for b in betas],
        "r2s":   [float(r) for r in r2s],
    }
    with open(os.path.join(out_dir, "result_v2.json"), 'w') as f:
        json.dump(result, f, indent=4)

    print(f"  Saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  type=str,   default="ml1m")
    parser.add_argument("--energy",   type=float, default=0.95)
    parser.add_argument("--noise",    type=float, nargs="+",
                        default=[float(x) for x in np.arange(0.0, 10.5, 0.5)])
    args = parser.parse_args()

    run_beta_tracking(args.dataset, args.energy, args.noise)
