import os
import sys
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import argparse
import numpy as np

sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import get_loader_and_svd, ensure_dir
from src.models.csar.ASPIRELayer import AspireEngine


def estimate_eta(R):
    """
    인기도 멱법칙 지수 η 추정.
    n_i ~ rank_i^{-η}  →  log(n_i) ~ -η · log(rank_i)
    slope의 절댓값이 η.
    """
    pop = np.array(R.sum(axis=0)).flatten()
    sorted_pop = np.sort(pop)[::-1]
    mask = sorted_pop > 0
    log_rank = np.log(np.arange(1, mask.sum() + 1, dtype=float))
    log_pop  = np.log(sorted_pop[mask])

    lm = LinearRegression()
    lm.fit(log_rank.reshape(-1, 1), log_pop)
    eta   = float(-lm.coef_[0])   # slope은 음수 → 절댓값
    r2    = float(lm.score(log_rank.reshape(-1, 1), log_pop))
    return eta, r2, log_rank, log_pop, lm


def estimate_alpha(S):
    """
    특이값 멱법칙 지수 α 추정.
    σ_k ~ k^{-α}  →  log(σ_k) ~ -α · log(k)
    slope의 절댓값이 α.
    """
    s_np = S.cpu().numpy()
    mask = s_np > 0
    log_k     = np.log(np.arange(1, mask.sum() + 1, dtype=float))
    log_sigma = np.log(s_np[mask])

    lm = LinearRegression()
    lm.fit(log_k.reshape(-1, 1), log_sigma)
    alpha = float(-lm.coef_[0])
    r2    = float(lm.score(log_k.reshape(-1, 1), log_sigma))
    return alpha, r2, log_k, log_sigma, lm


def run_corollary2(dataset_names, target_energy=0.95):
    print("=" * 60)
    print("Experiment 6: Corollary 2 Verification  β = η / (2α)")
    print("=" * 60)

    records = []

    for dataset_name in dataset_names:
        print(f"\n  Dataset: {dataset_name}")
        loader, R, S, V, config = get_loader_and_svd(
            dataset_name, target_energy=target_energy
        )
        label = config['dataset_name']

        # ── η 추정 ────────────────────────────────────────────────────────
        eta, r2_eta, log_rank, log_pop, lm_eta = estimate_eta(R)

        # ── α 추정 ────────────────────────────────────────────────────────
        alpha, r2_alpha, log_k, log_sigma, lm_alpha = estimate_alpha(S)

        # ── β_theory ──────────────────────────────────────────────────────
        beta_theory = eta / (2 * alpha)

        # ── β_measured (SPP + Huber, exp2와 동일) ─────────────────────────
        item_pops  = np.array(R.sum(axis=0)).flatten()
        p_tilde    = AspireEngine.compute_spp(V, item_pops)
        beta_meas, r2_spp = AspireEngine.estimate_beta(S, p_tilde, verbose=False)

        # ── 오차 ──────────────────────────────────────────────────────────
        abs_err = abs(beta_theory - beta_meas)
        rel_err = abs_err / (beta_meas + 1e-9)

        print(f"    η          = {eta:.4f}  (R²={r2_eta:.4f})")
        print(f"    α          = {alpha:.4f}  (R²={r2_alpha:.4f})")
        print(f"    β_theory   = η/(2α) = {beta_theory:.4f}")
        print(f"    β_measured = SPP+Huber = {float(beta_meas):.4f}  (R²={r2_spp:.4f})")
        print(f"    |Δβ|       = {abs_err:.4f}  ({rel_err:.1%})")

        records.append({
            "dataset":      label,
            "eta":          eta,
            "r2_eta":       r2_eta,
            "alpha":        alpha,
            "r2_alpha":     r2_alpha,
            "beta_theory":  float(beta_theory),
            "beta_measured":float(beta_meas),
            "r2_spp":       float(r2_spp),
            "abs_error":    float(abs_err),
            "rel_error":    float(rel_err),
            # 멱법칙 피팅용 (Figure 2, 3)
            "_log_rank":    log_rank.tolist(),
            "_log_pop":     log_pop.tolist(),
            "_log_k":       log_k.tolist(),
            "_log_sigma":   log_sigma.tolist(),
            "_lm_eta_coef": float(lm_eta.coef_[0]),
            "_lm_eta_int":  float(lm_eta.intercept_),
            "_lm_alpha_coef": float(lm_alpha.coef_[0]),
            "_lm_alpha_int":  float(lm_alpha.intercept_),
        })

    out_dir = ensure_dir("aspire_experiments/output/corollary2")

    # ── Figure 1: β_theory vs β_measured scatter ──────────────────────────
    # 대각선(y=x)에 가까울수록 Corollary 2 성립
    labels    = [r['dataset']       for r in records]
    bt_vals   = [r['beta_theory']   for r in records]
    bm_vals   = [r['beta_measured'] for r in records]

    fig, ax = plt.subplots(figsize=(5, 5))
    colors = ['royalblue', 'seagreen', 'darkorange']
    for i, (lbl, bt, bm) in enumerate(zip(labels, bt_vals, bm_vals)):
        ax.scatter(bt, bm, color=colors[i % len(colors)], s=120, zorder=3, label=lbl)
        ax.annotate(f" {lbl}", (bt, bm), fontsize=9)

    all_vals = bt_vals + bm_vals
    lo, hi = min(all_vals) * 0.85, max(all_vals) * 1.1
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, alpha=0.5, label='y = x (perfect)')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("β_theory = η / (2α)", fontsize=12)
    ax.set_ylabel("β_measured  (SPP + Huber)", fontsize=12)
    ax.set_title("Corollary 2 Verification\nβ_theory ≈ β_measured", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "corollary2_scatter.png"), dpi=150)
    plt.close()

    # ── Figure 2: 인기도 멱법칙 피팅 (η 시각화, 데이터셋별) ──────────────
    fig, axes = plt.subplots(1, len(records), figsize=(5 * len(records), 4), sharey=False)
    if len(records) == 1:
        axes = [axes]
    for ax, rec in zip(axes, records):
        lr = np.array(rec['_log_rank'])
        lp = np.array(rec['_log_pop'])
        fitted = rec['_lm_eta_coef'] * lr + rec['_lm_eta_int']
        ax.scatter(lr, lp, s=4, alpha=0.3, color='steelblue')
        ax.plot(lr, fitted, 'r-', linewidth=1.5,
                label=f"η={rec['eta']:.3f}  R²={rec['r2_eta']:.3f}")
        ax.set_xlabel("log(rank)", fontsize=10)
        ax.set_ylabel("log(n_i)", fontsize=10)
        ax.set_title(rec['dataset'], fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Popularity Power-law  (η estimation)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eta_powerlaw.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 3: 특이값 멱법칙 피팅 (α 시각화, 데이터셋별) ─────────────
    fig, axes = plt.subplots(1, len(records), figsize=(5 * len(records), 4), sharey=False)
    if len(records) == 1:
        axes = [axes]
    for ax, rec in zip(axes, records):
        lk = np.array(rec['_log_k'])
        ls = np.array(rec['_log_sigma'])
        fitted = rec['_lm_alpha_coef'] * lk + rec['_lm_alpha_int']
        ax.scatter(lk, ls, s=4, alpha=0.3, color='seagreen')
        ax.plot(lk, fitted, 'r-', linewidth=1.5,
                label=f"α={rec['alpha']:.3f}  R²={rec['r2_alpha']:.3f}")
        ax.set_xlabel("log(k)", fontsize=10)
        ax.set_ylabel("log(σ_k)", fontsize=10)
        ax.set_title(rec['dataset'], fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Singular Value Power-law  (α estimation)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "alpha_powerlaw.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 4: 데이터셋별 η, α, β_theory, β_measured 막대 비교 ─────────
    x = np.arange(len(records))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 1.5*width, [r['eta']           for r in records], width, label='η',           color='#4472C4')
    ax.bar(x - 0.5*width, [r['alpha']         for r in records], width, label='α',           color='#70AD47')
    ax.bar(x + 0.5*width, [r['beta_theory']   for r in records], width, label='β_theory',    color='#ED7D31')
    ax.bar(x + 1.5*width, [r['beta_measured'] for r in records], width, label='β_measured',  color='#FFC000',
           edgecolor='gray', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([r['dataset'] for r in records], fontsize=11)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("η,  α,  β_theory = η/(2α),  β_measured — per Dataset", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "components_bar.png"), dpi=150)
    plt.close()

    # ── 결과 저장 (raw 피팅 데이터 제외) ──────────────────────────────────
    clean_records = [
        {k: v for k, v in r.items() if not k.startswith('_')}
        for r in records
    ]
    result = {"experiments": clean_records}
    with open(os.path.join(out_dir, "result.json"), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"\n  Saved to {out_dir}")

    # ── 콘솔 요약 테이블 ──────────────────────────────────────────────────
    print("\n  Summary:")
    print(f"  {'Dataset':<12} {'η':>6} {'α':>6} {'β_theory':>10} {'β_measured':>12} {'|Δβ|':>8}")
    print("  " + "-" * 58)
    for r in clean_records:
        print(f"  {r['dataset']:<12} {r['eta']:>6.3f} {r['alpha']:>6.3f} "
              f"{r['beta_theory']:>10.4f} {r['beta_measured']:>12.4f} {r['abs_error']:>8.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, nargs="+", default=["ml100k", "ml1m", "ml20m"])
    parser.add_argument("--energy",  type=float, default=0.95)
    args = parser.parse_args()

    run_corollary2(args.dataset, target_energy=args.energy)
