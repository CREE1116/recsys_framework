# Usage: uv run python aspire_experiments/exp8_spectral_bias.py --dataset ml1m
#
# §Theory: SPP 가정 직접 검증
#
# "고유벡터 방향 k에서 인기 아이템의 하중이 집중된다"는 ASPIRE의 핵심 가정을
# 데이터에서 직접 시각화한다.
#
# 출력:
#   1. p̃_k vs σ_k 멱법칙 플롯 (방향별 오염 강도)
#   2. Top-K 방향에서 인기 / 롱테일 아이템의 V_{ki}² 분포 비교
#   3. 방향별 V_{ki}² 히트맵 (인기 아이템 그룹 vs. 롱테일 그룹)
#   4. 누적 인기 농도 커브 (각 방향이 얼마나 인기에 편향됐는가)

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr

sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import get_loader_and_svd, ensure_dir
from src.models.csar.ASPIRELayer import AspireEngine


# ─── 유틸 ──────────────────────────────────────────────────────────────────

def gini_coefficient(values: np.ndarray) -> float:
    """Gini 계수 — 불균등도 측정 (0=균등, 1=극단적 불균등)."""
    v = np.sort(np.abs(values))
    n = len(v)
    if n == 0 or v.sum() < 1e-12:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * (index * v).sum()) / (n * v.sum()) - (n + 1) / n)


# ─── 메인 실험 ─────────────────────────────────────────────────────────────

def run_spectral_bias(dataset_name: str, target_energy: float = 0.95,
                      n_top_dirs: int = 20, long_tail_pct: float = 0.8):
    """
    Parameters
    ----------
    n_top_dirs      : 상세 분석할 상위 특이 방향 수
    long_tail_pct   : 하위 이 비율을 롱테일로 정의 (0.8 = 하위 80%)
    """
    print(f"\n{'='*60}")
    print(f"  Exp 8: Spectral Bias Analysis  |  {dataset_name}")
    print(f"{'='*60}")

    loader, R, S, V, config = get_loader_and_svd(dataset_name,
                                                   target_energy=target_energy)
    dataset_label = config["dataset_name"]
    out_dir = ensure_dir(f"aspire_experiments/output/spectral_bias/{dataset_label}")

    S_np = S.cpu().numpy()          # (k,)  내림차순
    V_np = V.cpu().numpy()          # (n_items, k)
    k    = S_np.shape[0]
    n    = V_np.shape[0]

    # ── 인기도 정의 ──────────────────────────────────────────────────────
    item_pops = np.array(R.sum(axis=0)).flatten().astype(float)  # (n,)
    p_i       = item_pops / (item_pops.max() + 1e-9)

    # 인기 / 롱테일 아이템 분할
    pop_rank       = np.argsort(-item_pops)           # 인기 내림차순
    n_head         = int(n * (1 - long_tail_pct))
    head_items     = set(pop_rank[:n_head])
    tail_items     = set(pop_rank[n_head:])

    # ── SPP 계산 ──────────────────────────────────────────────────────────
    p_tilde = AspireEngine.compute_spp(V_np, item_pops)   # (k,)

    # ── β LAD 추정 (Robust Default) ──────────────────────────────────────
    from src.models.csar import beta_estimators
    beta_ref, r2_ref = beta_estimators.beta_lad(S_np, p_tilde)
    print(f"[Ref] β(LAD)={beta_ref:.4f}  R²={r2_ref:.4f}")

    # ─────────────────────────────────────────────────────────────────────
    # Figure 1: p̃_k vs σ_k 멱법칙 플롯
    # ─────────────────────────────────────────────────────────────────────
    mask = (S_np > 1e-9) & (p_tilde > 1e-9)
    log_s  = np.log(S_np[mask])
    log_pt = np.log(p_tilde[mask])
    slope_ref = 2.0 * beta_ref
    intercept_ref = np.mean(log_pt) - slope_ref * np.mean(log_s)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(log_s, log_pt, c=np.arange(mask.sum()),
                         cmap="RdYlGn_r", s=15, alpha=0.7,
                         label="Spectral direction k")
    x_line = np.linspace(log_s.min(), log_s.max(), 200)
    ax.plot(x_line, slope_ref * x_line + intercept_ref, "r--",
            linewidth=2.0, label=f"OLS fit: slope={slope_ref:.3f}, β={beta_ref:.3f}")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Direction index k (0=largest σ)", fontsize=9)
    ax.set_xlabel("log σ_k  (singular value, log scale)", fontsize=12)
    ax.set_ylabel("log p̃_k  (spectral propensity, log scale)", fontsize=12)
    ax.set_title(f"SPP Power-law: log p̃_k ∝ 2β · log σ_k\n"
                 f"{dataset_label}  β={beta_ref:.3f}  R²={r2_ref:.3f}", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "spp_powerlaw.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[Plot] SPP 멱법칙 플롯 저장")

    # ─────────────────────────────────────────────────────────────────────
    # Figure 2: 방향별 Head vs. Tail V_{ki}² 평균 비교
    # ─────────────────────────────────────────────────────────────────────
    head_idx = np.array(sorted(head_items))
    tail_idx = np.array(sorted(tail_items))

    head_loading = (V_np[head_idx] ** 2).mean(axis=0)   # (k,)
    tail_loading = (V_np[tail_idx] ** 2).mean(axis=0)   # (k,)
    ratio = head_loading / (tail_loading + 1e-12)        # >1 means head-biased

    dirs = np.arange(k)
    top = min(n_top_dirs, k)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 2-a) 절대 하중
    axes[0].plot(dirs[:top], head_loading[:top], color="#E91E63",
                 linewidth=2.0, label=f"Head items (top {int((1-long_tail_pct)*100)}%)")
    axes[0].plot(dirs[:top], tail_loading[:top], color="#4CAF50",
                 linewidth=2.0, label=f"Tail items (bottom {int(long_tail_pct*100)}%)")
    axes[0].set_ylabel("Mean V²_{ki}", fontsize=11)
    axes[0].set_title(f"Spectral Loading: Head vs. Tail Items  ({dataset_label})", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 2-b) 하중 비율 (Head/Tail)
    axes[1].bar(dirs[:top], ratio[:top], color=["#E91E63" if r > 1 else "#4CAF50"
                                                  for r in ratio[:top]])
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.5,
                    label="ratio = 1 (no bias)")
    axes[1].set_ylabel("Head/Tail Loading Ratio", fontsize=11)
    axes[1].set_title("Head/Tail Loading Ratio per Direction  (>1 = head-biased)", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # 2-c) p̃_k (color bar는 σ_k 크기)
    sc = axes[2].scatter(dirs[:top], p_tilde[:top], c=S_np[:top],
                         cmap="Blues", s=60, zorder=5)
    axes[2].plot(dirs[:top], p_tilde[:top], color="#2196F3",
                 linewidth=1.5, alpha=0.6)
    fig.colorbar(sc, ax=axes[2], label="σ_k (singular value)")
    axes[2].set_xlabel("Direction k  (sorted by σ_k descending)", fontsize=11)
    axes[2].set_ylabel("p̃_k (SPP value)", fontsize=11)
    axes[2].set_title("Spectral Propensity p̃_k per Direction", fontsize=12)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "head_tail_loading.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("[Plot] Head vs. Tail 하중 비교 저장")

    # ─────────────────────────────────────────────────────────────────────
    # Figure 3: 방향별 인기도-하중 Spearman 상관 (대형 히트맵 대신)
    # ─────────────────────────────────────────────────────────────────────
    spearman_per_dir = []
    for ki in range(min(top, k)):
        # 각 방향 ki에서 아이템 하중 V_{i,ki}² vs 아이템 인기도
        loading_ki = V_np[:, ki] ** 2
        rho, pval = spearmanr(loading_ki, p_i)
        spearman_per_dir.append(float(rho))

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#E91E63" if r > 0 else "#4CAF50" for r in spearman_per_dir]
    ax.bar(range(top), spearman_per_dir, color=colors, alpha=0.8)
    ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Direction k  (sorted by σ_k descending)", fontsize=12)
    ax.set_ylabel("Spearman ρ  (loading vs. popularity)", fontsize=12)
    ax.set_title(
        f"Per-direction Spearman Correlation: V²_k ↔ Item Popularity\n"
        f"{dataset_label}  (positive = popularity-biased direction)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "spearman_per_dir.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("[Plot] Spearman 상관 저장")

    # ─────────────────────────────────────────────────────────────────────
    # Figure 4: 누적 인기 농도 커브
    # ─────────────────────────────────────────────────────────────────────
    # k번째 방향까지 누적했을 때 Head 아이템이 전체 V² 에너지 중 몇 %를 차지?
    cumul_head_frac = []
    for ki in range(k):
        total_sq  = (V_np[:, :ki+1] ** 2).sum()
        head_sq   = (V_np[head_idx, :ki+1] ** 2).sum()
        cumul_head_frac.append(head_sq / (total_sq + 1e-12))

    expected_head_frac = len(head_idx) / n   # baseline (균등 분포)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(k), cumul_head_frac, color="#E91E63", linewidth=2.0,
            label=f"Head item energy fraction (top {int((1-long_tail_pct)*100)}%)")
    ax.axhline(expected_head_frac, color="gray", linestyle="--", linewidth=1.5,
               label=f"Expected (uniform) = {expected_head_frac:.3f}")
    ax.fill_between(range(k), expected_head_frac, cumul_head_frac,
                    where=[v > expected_head_frac for v in cumul_head_frac],
                    alpha=0.2, color="#E91E63", label="Bias region")
    ax.set_xlabel("Number of singular directions included (k)", fontsize=12)
    ax.set_ylabel("Head item energy fraction", fontsize=12)
    ax.set_title(
        f"Cumulative Head-Item Energy Fraction vs. Singular Directions\n"
        f"{dataset_label}  (above baseline = popularity bias in top directions)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cumulative_head_energy.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("[Plot] 누적 인기 에너지 커브 저장")

    # ─── 결과 저장 ────────────────────────────────────────────────────────
    result = {
        "dataset":            dataset_label,
        "k":                  int(k),
        "n_items":            int(n),
        "n_head":             int(len(head_idx)),
        "n_tail":             int(len(tail_idx)),
        "long_tail_pct":      long_tail_pct,
        "beta_ref":           float(beta_ref),
        "r2_ref":             float(r2_ref),
        "gini_item_pop":      gini_coefficient(item_pops),
        "mean_spearman_rho":  float(np.mean(spearman_per_dir)),
        "head_energy_at_k1":  float(cumul_head_frac[0]) if cumul_head_frac else 0.0,
        "head_energy_excess": float(np.mean(
            [v - expected_head_frac for v in cumul_head_frac
             if v > expected_head_frac]
        )) if cumul_head_frac else 0.0,
    }
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    # [NEW] Save detailed per-direction data
    pd.DataFrame({
        "direction_k": np.arange(k),
        "singular_value": S_np,
        "p_tilde": p_tilde,
        "head_loading_avg": head_loading,
        "tail_loading_avg": tail_loading,
        "bias_ratio": ratio,
        "spearman_rho": spearman_per_dir + [None]*(k-len(spearman_per_dir)), # fill with None if top limited
        "cumul_head_energy_frac": cumul_head_frac
    }).to_csv(os.path.join(out_dir, "spectral_bias_details.csv"), index=False)

    print(f"\n[Summary]")
    print(f"  β(OLS)={beta_ref:.4f}  R²={r2_ref:.4f}")
    print(f"  Gini(item_pop)={result['gini_item_pop']:.4f}")
    print(f"  Mean Spearman ρ (top {top} dirs) = {result['mean_spearman_rho']:.4f}")
    print(f"  Head energy excess = {result['head_energy_excess']:.4f}")
    print(f"\n[Done] 결과 저장: {out_dir}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASPIRE Spectral Bias Analysis")
    parser.add_argument("--dataset",       type=str,   nargs="+", default=["ml100k"])
    parser.add_argument("--energy",        type=float, default=0.95)
    parser.add_argument("--n_top_dirs",    type=int,   default=30,
                        help="상세 분석할 상위 방향 수 (기본 30)")
    parser.add_argument("--long_tail_pct", type=float, default=0.8,
                        help="롱테일 정의 비율 (기본 0.8 = 하위 80%%)")
    args = parser.parse_args()
    for ds in args.dataset:
        run_spectral_bias(ds, args.energy, args.n_top_dirs, args.long_tail_pct)
