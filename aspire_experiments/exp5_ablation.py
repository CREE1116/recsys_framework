# Usage: uv run python aspire_experiments/exp5_ablation.py --dataset ml1m --energy 0.95
#
# §6.3 실험 C: β 추정 방법 Ablation (v3)
# 각 방법별로 최적의 alpha를 개별적으로 HPO하여 공정한 최대 성능 비교 진행
#
# 측정 지표: NDCG@10 (전체, Head, Mid, Tail), Item Coverage@10

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.linear_model import HuberRegressor, LinearRegression

sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import get_loader_and_svd, ensure_dir
from src.models.csar.ASPIRELayer import AspireEngine


# ── 유틸리티 및 계산 함수 ──────────────────────────────────────────────────────

def dcg_at_k(relevance, k):
    r = np.asarray(relevance[:k], dtype=float)
    if r.size == 0: return 0.0
    discounts = np.log2(np.arange(2, r.size + 2))
    return (r / discounts).sum()


def ndcg_at_k(scores, labels, k=10):
    sorted_idx = np.argsort(scores)[::-1]
    rel = labels[sorted_idx]
    ideal = np.sort(labels)[::-1]
    dcg = dcg_at_k(rel, k)
    idcg = dcg_at_k(ideal, k)
    return dcg / (idcg + 1e-9) if idcg > 0 else 0.0


def make_pop_groups(R_train, head_ratio=0.20, mid_ratio=0.40):
    pop = np.array(R_train.sum(axis=0)).flatten()
    sorted_idx = np.argsort(pop)[::-1]
    n = len(pop)
    h_end = int(n * head_ratio)
    m_end = int(n * (head_ratio + mid_ratio))
    return {'head': sorted_idx[:h_end], 'mid': sorted_idx[h_end:m_end], 'tail': sorted_idx[m_end:]}


def split_data(R, val_ratio=0.1, test_ratio=0.1, seed=42):
    from scipy.sparse import csr_matrix
    rng = np.random.default_rng(seed)
    rows_tr, cols_tr, rows_va, cols_va, rows_te, cols_te = [], [], [], [], [], []
    R_csr = R.tocsr()
    for u in range(R_csr.shape[0]):
        items = R_csr.getrow(u).nonzero()[1]
        if len(items) < 3:
            rows_tr.extend([u] * len(items)); cols_tr.extend(items.tolist()); continue
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio)); n_test = max(1, int(len(items) * test_ratio))
        te, va, tr = items[:n_test], items[n_test:n_test + n_val], items[n_test + n_val:]
        rows_tr.extend([u] * len(tr)); cols_tr.extend(tr.tolist())
        rows_va.extend([u] * len(va)); cols_va.extend(va.tolist())
        rows_te.extend([u] * len(te)); cols_te.extend(te.tolist())
    def to_csr(r, c): return csr_matrix((np.ones(len(r)), (r, c)), shape=R.shape)
    return to_csr(rows_tr, cols_tr), to_csr(rows_va, cols_va), to_csr(rows_te, cols_te)


# ── β 추정 방법들 ────────────────────────────────────────────────────────────

def estimate_beta_ols(S, p_tilde):
    x = np.log(S.numpy() + 1e-9).reshape(-1, 1)
    y = np.log(p_tilde + 1e-9)
    lm = LinearRegression().fit(x, y)
    beta = lm.coef_[0] / 2.0
    return float(beta), float(lm.score(x, y))


def estimate_beta_slope(R_train):
    pop = np.array(R_train.sum(axis=0)).flatten()
    ranks = np.arange(1, len(pop) + 1, dtype=float)
    sorted_pop = np.sort(pop)[::-1]
    mask = sorted_pop > 0
    lm_n = LinearRegression().fit(np.log(ranks[mask]).reshape(-1, 1), np.log(sorted_pop[mask]))
    slope_n = float(lm_n.coef_[0])

    from scipy.sparse.linalg import svds
    _, s, _ = svds(R_train.astype(float), k=min(100, min(R_train.shape) - 1))
    s = np.sort(s)[::-1]
    s_ranks = np.arange(1, len(s) + 1, dtype=float)
    lm_s = LinearRegression().fit(np.log(s_ranks[s > 0]).reshape(-1, 1), np.log(s[s > 0]))
    slope_sigma = float(lm_s.coef_[0])

    beta = (slope_n / (slope_sigma + 1e-9)) / 2.0
    return float(beta)


# ── 필터 적용 및 평가 ─────────────────────────────────────────────────────────

def evaluate_filter(R_train, R_test, beta, alpha, item_pop_groups, k=10):
    from scipy.sparse.linalg import svds
    n_users, n_items = R_train.shape
    k_svd = min(200, min(R_train.shape) - 1)
    u, s, vt = svds(R_train.astype(float), k=k_svd)
    idx = np.argsort(s)[::-1]; s, vt = s[idx], vt[idx, :]; V = vt.T

    s_tensor = torch.from_numpy(s.copy()).float()
    h = AspireEngine.apply_filter(s_tensor, alpha=alpha, beta=beta)
    h_np = h.numpy()

    X_dense = np.array(R_train.todense())
    XV = X_dense @ V
    XVh = XV * h_np
    R_pred = XVh @ V.T

    R_train_dense = np.array(R_train.todense())
    R_pred = R_pred * (1 - R_train_dense)
    R_test_dense = np.array(R_test.todense())

    ndcg_all, ndcg_head, ndcg_mid, ndcg_tail = [], [], [], []
    recommended_items = set()

    for u in range(n_users):
        if R_test_dense[u].sum() == 0: continue
        scores = R_pred[u]
        labels = R_test_dense[u]
        ndcg_all.append(ndcg_at_k(scores, labels, k))
        
        top_k_idx = np.argsort(scores)[::-1][:k]
        recommended_items.update(top_k_idx.tolist())

        for group_name, group_idx in item_pop_groups.items():
            s_g = scores[group_idx]; l_g = labels[group_idx]
            val = ndcg_at_k(s_g, l_g, k)
            if group_name == 'head': ndcg_head.append(val)
            elif group_name == 'mid': ndcg_mid.append(val)
            elif group_name == 'tail': ndcg_tail.append(val)

    coverage = len(recommended_items) / n_items
    return {
        'ndcg_all': float(np.mean(ndcg_all)),
        'ndcg_head': float(np.mean(ndcg_head)) if ndcg_head else 0.0,
        'ndcg_mid': float(np.mean(ndcg_mid)) if ndcg_mid else 0.0,
        'ndcg_tail': float(np.mean(ndcg_tail)) if ndcg_tail else 0.0,
        'coverage': float(coverage)
    }


def find_alpha(R_train, R_val, beta, alpha_grid=None, k_svd=200):
    from scipy.sparse.linalg import svds
    # 알파 범위를 0.1(10^-1)에서 1,000,000(10^6)까지 대폭 확대하며 로그 스케일링
    if alpha_grid is None: alpha_grid = np.logspace(-1, 6, 100) 
    k_svd = min(k_svd, min(R_train.shape) - 1)
    u, s, vt = svds(R_train.astype(float), k=k_svd)
    idx = np.argsort(s)[::-1]; s, vt = s[idx], vt[idx, :]; V = vt.T
    X_dense = np.array(R_train.todense()); XV = X_dense @ V
    R_val_dense = np.array(R_val.todense()); R_train_dense = np.array(R_train.todense())

    best_alpha, best_ndcg = 1.0, -1.0
    for alpha in alpha_grid:
        h = (s ** (2 - 2 * beta)) / (s ** (2 - 2 * beta) + alpha)
        R_pred = (XV * h) @ V.T * (1 - R_train_dense)
        ndcgs = [ndcg_at_k(R_pred[u], R_val_dense[u], k=10) for u in range(R_val_dense.shape[0]) if R_val_dense[u].sum() > 0]
        score = float(np.mean(ndcgs)) if ndcgs else 0.0
        if score > best_ndcg: best_ndcg, best_alpha = score, float(alpha)
    return best_alpha


def beta_hpo_joint_sweep(R_train, R_val, beta_grid=None, alpha_grid=None):
    if beta_grid is None: beta_grid = np.linspace(0.0, 1.5, 31) 
    best_beta, best_alpha, best_ndcg = 0.5, 1.0, -1.0
    for beta in beta_grid:
        alpha = find_alpha(R_train, R_val, beta, alpha_grid=alpha_grid)
        res = evaluate_filter(R_train, R_val, beta, alpha, {'h':[]}, k=10)
        if res['ndcg_all'] > best_ndcg:
            best_ndcg, best_beta, best_alpha = res['ndcg_all'], float(beta), float(alpha)
    return best_beta, best_alpha


# ── 메인 실험 로직 ────────────────────────────────────────────────────────────

def run_ablation(dataset_name, target_energy=0.95, k_svd=200):
    print(f"\n{'='*60}\nRunning Experiment 5 (Per-Method HPO) on {dataset_name}\n{'='*60}")
    loader, R_full, _, _, config = get_loader_and_svd(dataset_name, target_energy=target_energy)
    R_train, R_val, R_test = split_data(R_full)
    pop_groups = make_pop_groups(R_train)

    from scipy.sparse.linalg import svds
    k_svd_est = min(k_svd, min(R_train.shape) - 1)
    _, s_svd, vt_svd = svds(R_train.astype(float), k=k_svd_est)
    idx = np.argsort(s_svd)[::-1]; s_svd, vt_svd = s_svd[idx], vt_svd[idx, :]
    V_tr = torch.from_numpy(vt_svd.T.copy()).float(); S_tr = torch.from_numpy(s_svd.copy()).float()
    item_pops = np.array(R_train.sum(axis=0)).flatten()
    p_tilde = AspireEngine.compute_spp(V_tr, item_pops)

    # 1. 7가지 Beta 추정
    beta_huber, _ = AspireEngine.estimate_beta(S_tr, p_tilde, verbose=False)
    beta_ols, _ = estimate_beta_ols(S_tr, p_tilde)
    beta_slope = estimate_beta_slope(R_train)
    beta_fixed = 0.5
    beta_direct = 1.0
    beta_damped = float(beta_huber * 0.5)
    
    # 상한선 (Joint Sweep)
    beta_hpo, alpha_hpo = beta_hpo_joint_sweep(R_train, R_val)

    betas = {
        "(1) SPP+Huber": beta_huber,
        "(2) SPP+OLS": beta_ols,
        "(3) Slope ratio": beta_slope,
        "(4) β=0.5 fixed": beta_fixed,
        "(6) SPP direct": beta_direct,
        "(7) 0.5 damping": beta_damped,
        "(5) β=HPO": beta_hpo
    }

    print("\n  Optimizing alpha for EACH beta (Per-method HPO)...")
    results_table = []
    print(f"\n  {'Method':<20} | {'Beta':<5} | {'Alpha':<7} | {'NDCG':<7} | {'Tail':<7} | {'Cov':<7}")
    print(f"  {'-'*20}-|-{'-'*5}-|-{'-'*7}-|-{'-'*7}-|-{'-'*7}-|-{'-'*7}")
    
    for name, beta in betas.items():
        if name == "(5) β=HPO":
            alpha = alpha_hpo
        else:
            alpha = find_alpha(R_train, R_val, beta)
            
        m = evaluate_filter(R_train, R_test, beta, alpha, pop_groups, k=10)
        results_table.append({"method": name, "beta": float(beta), "alpha": float(alpha), **m})
        print(f"  {name:<20} | {beta:.3f} | {alpha:.2f} | {m['ndcg_all']:.4f} | {m['ndcg_tail']:.4f} | {m['coverage']:.3f}")

    dataset_label = config['dataset_name']
    out_dir = ensure_dir(f"aspire_experiments/output/ablation/{dataset_label}")
    with open(os.path.join(out_dir, "result_per_method.json"), 'w') as f:
        json.dump({"dataset": dataset_label, "methods": results_table}, f, indent=4)

    # Visualization: NDCG comparison
    m_names = [r['method'] for r in results_table]
    plt.figure(figsize=(12, 6))
    x = np.arange(len(m_names))
    plt.bar(x, [r['ndcg_all'] for r in results_table], color='skyblue', label='Overall NDCG')
    plt.xticks(x, m_names, rotation=15, ha='right')
    plt.title(f"Ablation (Per-method HPO): Overall NDCG@10 — {dataset_label}")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "ablation_per_method_ndcg.png"), dpi=150); plt.close()

    return results_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, nargs="+", default=["ml100k"])
    parser.add_argument("--energy", type=float, default=0.95)
    args = parser.parse_args()
    for ds in args.dataset: run_ablation(ds, args.energy)
