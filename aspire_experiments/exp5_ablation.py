# Usage: uv run python aspire_experiments/exp5_ablation.py --dataset ml1m --energy 0.95
#
# §6.3 실험 C: β 추정 방법 Ablation (v6 - Bayesian HPO Integration)
# 프레임워크의 Bayesian HPO (Optuna) 로직을 통합하여 Alpha와 Beta를 세밀하게 피팅.
#
# 비교 방법:
#   (1) SPP + Huber (Proposed)
#   (2) SPP + OLS
#   (3) Slope ratio (ChebyASPIRE 방식)
#   (4) β = 0.5 fixed
#   (5) β = HPO (Joint Bayesian)
#   (6) SPP direct (β=1.0)
#   (7) 0.5 damping (β = β_huber * 0.5)

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression

sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import get_loader_and_svd, ensure_dir, AspireHPO
from src.models.csar.ASPIRELayer import AspireEngine
from src.evaluation import evaluate_metrics


# ── 프레임워크 호환 모델 래퍼 ──────────────────────────────────────────────────

class ASPIREAblationModel(nn.Module):
    def __init__(self, V, filter_diag, XV=None, R_train=None):
        super().__init__()
        self.V = V # (n_items, k)
        self.filter_diag = filter_diag
        self.XV = XV # (n_users, k)
        self.R_train = R_train 
        self.n_items = V.shape[0]

    def forward(self, user_ids):
        if self.XV is not None:
            xv_batch = self.XV[user_ids]
        else:
            X_batch = self.R_train[user_ids]
            xv_batch = torch.mm(X_batch, self.V)
            
        XVh = xv_batch * self.filter_diag
        scores = torch.mm(XVh, self.V.t())
        return scores


# ── 효율적인 HPO용 NDCG 계산 ──────────────────────────

def fast_evaluate_ndcg(scores, ground_truth, user_history, k_target=10):
    batch_size, n_items = scores.shape
    u_ids = list(ground_truth.keys())
    
    # 1. Masking
    for idx, u_id in enumerate(u_ids):
        history = user_history.get(u_id, set())
        gt = set(ground_truth[u_id])
        to_exclude = list(history - gt)
        if to_exclude:
            scores[idx, to_exclude] = -1e9

    # 2. Top-K
    _, top_indices = torch.topk(scores, k=k_target, dim=1)
    top_indices = top_indices.cpu().numpy()
    
    all_ndcg = []
    for idx, u_id in enumerate(u_ids):
        pred_list = top_indices[idx]
        gt = ground_truth[u_id]
        
        dcg = 0.0
        for i, item in enumerate(pred_list):
            if item in gt:
                dcg += 1.0 / np.log2(i + 2)
        
        idcg = 0.0
        n_relevant = min(len(gt), k_target)
        for i in range(n_relevant):
            idcg += 1.0 / np.log2(i + 2)
            
        all_ndcg.append(dcg / idcg if idcg > 0 else 0.0)
        
    return np.mean(all_ndcg) if all_ndcg else 0.0


# ── Bayesian HPO (AspireHPO) 통합 ───────────────────────────────────────────
# AspireHPO는 scripts/bayesian_opt.py의 BayesianOptimizer 패턴을 따르는 경량 래퍼.
# (TPESampler, EarlyStoppingCallback, save_results 포함)

def find_best_params_bayesian(XV_val, S, V, val_gt, val_history, device,
                              beta_val=None, n_trials=30, patience=20, seed=42,
                              study_name="HPO", out_dir=None):
    """
    beta_val이 지정되면 alpha만 검색,
    beta_val이 None이면 (alpha, beta) 공동 검색.
    out_dir이 지정되면 시각화 및 CSV를 저장 (프레임워크 BayesianOptimizer와 동일).
    """
    XV_val = XV_val.to(device)
    S = S.to(device)
    V = V.to(device)

    def objective_fn(params):
        alpha = params['alpha']
        beta  = params.get('beta', beta_val)
        h = AspireEngine.apply_filter(S, alpha, beta).to(device)
        scores = torch.mm(XV_val * h, V.t())
        return fast_evaluate_ndcg(scores, val_gt, val_history, k_target=10)

    # 프레임워크 bayesian_opt.py와 동일한 파라미터 스펙 포맷
    params_spec = [
        {'name': 'alpha', 'type': 'float', 'range': '0.1 1000000.0', 'log': True},
    ]
    if beta_val is None:
        params_spec.append({'name': 'beta', 'type': 'float', 'range': '0.0 1.5'})

    hpo = AspireHPO(params_spec, n_trials=n_trials, patience=patience, seed=seed)
    return hpo.search(objective_fn, study_name=study_name, output_dir=out_dir)


# ── β 추정 사이드 로직 ──────────────────────────────────────

def estimate_beta_ols_trimmed(s, p_tilde, trim=0.05):
    k = len(s)
    lo, hi = max(1, int(k * trim)), max(1 + 5, int(k * (1 - trim)))
    s_, pt_ = s[lo:hi], p_tilde[lo:hi]
    mask = (s_ > 1e-9) & (pt_ > 1e-9)
    if mask.sum() < 4: return 0.5
    log_s = np.log(s_[mask]).reshape(-1, 1)
    log_pt = np.log(pt_[mask])
    lm = LinearRegression().fit(log_s, log_pt)
    return float(np.clip(lm.coef_[0] / 2.0, 0.0, 0.999))


# ── 메인 실험 ────────────────────────────────────────────────────────────────

def run_ablation(dataset_name, target_energy=0.95):
    print(f"\n{'='*60}\nRunning Ablation (v6: Bayesian) on {dataset_name}\n{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    # 1. Data & SVD
    loader, R, S, V, config = get_loader_and_svd(dataset_name, target_energy=target_energy)

    # 출력 디렉토리를 먼저 정의 (HPO 결과 저장용)
    dataset_label = config['dataset_name']
    out_dir = ensure_dir(f"aspire_experiments/output/ablation/{dataset_label}")
    hpo_dir  = ensure_dir(os.path.join(out_dir, "hpo"))

    R_train_tensor = torch.from_numpy(R.todense()).float().to(device)
    S_tensor = S.to(device)
    V_tensor = V.to(device)

    print(f"[HPO] Pre-computing Latent Projections (XV)...")
    XV_tensor = torch.mm(R_train_tensor, V_tensor)

    # 2. Validation Data Preparation (ONCE)
    print(f"[HPO] Collecting Validation Data...")
    val_loader = loader.get_validation_loader(batch_size=2048)
    val_users, val_items = [], []
    for u_batch, i_batch in val_loader:
        val_users.append(u_batch.numpy()); val_items.append(i_batch.numpy())
    val_users = np.concatenate(val_users)
    val_items = np.concatenate(val_items)

    val_df = pd.DataFrame({'u': val_users, 'i': val_items})
    val_gt = val_df.groupby('u')['i'].apply(list).to_dict()
    val_history = loader.train_user_history

    val_users_tensor = torch.LongTensor(list(val_gt.keys())).to(device)
    XV_val = XV_tensor[val_users_tensor]

    # 3. Beta Estimation
    s_np, V_np = S.cpu().numpy(), V.cpu().numpy()
    item_pops = np.array(R.sum(axis=0)).flatten()
    p_tilde = AspireEngine.compute_spp(V_np, item_pops)
    beta_huber, _ = AspireEngine.estimate_beta(S_tensor, p_tilde, dataset_name=dataset_name)
    beta_ols = estimate_beta_ols_trimmed(s_np, p_tilde)
    beta_slope, _ = AspireEngine.estimate_beta_from_slope(s_np, item_pops, dataset_name=dataset_name)
    beta_fixed, beta_direct = 0.5, 1.0
    beta_damped = float(beta_huber * 0.5)

    # 4. Joint Bayesian Search (Optimal Upper Bound) — AspireHPO 사용
    print("\n[HPO] Joint Bayesian Search (Alpha & Beta) for Upper Bound...")
    joint_params, joint_val = find_best_params_bayesian(
        XV_val, S_tensor, V_tensor, val_gt, val_history, device,
        beta_val=None, n_trials=50, patience=20, seed=42,
        study_name="Joint_HPO",
        out_dir=os.path.join(hpo_dir, "Joint_HPO"),
    )
    best_beta_hpo  = joint_params['beta']
    best_alpha_hpo = joint_params['alpha']
    print(f"  -> Best Joint: Beta={best_beta_hpo:.4f}, Alpha={best_alpha_hpo:.2f} (Val NDCG: {joint_val:.4f})")

    betas = {
        "(1) SPP+Huber": beta_huber,
        "(2) SPP+OLS":   beta_ols,
        "(3) Slope ratio": beta_slope,
        "(4) β=0.5 fixed": beta_fixed,
        "(6) SPP direct": beta_direct,
        "(7) 0.5 damping": beta_damped,
        "(5) β=HPO":     best_beta_hpo,
    }

    # 5. Final Evaluation
    eval_config = loader.config.copy()
    eval_config['metrics'] = ['NDCG', 'HeadNDCG', 'LongTailNDCG', 'Coverage']
    eval_config['top_k'] = [10]
    eval_config['long_tail_percent'] = 0.8

    test_loader = loader.get_full_test_loader(batch_size=2048)
    results_table = []

    print("\n[Evaluation] Running Final Test and Bayesian Alpha Fitting for each method...")
    for name, beta in betas.items():
        if name == "(5) β=HPO":
            alpha = best_alpha_hpo
        else:
            print(f"  Fitting Alpha for {name} (Beta={beta:.3f})...")
            safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')
            alpha_params, _ = find_best_params_bayesian(
                XV_val, S_tensor, V_tensor, val_gt, val_history, device,
                beta_val=beta, n_trials=30, patience=20, seed=42,
                study_name=f"Alpha_{safe_name}",
                out_dir=os.path.join(hpo_dir, f"Alpha_{safe_name}"),
            )
            alpha = alpha_params['alpha']

        h = AspireEngine.apply_filter(S_tensor, alpha, beta).to(device)
        model = ASPIREAblationModel(V_tensor, h, XV=XV_tensor)
        res = evaluate_metrics(model, loader, eval_config, device, test_loader, is_final=True)

        row = {
            "method": name, "beta": float(beta), "alpha": float(alpha),
            "ndcg_all":  res.get('NDCG@10', 0.0),
            "ndcg_head": res.get('HeadNDCG@10', 0.0),
            "ndcg_tail": res.get('LongTailNDCG@10', 0.0),
            "coverage":  res.get('Coverage@10', 0.0),
        }
        results_table.append(row)
        print(f"  {name:<15} | b={beta:.3f} a={alpha:.2f} | NDCG: {row['ndcg_all']:.4f} "
              f"(H:{row['ndcg_head']:.4f}/T:{row['ndcg_tail']:.4f}) | Cov: {row['coverage']:.3f}")

    # 6. Output & Visualization
    with open(os.path.join(out_dir, "result_per_method_bayesian.json"), 'w') as f:
        json.dump({"dataset": dataset_label, "methods": results_table, "k": int(V.shape[1])}, f, indent=4)

    plt.figure(figsize=(12, 6))
    m = [r['method'] for r in results_table]
    x = np.arange(len(m)); width = 0.25
    plt.bar(x - width, [r['ndcg_head'] for r in results_table], width, label='Head NDCG', color='lightcoral')
    plt.bar(x, [r['ndcg_all'] for r in results_table], width, label='Overall NDCG', color='skyblue')
    plt.bar(x + width, [r['ndcg_tail'] for r in results_table], width, label='Tail NDCG', color='lightgreen')
    plt.xticks(x, m, rotation=15, ha='right'); plt.ylabel("NDCG@10"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ablation_breakdown_bayesian.png"), dpi=150); plt.close()

    return results_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, nargs="+", default=["ml100k"])
    parser.add_argument("--energy", type=float, default=0.95)
    args = parser.parse_args()
    for ds in args.dataset: run_ablation(ds, args.energy)
