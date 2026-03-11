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
from scipy.stats import theilslopes
from sklearn.linear_model import LinearRegression

sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import get_loader_and_svd, ensure_dir, AspireHPO
from src.models.csar import beta_estimators
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

def run_ablation(dataset_name, target_energy=0.95, seeds=[42]):
    print(f"\n{'='*60}\nRunning Ablation (v6: Bayesian) on {dataset_name} with seeds {seeds}\n{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    # 1. Data & SVD
    loader, R, S, V, config = get_loader_and_svd(dataset_name, target_energy=target_energy)

    # 출력 디렉토리를 먼저 정의 (HPO 결과 저장용)
    dataset_label = config['dataset_name']
    out_dir = ensure_dir(f"aspire_experiments/output/ablation/{dataset_label}")
    hpo_dir  = ensure_dir(os.path.join(out_dir, "hpo"))

    print(f"[HPO] Pre-computing Latent Projections (XV)...")
    # Avoid R.todense() to prevent massive OOM on large datasets (e.g., ml-20m)
    XV_np = R.dot(V.cpu().numpy())
    XV_tensor = torch.from_numpy(XV_np).float().to(device)
    
    S_tensor = S.to(device)
    V_tensor = V.to(device)

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
        beta_val=None, n_trials=60, patience=20, seed=42,
        study_name="Joint_HPO",
        out_dir=os.path.join(hpo_dir, "Joint_HPO"),
    )
    best_beta_hpo  = joint_params['beta']
    best_alpha_hpo = joint_params['alpha']
    print(f"  -> Best Joint: Beta={best_beta_hpo:.4f}, Alpha={best_alpha_hpo:.2f} (Val NDCG: {joint_val:.4f})")

    # Mapping name to estimator_type for R2 retrieval
    name_to_type = {
        "SPP+Huber (sklearn)": "huber",
        "SPP+Huber (MAD)": "huber_mad",
        "SPP+LAD": "lad",
        "Slope ratio": "slope_ratio",
        "β=0.5 fixed": "fixed_0.5",
        "β=HPO (Joint)": "huber", # or just use a default
    }

    # ------------------------------------------------------------------
    # Legacy Estimators (Moved from beta_estimators.py for ablation only)
    # ------------------------------------------------------------------
    def _local_beta_ols(sigma_k, p_tilde_k):
        x, y = beta_estimators._log_xy(sigma_k, p_tilde_k)
        if len(x) < 2: return 0.5, 0.0
        A = np.column_stack([x, np.ones_like(x)])
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        slope, intercept = coef[0], coef[1]
        r2 = beta_estimators._compute_r2(x, y, slope, intercept)
        return float(np.clip(slope / 2, 0, 1)), float(r2)

    def _local_beta_theil_sen(sigma_k, p_tilde_k):
        x, y = beta_estimators._log_xy(sigma_k, p_tilde_k)
        if len(x) < 2: return 0.5, 0.0
        result = theilslopes(y, x)
        slope, intercept = result.slope, result.intercept
        r2 = beta_estimators._compute_r2(x, y, slope, intercept)
        return float(np.clip(slope / 2, 0, 1)), float(r2)

    def _local_beta_huber_fixed(sigma_k, p_tilde_k, delta_h=1.35):
        x, y = beta_estimators._log_xy(sigma_k, p_tilde_k)
        if len(x) < 2: return 0.5, 0.0
        A = np.column_stack([x, np.ones_like(x)])
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        for _ in range(100):
            resid = y - A @ coef
            w = np.where(np.abs(resid) <= delta_h, 1.0, delta_h / np.clip(np.abs(resid), 1e-9, None))
            W = np.diag(w)
            coef_new = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ y, rcond=None)[0]
            if np.max(np.abs(coef_new - coef)) < 1e-8: break
            coef = coef_new
        slope, intercept = coef[0], coef[1]
        r2 = beta_estimators._compute_r2(x, y, slope, intercept)
        return float(np.clip(slope / 2, 0, 1)), float(r2)

    # Note: SPP+OLS, Theil-Sen, Huber (Fixed) do not have corresponding R2 extraction logic in run_ablation
    # anymore since they are removed from the global name_to_type mapping.
    # They are kept here just for the NDCG ablation evaluation.

    S_tensor_np = S_tensor.detach().cpu().numpy()

    betas = {
        "SPP+Huber (sklearn)": beta_huber,
        "SPP+OLS":           _local_beta_ols(S_tensor_np, p_tilde)[0],
        "SPP+Theil-Sen":     _local_beta_theil_sen(S_tensor_np, p_tilde)[0],
        "SPP+Huber (Fixed)": _local_beta_huber_fixed(S_tensor_np, p_tilde)[0],
        "SPP+Huber (MAD)":   AspireEngine.estimate_beta(S_tensor, p_tilde, estimator_type="huber_mad")[0],
        "SPP+LAD":           AspireEngine.estimate_beta(S_tensor, p_tilde, estimator_type="lad")[0],
        # "Slope ratio":       beta_slope,
        "β=0.5 fixed":       beta_fixed,
        "β=HPO (Joint)":     best_beta_hpo,
    }

    # 5. Final Evaluation across multiple seeds
    eval_config = loader.config.copy()
    eval_config['metrics'] = ['NDCG', 'HeadNDCG', 'LongTailNDCG', 'Coverage']
    eval_config['top_k'] = [10]
    eval_config['long_tail_percent'] = 0.8
    test_loader = loader.get_full_test_loader(batch_size=2048)

    results_table = []
    
    print(f"\n[Evaluation] Running Final Test across {len(seeds)} seeds: {seeds}")
    
    for name, beta in betas.items():
        seed_results = []
        for seed in seeds:
            print(f"  Fitting Alpha for {name} (Beta={beta:.3f}, Seed={seed})...")
            
            if name == "β=HPO (Joint)":
                # For Joint HPO, we already found the best alpha/beta above using seed=42 (or first seed).
                # To be rigorous, we could re-run joint HPO per seed, but for ablation, reusing the best joint
                # params and just re-evaluating is faster, or we re-run alpha fit with the best joint beta.
                # Let's re-run only alpha fitting using the fixed best Joint beta to be consistent with others.
                alpha_params, _ = find_best_params_bayesian(
                    XV_val, S_tensor, V_tensor, val_gt, val_history, device,
                    beta_val=best_beta_hpo, n_trials=60, patience=20, seed=seed,
                    study_name=f"Alpha_Joint_HPO_{seed}",
                    out_dir=os.path.join(hpo_dir, f"Alpha_Joint_HPO_{seed}"),
                )
                alpha = alpha_params['alpha']
            else:
                safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')
                alpha_params, _ = find_best_params_bayesian(
                    XV_val, S_tensor, V_tensor, val_gt, val_history, device,
                    beta_val=beta, n_trials=60, patience=20, seed=seed,
                    study_name=f"Alpha_{safe_name}_{seed}",
                    out_dir=os.path.join(hpo_dir, f"Alpha_{safe_name}_{seed}"),
                )
                alpha = alpha_params['alpha']

            h = AspireEngine.apply_filter(S_tensor, alpha, beta).to(device)
            model = ASPIREAblationModel(V_tensor, h, XV=XV_tensor)
            res = evaluate_metrics(model, loader, eval_config, device, test_loader, is_final=True)
            
            seed_results.append({
                "alpha": float(alpha),
                "ndcg_all":  res.get('NDCG@10', 0.0),
                "ndcg_head": res.get('HeadNDCG@10', 0.0),
                "ndcg_tail": res.get('LongTailNDCG@10', 0.0),
                "coverage":  res.get('Coverage@10', 0.0)
            })
            
        # Average results over seeds
        avg_alpha = np.mean([r["alpha"] for r in seed_results])
        avg_ndcg_all = np.mean([r["ndcg_all"] for r in seed_results])
        avg_ndcg_head = np.mean([r["ndcg_head"] for r in seed_results])
        avg_ndcg_tail = np.mean([r["ndcg_tail"] for r in seed_results])
        avg_coverage = np.mean([r["coverage"] for r in seed_results])
        
        # Get Fit Diagnostics (R2)
        _, r2 = AspireEngine.estimate_beta(S_tensor, p_tilde, estimator_type=name_to_type.get(name, "huber"))

        row = {
            "method": name, "beta": float(beta), "alpha": float(avg_alpha), "r2": float(r2),
            "ndcg_all":  float(avg_ndcg_all),
            "ndcg_head": float(avg_ndcg_head),
            "ndcg_tail": float(avg_ndcg_tail),
            "coverage":  float(avg_coverage),
            "seeds": seeds,
        }
        results_table.append(row)
        print(f"  > {name:<15} | b={beta:.3f} avg_a={avg_alpha:.2f} | avg_NDCG: {row['ndcg_all']:.4f} "
              f"(H:{row['ndcg_head']:.4f}/T:{row['ndcg_tail']:.4f}) | Cov: {row['coverage']:.3f}")

    # 6. Output & Visualization
    with open(os.path.join(out_dir, "result_per_method_bayesian.json"), 'w', encoding='utf-8') as f:
        json.dump({"dataset": dataset_label, "seeds": seeds, "methods": results_table, "k": int(V.shape[1])}, f, indent=4)

    m = [r['method'] for r in results_table]
    x = np.arange(len(m))
    head_vals = [r['ndcg_head'] for r in results_table]
    all_vals  = [r['ndcg_all'] for r in results_table]
    tail_vals = [r['ndcg_tail'] for r in results_table]

    # 3개의 완전히 분리된 서브플롯 생성
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.subplots_adjust(hspace=0.2)

    # 1. Overall NDCG
    axes[0].bar(x, all_vals, color='skyblue', edgecolor='black')
    axes[0].set_title('Overall NDCG@10', fontsize=12, pad=10)
    axes[0].set_ylabel('NDCG')
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)
    axes[0].set_ylim([max(0, min(all_vals) * 0.9), max(all_vals) * 1.05])

    # 2. Head NDCG
    axes[1].bar(x, head_vals, color='lightcoral', edgecolor='black')
    axes[1].set_title('Head NDCG@10', fontsize=12, pad=10)
    axes[1].set_ylabel('NDCG')
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)
    axes[1].set_ylim([max(0, min(head_vals) * 0.9), max(head_vals) * 1.05])

    # 3. Tail NDCG
    axes[2].bar(x, tail_vals, color='lightgreen', edgecolor='black')
    axes[2].set_title('Long-Tail NDCG@10', fontsize=12, pad=10)
    axes[2].set_ylabel('NDCG')
    axes[2].grid(axis='y', linestyle='--', alpha=0.6)
    
    # 꼬리(Tail) 값들은 편차가 작고 0에 가깝기 때문에 하한을 0에 맞추거나 타이트하게 조정
    t_min = min(tail_vals)
    t_max = max(tail_vals)
    y_margin = (t_max - t_min) * 0.15 if t_max > t_min else t_max * 0.1
    axes[2].set_ylim([max(0, t_min - y_margin), t_max + y_margin])

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(m, rotation=30, ha='right', fontsize=10)

    # 전체 타이틀 및 마무리
    fig.suptitle(f'Ablation Study Results ({dataset_label})', fontsize=14, y=0.96)
    plt.savefig(os.path.join(out_dir, "ablation_breakdown_bayesian.png"), dpi=150, bbox_inches='tight')
    plt.close()

    return results_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, nargs="+", default=["ml100k"])
    parser.add_argument("--energy", type=float, default=0.95)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="List of random seeds to use (e.g. 42 43 44)")
    args = parser.parse_args()
    for ds in args.dataset: run_ablation(ds, args.energy, args.seeds)
