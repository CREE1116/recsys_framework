# Usage: uv run python aspire_experiments/exp5_ablation.py --dataset ml1m --energy 0.95
#
# §6.3 실험 C: β 추정 방법 Ablation (v5 - Efficiency & Alignment)
# 프레임워크 로직과 NDCG 계산 방식을 엄격하게 준수하되, HPO 시 불필요한 데이터 수집 반복 제거.
#
# 비교 방법:
#   (1) SPP + Huber (Proposed)
#   (2) SPP + OLS
#   (3) Slope ratio (ChebyASPIRE 방식)
#   (4) β = 0.5 fixed
#   (5) β = HPO (Joint)
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
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression

sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import get_loader_and_svd, ensure_dir
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


# ── 효율적인 HPO용 NDCG 계산 (Framework Logic 동기화) ──────────────────────────

def fast_evaluate_ndcg(scores, ground_truth, user_history, k_target=10):
    """
    scores: (batch_size, n_items) torch.Tensor
    ground_truth: dict {u_id: [item_ids]}
    user_history: dict {u_id: {item_ids}}
    """
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
    # 3. NDCG Calculation (same as src/evaluation.py)
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


# ── 최적의 Alpha 찾기 ────────────────────────────────────────

def find_best_alpha(XV, S, V, beta, alpha_grid, val_gt, val_history, device):
    best_alpha, best_ndcg = 1.0, -1.0
    val_users = torch.LongTensor(list(val_gt.keys())).to(device)
    
    # Pre-calculate XV_val once
    XV_val = XV[val_users]
    
    for alpha in alpha_grid:
        h = AspireEngine.apply_filter(S, alpha, beta).to(device)
        # Fast score calculation
        scores = torch.mm(XV_val * h, V.t())
        score = fast_evaluate_ndcg(scores, val_gt, val_history, k_target=10)
        
        if score > best_ndcg:
            best_ndcg, best_alpha = score, alpha
            
    return best_alpha


# ── 메인 실험 ────────────────────────────────────────────────────────────────

def run_ablation(dataset_name, target_energy=0.95):
    print(f"\n{'='*60}\nRunning Ablation (v5: Optimized) on {dataset_name}\n{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    # 1. Data & SVD
    loader, R, S, V, config = get_loader_and_svd(dataset_name, target_energy=target_energy)
    
    R_train_tensor = torch.from_numpy(R.todense()).float().to(device)
    S_tensor = S.to(device)
    V_tensor = V.to(device)
    
    print(f"[HPO] Pre-computing Latent Projections (XV)...")
    XV_tensor = torch.mm(R_train_tensor, V_tensor)
    
    # 2. Validation Ground Truth Collection (ONCE)
    print(f"[HPO] Collecting Validation Data once...")
    val_loader = loader.get_validation_loader(batch_size=2048)
    val_users, val_items = [], []
    for u_batch, i_batch in val_loader:
        val_users.append(u_batch.numpy()); val_items.append(i_batch.numpy())
    val_users, val_items = np.concatenate(val_users), np.concatenate(val_items)
    val_df = pd.DataFrame({'u': val_users, 'i': val_items})
    val_gt = val_df.groupby('u')['i'].apply(list).to_dict()
    val_history = loader.train_user_history # Validation시는 train만 마스킹

    # 3. Beta Estimation
    s_np, V_np = S.cpu().numpy(), V.cpu().numpy()
    item_pops = np.array(R.sum(axis=0)).flatten()
    p_tilde = AspireEngine.compute_spp(V_np, item_pops)
    beta_huber, _ = AspireEngine.estimate_beta(S_tensor, p_tilde, dataset_name=dataset_name)
    beta_ols = estimate_beta_ols_trimmed(s_np, p_tilde)
    beta_slope, _ = AspireEngine.estimate_beta_from_slope(s_np, item_pops, dataset_name=dataset_name)
    beta_fixed, beta_direct = 0.5, 1.0
    beta_damped = float(beta_huber * 0.5)

    # 4. HPO Sweep
    print("\n[HPO] Joint Beta/Alpha Sweep for Upper Bound...")
    beta_grid = np.linspace(0.0, 1.5, 16)
    alpha_grid = np.logspace(-1, 6, 30)
    
    best_beta_hpo, best_alpha_hpo, max_val_ndcg = 0.5, 1000.0, -1.0
    for b in tqdm(beta_grid, desc="Beta Sweep"):
        a = find_best_alpha(XV_tensor, S_tensor, V_tensor, b, alpha_grid, val_gt, val_history, device)
        # Re-check best score
        h = AspireEngine.apply_filter(S_tensor, a, b).to(device)
        score = fast_evaluate_ndcg(torch.mm(XV_tensor[torch.LongTensor(list(val_gt.keys())).to(device)] * h, V_tensor.t()), val_gt, val_history)
        if score > max_val_ndcg:
            max_val_ndcg, best_beta_hpo, best_alpha_hpo = score, b, a

    betas = {
        "(1) SPP+Huber": beta_huber,
        "(2) SPP+OLS": beta_ols,
        "(3) Slope ratio": beta_slope,
        "(4) β=0.5 fixed": beta_fixed,
        "(6) SPP direct": beta_direct,
        "(7) 0.5 damping": beta_damped,
        "(5) β=HPO": best_beta_hpo
    }

    # 5. Final Evaluation (Framework Aligned)
    eval_config = loader.config.copy()
    eval_config['metrics'] = ['NDCG', 'HeadNDCG', 'LongTailNDCG', 'Coverage']
    eval_config['top_k'] = [10]
    eval_config['long_tail_percent'] = 0.8
    
    test_loader = loader.get_full_test_loader(batch_size=2048)
    results_table = []
    
    print("\n[Evaluation] Running Final Test and Alpha HPO for each method...")
    for name, beta in betas.items():
        if name == "(5) β=HPO":
            alpha = best_alpha_hpo
        else:
            alpha = find_best_alpha(XV_tensor, S_tensor, V_tensor, beta, alpha_grid, val_gt, val_history, device)
            
        model = ASPIREAblationModel(V_tensor, AspireEngine.apply_filter(S_tensor, alpha, beta).to(device), XV=XV_tensor)
        res = evaluate_metrics(model, loader, eval_config, device, test_loader, is_final=True)
        
        row = {
            "method": name, "beta": float(beta), "alpha": float(alpha),
            "ndcg_all": res.get('NDCG@10', 0.0), "ndcg_head": res.get('HeadNDCG@10', 0.0),
            "ndcg_tail": res.get('LongTailNDCG@10', 0.0), "coverage": res.get('Coverage@10', 0.0)
        }
        results_table.append(row)
        print(f"  {name:<15} | b={beta:.3f} a={alpha:.0f} | NDCG: {row['ndcg_all']:.4f} (H:{row['ndcg_head']:.4f}/T:{row['ndcg_tail']:.4f}) | Cov: {row['coverage']:.3f}")

    # 6. Output & Visualization
    dataset_label = config['dataset_name']
    out_dir = ensure_dir(f"aspire_experiments/output/ablation/{dataset_label}")
    with open(os.path.join(out_dir, "result_per_method.json"), 'w') as f:
        json.dump({"dataset": dataset_label, "methods": results_table, "k": int(V.shape[1])}, f, indent=4)

    plt.figure(figsize=(12, 6))
    m = [r['method'] for r in results_table]
    x = np.arange(len(m)); width = 0.25
    plt.bar(x - width, [r['ndcg_head'] for r in results_table], width, label='Head NDCG', color='lightcoral')
    plt.bar(x, [r['ndcg_all'] for r in results_table], width, label='Overall NDCG', color='skyblue')
    plt.bar(x + width, [r['ndcg_tail'] for r in results_table], width, label='Tail NDCG', color='lightgreen')
    plt.xticks(x, m, rotation=15, ha='right'); plt.ylabel("NDCG@10"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ablation_breakdown.png"), dpi=150); plt.close()

    return results_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, nargs="+", default=["ml100k"])
    parser.add_argument("--energy", type=float, default=0.95)
    args = parser.parse_args()
    for ds in args.dataset: run_ablation(ds, args.energy)
