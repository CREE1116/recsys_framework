import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from scipy.sparse import csr_matrix
from sklearn.metrics import r2_score

# Framework root path
sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import load_config, ensure_dir, AspireHPO
from src.data_loader import DataLoader
from src.utils.gpu_accel import get_device, gpu_gram_solve, SVDCacheManager, EVDCacheManager

def train_ease(X_train, reg, device):
    P = gpu_gram_solve(X_train, max(reg, 1e-4), device=device, return_tensor=True)
    B = -P / torch.diagonal(P).view(1, -1).clamp(min=1e-12)
    B.fill_diagonal_(0)
    return B.float()

def train_aspire(X_train, gamma, k, device):
    manager = SVDCacheManager(device=device)
    _, s, V, _ = manager.get_svd(X_train, k=k, dataset_name=None, force_recompute=True)
    s_f = s.float()
    ref_val = torch.mean(s_f[:max(1, int(len(s_f)*0.01))])
    h = torch.pow(s_f, gamma) / (torch.pow(s_f, gamma) + torch.pow(ref_val, gamma) + 1e-12)
    B = torch.mm(V * h, V.t())
    B.fill_diagonal_(0)
    return B.float()

def analyze_spectral(X_sparse):
    manager = EVDCacheManager()
    _, S, V, _ = manager.get_evd(X_sparse, k=None, force_recompute=True)
    sigma = np.sqrt(np.maximum(S.cpu().numpy(), 1e-12))
    n_i = np.array(X_sparse.sum(axis=0)).flatten()
    pi_i = n_i / (n_i.sum() + 1e-12)
    pk = np.sum(pi_i[:, np.newaxis] * (V.cpu().numpy()**2), axis=0)
    x, y = np.log(sigma + 1e-12), np.log(pk + 1e-12)
    slope, intercept = np.polyfit(x, y, 1)
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2_score(y, slope*x + intercept)), "x": x.tolist(), "y": y.tolist()}

def run_hpo(X_train, model_type, device, n_trials=20, t=0):
    X_coo = X_train.tocoo()
    idx = np.random.permutation(len(X_coo.data))
    v_size = int(len(idx) * 0.1)
    X_t = csr_matrix((X_coo.data[idx[v_size:]], (X_coo.row[idx[v_size:]], X_coo.col[idx[v_size:]])), shape=X_train.shape)
    X_v = csr_matrix((X_coo.data[idx[:v_size]], (X_coo.row[idx[:v_size]], X_coo.col[idx[:v_size]])), shape=X_train.shape)
    X_v_cpu, X_t_dense = torch.from_numpy(X_v.toarray()).float(), torch.from_numpy(X_t.toarray()).float().to(device)

    def objective(p):
        B = train_ease(X_t, p['reg'], device) if model_type == 'ease' else train_aspire(X_t, p['gamma'], int(p['k']), device)
        if torch.isnan(B).any(): return 0.0
        Scores = (X_t_dense @ B).cpu()
        Scores[X_t_dense.cpu() > 0] = -1e9
        _, top_idx = torch.topk(Scores, k=10, dim=1)
        hits = X_v_cpu.gather(1, top_idx)
        dcg = (hits / torch.log2(torch.arange(2, 12, dtype=torch.float32))).sum(dim=1)
        return dcg.mean().item()

    spec = [{'name': 'reg', 'type': 'float', 'range': '0.01 1000.0', 'log': True}] if model_type == 'ease' else \
           [{'name': 'gamma', 'type': 'float', 'range': '0.01 2.0', 'log': True},
            {'name': 'k', 'type': 'int', 'range': f'50 {min(X_train.shape)-1}', 'log': True}]
    
    best_params, _ = AspireHPO(spec, n_trials=n_trials, patience=10, seed=42+t).search(objective, study_name=model_type)
    return {k: (float(v) if k != 'k' else int(v)) for k, v in best_params.items()}

def run_simulation(args, model_type, device):
    print(f"\n>>> {model_type.upper()} Simulation")
    np.random.seed(42)
    nnz = int(args.num_users * args.num_items * args.density)
    X = csr_matrix((np.ones(nnz, dtype=np.float32), (np.random.randint(0, args.num_users, nnz), np.random.randint(0, args.num_items, nnz))), shape=(args.num_users, args.num_items))
    X.data = np.ones_like(X.data)
    
    history, params_log = [analyze_spectral(X)], []
    for t in range(1, args.iter + 1):
        p = run_hpo(X, model_type, device, n_trials=20, t=t)
        params_log.append(p)
        B = train_ease(X, p['reg'], device) if model_type == 'ease' else train_aspire(X, p['gamma'], p['k'], device)
        Scores = (torch.from_numpy(X.toarray()).float().to(device) @ B).cpu()
        Scores[torch.from_numpy(X.toarray() > 0)] = -1e9
        Scores_np = Scores.numpy()
        
        new_r, new_c = [], []
        for u in range(X.shape[0]):
            top_idx = np.argsort(Scores_np[u])[-200:]
            probs = np.exp((Scores_np[u][top_idx] - np.max(Scores_np[u][top_idx])) / 0.5)
            clicked = np.random.choice(top_idx, size=args.num_clicks, replace=False, p=probs/probs.sum())
            for i in clicked: new_r.append(u); new_c.append(i)
            
        X = X + csr_matrix((np.ones(len(new_r)), (new_r, new_c)), shape=X.shape)
        X.data = np.ones_like(X.data)
        if t % 5 == 0 or t == args.iter:
            print(f"  Iter {t}/{args.iter} | Slope: {analyze_spectral(X)['slope']:.4f} | Params: {p}")
            
    history.append(analyze_spectral(X))
    return history, params_log

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=40)
    parser.add_argument("--num_users", type=int, default=1000); parser.add_argument("--num_items", type=int, default=1000)
    parser.add_argument("--density", type=float, default=0.01); parser.add_argument("--num_clicks", type=int, default=2)
    args = parser.parse_args()

    device, out_dir = get_device(), ensure_dir("aspire_experiments/output/exp16")
    results = {m: dict(zip(['history', 'params'], run_simulation(args, m, device))) for m in ['ease', 'aspire']}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    h_all = results['ease']['history'] + results['aspire']['history']
    xlim, ylim = (min(min(h['x']) for h in h_all), max(max(h['x']) for h in h_all)), (min(min(h['y']) for h in h_all), max(max(h['y']) for h in h_all))

    for i, m in enumerate(['ease', 'aspire']):
        h0, ht = results[m]['history'][0], results[m]['history'][-1]
        axes[i].scatter(h0['x'], h0['y'], c='gray', alpha=0.2, s=5, label='Initial')
        axes[i].scatter(ht['x'], ht['y'], c=('blue' if m=='ease' else 'orange'), alpha=0.5, s=10, label=f'Final (Slope {ht["slope"]:.2f})')
        axes[i].plot(ht['x'], np.array(ht['x'])*ht['slope'] + ht['intercept'], 'k--')
        axes[i].set_xlim(xlim); axes[i].set_ylim(ylim); axes[i].set_title(f"{m.upper()} Evolution"); axes[i].legend()

    ax3 = axes[2]; iters = np.arange(1, args.iter+1)
    ax3.plot(iters, [p['reg'] for p in results['ease']['params']], 'b-', label='EASE reg')
    ax3_tw = ax3.twinx(); ax3_tw.plot(iters, [p['gamma'] for p in results['aspire']['params']], 'orange', label='ASPIRE gamma')
    ax3.set_title("Parameter Drift"); ax3.legend(loc='upper left'); ax3_tw.legend(loc='upper right')

    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "summary_plot.png"), dpi=150)
    for m in results:
        for h in results[m]['history']: h.pop('x', None); h.pop('y', None)
    with open(os.path.join(out_dir, "results.json"), "w") as f: json.dump({"config": vars(args), "results": results}, f, indent=4)
    print(f"\n>>> Results saved to: {out_dir}")

if __name__ == "__main__": main()
