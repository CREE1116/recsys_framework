import os
import sys
import torch
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import skew, kurtosis

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_loader import DataLoader
from src.models.general.mf import MF

# ==========================================================
# Hardcoded Variables
# ==========================================================
BASE_DIR = "/Users/leejongmin/code/recsys_framework"
EXP_BPR = os.path.join(BASE_DIR, "trained_model/ml-100k/mf__loss_type=pairwise")
EXP_MSE = os.path.join(BASE_DIR, "trained_model/ml-100k/mf__loss_type=pointwise")

NUM_USERS = 200 
OUTPUT_DIR = "/Users/leejongmin/code/recsys_framework/output/mf/"
PLOT_FILE = os.path.join(OUTPUT_DIR, "score_dist_comparison_comprehensive.png")
STATS_FILE = os.path.join(OUTPUT_DIR, "analysis_results.json")
# ==========================================================

def load_model_and_config(exp_dir, device):
    config_path = os.path.join(exp_dir, 'config.yaml')
    model_path = os.path.join(exp_dir, 'best_model.pt')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize DataLoader
    data_loader = DataLoader(config)
    
    # Initialize Model
    model = MF(config, data_loader)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, config, data_loader

def get_comprehensive_stats(model, data_loader, device, num_users=None):
    n_users = data_loader.n_users
    n_items = data_loader.n_items
    
    # Get test ground truth
    test_gt = data_loader.test_df.groupby('user_id')['item_id'].apply(list).to_dict()
    test_users = sorted(list(test_gt.keys()))
    
    # Use specified number of users or all
    if num_users:
        user_ids = np.random.choice(test_users, min(len(test_users), num_users), replace=False)
    else:
        user_ids = test_users
        
    pos_scores_all = []
    neg_scores_all = []
    per_user_metrics = []
    
    # Masking: For test evaluation, we mask everything in train + valid
    mask_history = data_loader.eval_user_history
    
    with torch.no_grad():
        for u_id in tqdm(user_ids, desc=f"Analyzing {model.loss_type}"):
            u_tensor = torch.LongTensor([u_id]).to(device)
            scores = model.forward(u_tensor).cpu().numpy().flatten()
            
            # Positive items (Test set)
            pos_items = test_gt.get(u_id, [])
            if not pos_items: continue
                
            # Negative items (Exclude train + valid + test items)
            # Standard RecBole evaluation: items not in (train+valid+test)
            all_seen = data_loader.user_history.get(u_id, set()) 
            neg_items = list(set(range(n_items)) - all_seen)
            if not neg_items: continue
            
            p_scores = scores[pos_items]
            n_scores = scores[neg_items]
            
            # Per-user separation
            u_sep = np.mean(p_scores) - np.mean(n_scores)
            
            # Per-user AUC
            n_pos = len(p_scores)
            n_neg = len(n_scores)
            correct_pairs = 0
            for ps in p_scores:
                correct_pairs += np.sum(ps > n_scores)
            u_auc = correct_pairs / (n_pos * n_neg) if (n_pos * n_neg) > 0 else 0.5
            
            per_user_metrics.append({
                'user_id': int(u_id),
                'separation': float(u_sep),
                'auc': float(u_auc),
                'n_pos': n_pos
            })
            
            pos_scores_all.extend(p_scores)
            neg_scores_all.extend(n_scores)
            
    return np.array(pos_scores_all), np.array(neg_scores_all), per_user_metrics

def calculate_stats(scores):
    return {
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores)),
        'median': float(np.median(scores)),
        'min': float(np.min(scores)),
        'max': float(np.max(scores)),
        'q1': float(np.percentile(scores, 25)),
        'q3': float(np.percentile(scores, 75)),
        'iqr': float(np.percentile(scores, 75) - np.percentile(scores, 25)),
        'skew': float(skew(scores)),
        'kurtosis': float(kurtosis(scores))
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Models
    model_bpr, _, loader_bpr = load_model_and_config(EXP_BPR, device)
    model_mse, _, loader_mse = load_model_and_config(EXP_MSE, device)
    
    # Analyze
    bpr_pos, bpr_neg, bpr_user = get_comprehensive_stats(model_bpr, loader_bpr, device, NUM_USERS)
    mse_pos, mse_neg, mse_user = get_comprehensive_stats(model_mse, loader_mse, device, NUM_USERS)
    
    # Global Stats
    results = {
        'BPR': {
            'pos': calculate_stats(bpr_pos),
            'neg': calculate_stats(bpr_neg),
            'global_separation': float(np.mean(bpr_pos) - np.mean(bpr_neg)),
            'avg_user_auc': float(np.mean([u['auc'] for u in bpr_user])),
            'avg_user_sep': float(np.mean([u['separation'] for u in bpr_user]))
        },
        'WMSE': {
            'pos': calculate_stats(mse_pos),
            'neg': calculate_stats(mse_neg),
            'global_separation': float(np.mean(mse_pos) - np.mean(mse_neg)),
            'avg_user_auc': float(np.mean([u['auc'] for u in mse_user])),
            'avg_user_sep': float(np.mean([u['separation'] for u in mse_user]))
        }
    }
    
    # Save JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(STATS_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Stats saved to {STATS_FILE}")
    
    # Visualization
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.2)
    
    # 1. KDE Plots (BPR & WMSE)
    def plot_kde(ax, pos, neg, title):
        sns.kdeplot(pos, ax=ax, color="green", label="Positive", fill=True, alpha=0.3)
        sns.kdeplot(neg, ax=ax, color="red", label="Negative", fill=True, alpha=0.1)
        ax.set_title(f"{title}: Score Density")
        ax.legend()
        
    ax1 = fig.add_subplot(gs[0, 0])
    plot_kde(ax1, bpr_pos, bpr_neg, "BPR (Pairwise)")
    ax2 = fig.add_subplot(gs[0, 1])
    plot_kde(ax2, mse_pos, mse_neg, "WMSE (Pointwise)")
    
    # 2. Violin Plots (Comparison)
    ax3 = fig.add_subplot(gs[1, 0])
    data_violin = []
    # Use sampled data for violin plots if too many
    sample_size = 5000
    bpr_p_sample = np.random.choice(bpr_pos, min(len(bpr_pos), sample_size))
    bpr_n_sample = np.random.choice(bpr_neg, min(len(bpr_neg), sample_size))
    mse_p_sample = np.random.choice(mse_pos, min(len(mse_pos), sample_size))
    mse_n_sample = np.random.choice(mse_neg, min(len(mse_neg), sample_size))
    
    sns.violinplot(data=[bpr_p_sample, bpr_n_sample], ax=ax3, palette=["green", "red"])
    ax3.set_xticklabels(["Pos", "Neg"])
    ax3.set_title("BPR Score Range & Density")
    
    ax4 = fig.add_subplot(gs[1, 1])
    sns.violinplot(data=[mse_p_sample, mse_n_sample], ax=ax4, palette=["green", "red"])
    ax4.set_xticklabels(["Pos", "Neg"])
    ax4.set_title("WMSE Score Range & Density")
    
    # 3. Per-User Metric Distributions
    ax5 = fig.add_subplot(gs[2, 0])
    bpr_auc_dist = [u['auc'] for u in bpr_user]
    mse_auc_dist = [u['auc'] for u in mse_user]
    sns.histplot(bpr_auc_dist, color="blue", label="BPR AUC", kde=True, ax=ax5, alpha=0.5)
    sns.histplot(mse_auc_dist, color="orange", label="WMSE AUC", kde=True, ax=ax5, alpha=0.5)
    ax5.set_title("Per-User AUC Distribution")
    ax5.set_xlabel("AUC Score")
    ax5.legend()
    
    ax6 = fig.add_subplot(gs[2, 1])
    # Boxplot of AUC comparison
    sns.boxplot(data=[bpr_auc_dist, mse_auc_dist], ax=ax6, palette=["blue", "orange"])
    ax6.set_xticklabels(["BPR", "WMSE"])
    ax6.set_title("Model AUC Comparison (Bootstrap)")
    
    plt.suptitle("Matrix Factorization: Comprehensive Loss Function Analysis", fontsize=22)
    plt.savefig(PLOT_FILE)
    print(f"Comprehensive plot saved to {PLOT_FILE}")
    
    # Console Summary
    print("\n" + "="*50)
    print(f"{'Metric':<20} | {'BPR':<10} | {'WMSE':<10}")
    print("-"*50)
    print(f"{'Avg User AUC':<20} | {results['BPR']['avg_user_auc']:.4f} | {results['WMSE']['avg_user_auc']:.4f}")
    print(f"{'Separation (Mean)':<20} | {results['BPR']['global_separation']:.4f} | {results['WMSE']['global_separation']:.4f}")
    print(f"{'Pos Skew':<20} | {results['BPR']['pos']['skew']:.4f} | {results['WMSE']['pos']['skew']:.4f}")
    print(f"{'Neg Skew':<20} | {results['BPR']['neg']['skew']:.4f} | {results['WMSE']['neg']['skew']:.4f}")
    print("="*50)
    
    # 4. Margin Matching Analysis (User Request)
    print("\n[Margin Matching Analysis]")
    bpr_margin = results['BPR']['global_separation']
    mse_margin = results['WMSE']['global_separation']
    
    if mse_margin > 0:
        # Linear Scaling Factor
        k = bpr_margin / mse_margin
        print(f"  - Linear Scaling (k * x): k ≈ {k:.2f} (Embedding Scale α ≈ {np.sqrt(k):.2f})")
        
        pos_m = results['WMSE']['pos']['mean']
        neg_m = results['WMSE']['neg']['mean']
        
        from scipy.optimize import fsolve

        # Power Transform (sign(x) * |x|^p)
        # Find p such that sign(pos)*|pos|^p - sign(neg)*|neg|^p = bpr_margin
        def sign_pow_func(p):
            p_val = np.sign(pos_m) * np.power(np.abs(pos_m), p)
            n_val = np.sign(neg_m) * np.power(np.abs(neg_m), p)
            return (p_val - n_val) - bpr_margin
        
        try:
            # Note: For x in (0, 1), decreasing p increases x^p.
            # To get a positive margin of 6.6 with pos > neg, 
            # we'd need pos^p > neg^p + 6.6. 
            # But for p < 0, if pos > neg, then pos^p < neg^p.
            # So (pos^p - neg^p) will be NEGATIVE for p < 0.
            # If p > 0, pos^p - neg^p is positive but stays <= 1.
            # Thus, for pos, neg in (0, 1), NO p can satisfy (pos^p - neg^p) = 6.6
            
            p_pow = fsolve(sign_pow_func, 1.0)[0]
            # Since fsolve might return a garbage value if no solution exists, we verify
            err = sign_pow_func(p_pow)
            if np.abs(err) > 0.1:
                print(f"  - sign(x)*|x|^p: No mathematical solution found for p.")
                print(f"    (Reason: With scores in (0, 1), p > 0 stays in (0, 1) [margin < 1],")
                print(f"     and p < 0 flips the order making the margin negative.)")
            else:
                print(f"  - sign(x)*|x|^p: p ≈ {p_pow:.4f}")
        except Exception as e:
            print(f"  - Power transform analysis failed: {e}")

        # Ratio Matching (Relative Separation)
        # BPR model's separation 'd' can be seen as a logit gap.
        # We find p such that (pos_w / neg_w)^p matches the 'strength' of BPR,
        # i.e., log(ratio^p) = bpr_margin => p * log(pos_w/neg_w) = bpr_margin
        try:
            ratio_w = pos_m / max(1e-9, neg_m)
            p_ratio = bpr_margin / np.log(ratio_w)
            print(f"  - Ratio Power (x^p): p ≈ {p_ratio:.4f}")
            print(f"    (Matches the logit gap of BPR while PRESERVING RANKING order)")
        except Exception as e:
            print(f"  - Ratio power analysis failed: {e}")

        # Exponential Scaling (Inverse Temperature)
        def exp_func(p):
            return np.exp(p * pos_m) - np.exp(p * neg_m) - bpr_margin
        
        try:
            p_exp = fsolve(exp_func, 20.0)[0]
            print(f"  - Exponential (e^(p*x)): p ≈ {p_exp:.2f}")
        except:
            print("  - Exponential scaling solution not found.")
    
    print("="*50)

    # 5. NDCG Stability Analysis (p-value sweep on scores)
    print("\n[NDCG Sweep Analysis (p-value, 0.1 steps)]")
    p_sweep_values = np.arange(0.1, 2.1, 0.1)
    sweep_results = []
    
    # Get test ground truth
    test_gt = loader_mse.test_df.groupby('user_id')['item_id'].apply(list).to_dict()
    test_users = sorted(list(test_gt.keys()))
    
    # Mask history for test ranking (train + valid)
    eval_mask = loader_mse.eval_user_history
    
    k_val = 10
    for p in p_sweep_values:
        subset_ndcgs = []
        with torch.no_grad():
            for u_id in tqdm(test_users, desc=f"Score Sweep p={p:.1f}", leave=False):
                u_tensor = torch.LongTensor([u_id]).to(device)
                scores = model_mse.forward(u_tensor).cpu().numpy().flatten()
                
                # Apply score power transform
                # x^p is monotonic for p > 0, so rank shouldn't change
                if p == 1.0:
                    scores_p = scores
                else:
                    scores_p = np.sign(scores) * np.power(np.abs(scores), p)
                
                # Mask training + validation items
                mask_items = list(eval_mask.get(u_id, set()))
                scores_p[mask_items] = -np.inf
                
                # Ground truth (from test)
                gt = test_gt.get(u_id, [])
                if not gt: continue
                
                # Rank
                rank_indices = np.argsort(-scores_p)[:k_val]
                
                # NDCG
                dcg = 0.0
                for i, item in enumerate(rank_indices):
                    if item in gt:
                        dcg += 1.0 / np.log2(i + 2)
                
                idcg = 0.0
                for i in range(min(len(gt), k_val)):
                    idcg += 1.0 / np.log2(i + 2)
                
                ndcg = dcg / idcg if idcg > 0 else 0.0
                subset_ndcgs.append(ndcg)
        
        avg_ndcg = np.mean(subset_ndcgs)
        print(f"  - Score p = {p:.1f} | NDCG@{k_val} = {avg_ndcg:.4f}")
        sweep_results.append({'p': float(p), 'ndcg_at_10': float(avg_ndcg)})

    # Export Sweep Results
    SWEEP_FILE = os.path.join(OUTPUT_DIR, "ndcg_sweep_p.json")
    with open(SWEEP_FILE, 'w') as f:
        json.dump(sweep_results, f, indent=4)
    print(f"\nNDCG sweep results saved to {SWEEP_FILE}")
    print("="*50)

    # 6. Embedding Power Sweep Analysis (User Request)
    print("\n[Embedding Power Sweep Analysis (p-value, 0.1 steps)]")
    p_sweep_values = np.arange(0.1, 2.1, 0.1) 
    emb_sweep_results = []
    
    k_val = 10
    for p in p_sweep_values:
        subset_ndcgs = []
        with torch.no_grad():
            for u_id in tqdm(test_users, desc=f"Emb Sweep p={p:.1f}", leave=False):
                u_tensor = torch.LongTensor([u_id]).to(device)
                
                # Manual Dot Product with Transform
                u_emb = model_mse.user_embedding(u_tensor)
                i_emb = model_mse.item_embedding.weight
                
                # Apply Power Transform to Embeddings: sign(E) * |E|^p
                u_emb_p = torch.sign(u_emb) * torch.pow(torch.abs(u_emb), p)
                i_emb_p = torch.sign(i_emb) * torch.pow(torch.abs(i_emb), p)
                
                scores = torch.matmul(u_emb_p, i_emb_p.t()).cpu().numpy().flatten()
                
                # Mask training + validation items
                mask_items = list(eval_mask.get(u_id, set()))
                scores[mask_items] = -np.inf
                
                # Ground truth (from test)
                gt = test_gt.get(u_id, [])
                if not gt: continue
                
                # Rank
                rank_indices = np.argsort(-scores)[:k_val]
                
                # NDCG
                dcg = 0.0
                for i, item in enumerate(rank_indices):
                    if item in gt:
                        dcg += 1.0 / np.log2(i + 2)
                
                idcg = 0.0
                for i in range(min(len(gt), k_val)):
                    idcg += 1.0 / np.log2(i + 2)
                
                ndcg = dcg / idcg if idcg > 0 else 0.0
                subset_ndcgs.append(ndcg)
        
        avg_ndcg = np.mean(subset_ndcgs)
        print(f"  - Embedding p = {p:.1f} | NDCG@{k_val} = {avg_ndcg:.4f}")
        emb_sweep_results.append({'p': float(p), 'ndcg_at_10': float(avg_ndcg)})

    # Export Embedding Sweep Results
    EMB_SWEEP_FILE = os.path.join(OUTPUT_DIR, "ndcg_sweep_embedding_p.json")
    with open(EMB_SWEEP_FILE, 'w') as f:
        json.dump(emb_sweep_results, f, indent=4)
    print(f"\nEmbedding sweep results saved to {EMB_SWEEP_FILE}")
    print("="*50)

if __name__ == "__main__":
    main()
