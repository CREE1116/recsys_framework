"""
모델의 인기도 편향(Model Bias) 및 Head/Tail 성능 분석 스크립트.
"""
import sys
from pathlib import Path

# 프로젝트 루트 추가
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm
from scipy.stats import spearmanr
from collections import Counter

from analysis.utils import load_trained_model, get_item_popularity, split_items_by_popularity
from analysis.dataset.analyze_data_bias import calculate_gini_coefficient

def analyze_model_bias(exp_dir, device='cpu'):
    print(f"\nAnalyzing model in {exp_dir}...")
    try:
        model, data_loader, config = load_trained_model(exp_dir, device)
    except Exception as e:
        print(f"Skipping {exp_dir}: {e}")
        return None

    # 1. Prepare Item Metrics
    # Popularity (Train set interaction count)
    item_pop = get_item_popularity(data_loader)
    pop_values = item_pop.values
    log_pop = np.log1p(pop_values)
    
    # Item Groups (Head/Tail)
    groups = split_items_by_popularity(item_pop, ratios={'head': 0.2, 'tail': 0.8})
    head_items = set(groups['head'])
    tail_items = set(groups['tail'])

    # 2. Embedding Analysis
    if hasattr(model, 'item_embedding'):
        # Get Item Embeddings
        if isinstance(model.item_embedding, torch.nn.Embedding):
            item_embs = model.item_embedding.weight.detach().cpu().numpy()
        else:
            # LightGCN 등은 forward로 얻어야 할 수도 있음 (여기선 weight 가정)
            # 만약 weight가 없다면 get_final_item_embeddings() 등의 메서드 체크
            if hasattr(model, 'get_final_item_embeddings'):
                item_embs = model.get_final_item_embeddings().detach().cpu().numpy()
            else:
                 item_embs = None
        
        emb_stats = {}
        if item_embs is not None:
            # L2 Norm
            norms = np.linalg.norm(item_embs, axis=1)
            
            # Correlation (Log Popularity vs Norm)
            corr, _ = spearmanr(log_pop, norms)
            
            emb_stats['norm_pop_correlation'] = float(corr)
            emb_stats['avg_norm_head'] = float(norms[list(head_items)].mean())
            emb_stats['avg_norm_tail'] = float(norms[list(tail_items)].mean())
            emb_stats['norm_ratio_head_tail'] = emb_stats['avg_norm_head'] / (emb_stats['avg_norm_tail'] + 1e-8)
    else:
        emb_stats = {}

    # 3. Recommendation Analysis (Performance & Bias)
    # Test User들에 대해 Top-K 추천
    # (일부 유저만 샘플링하여 빠르게 분석)
    num_test_users = min(2000, data_loader.n_users) # 최대 2000명만
    test_users = np.random.choice(data_loader.n_users, num_test_users, replace=False)
    
    # Ground Truth (Test Set)
    # data_loader.test_df 사용
    test_df = data_loader.test_df
    test_df = test_df[test_df['user_id'].isin(test_users)]
    
    # Model Inference
    k = 20
    recommendations = [] # List of list of item_ids
    
    batch_size = 100 # Inference Batch
    model.eval()
    
    all_rec_items = []
    
    # 세그먼트별 성능을 위한 카운터
    hits_head = 0
    hits_tail = 0
    total_head_targets = 0
    total_tail_targets = 0
    
    ndcg_sum = 0
    recall_sum = 0
    
    with torch.no_grad():
        for i in range(0, len(test_users), batch_size):
            batch_users = torch.LongTensor(test_users[i:i+batch_size]).to(device)
            # Scores: [B, N_items]
            scores = model.forward(batch_users)
            # Masking Train Items (Optional but recommended for strict eval)
            # 여기서는 편향 분석이 주 목적이므로 생략하거나, data_loader 설정을 따름
            # RecSysFramework는 evaluation loop에서 마스킹을 하므로, 
            # 여기서는 raw score 기반 top-k만 봅니다 (간단 분석)
            
            _, indices = torch.topk(scores, k=k)
            batch_recs = indices.cpu().numpy()
            
            for u_idx, rec_items in enumerate(batch_recs):
                u_id = batch_users[u_idx].item()
                recommendations.append(rec_items)
                all_rec_items.extend(rec_items)
                
                # Ground Truth for this user (Test Set)
                gt_items = test_df[test_df['user_id'] == u_id]['item_id'].values
                if len(gt_items) == 0: continue
                
                # --- NDCG Calculation ---
                dcg = 0.0
                idcg = 0.0
                
                # IDCG (Ideal)
                num_pos = len(gt_items)
                for i in range(min(num_pos, k)):
                    idcg += 1.0 / np.log2(i + 2)
                    
                # DCG (Actual)
                for i, r_item in enumerate(rec_items):
                    if r_item in gt_items:
                        dcg += 1.0 / np.log2(i + 2)
                
                ndcg_sum += (dcg / idcg) if idcg > 0 else 0
    
    avg_ndcg = ndcg_sum / len(test_users)

    # Bias Metrics
    rec_counts = Counter(all_rec_items)
    rec_pop_values = np.array([item_pop.get(i, 0) for i in all_rec_items])
    
    # 1. APR (Avg Popularity of Recommendations)
    apr = rec_pop_values.mean()
    
    # 2. Coverage
    unique_rec_items = set(all_rec_items)
    coverage = len(unique_rec_items) / data_loader.n_items
    
    # 3. Head/Tail Coverage (Absolute & Ratio)
    # Tail Coverage: How many tail items are covered? / Total tail items
    tail_rec_items = unique_rec_items.intersection(tail_items)
    tail_coverage = len(tail_rec_items) / len(tail_items) if len(tail_items) > 0 else 0
    
    # 4. Gini of Recommendations
    rec_count_distribution = np.zeros(data_loader.n_items)
    for i, c in rec_counts.items():
        rec_count_distribution[i] = c
    rec_gini = calculate_gini_coefficient(rec_count_distribution)

    rec_stats = {
        'NDCG': float(avg_ndcg),
        'APR': float(apr),
        'Coverage': float(coverage),
        'Tail_Coverage': float(tail_coverage),
        'Rec_Gini': float(rec_gini),
        'Recall_Head': hits_head / total_head_targets if total_head_targets > 0 else 0,
        'Recall_Tail': hits_tail / total_tail_targets if total_tail_targets > 0 else 0,
    }
    
    return {
        'model_name': config['model']['name'],
        'run_name': config.get('run_name', 'default'),
        'embedding_bias': emb_stats,
        'recommendation_bias': rec_stats
    }

def simulate_tradeoff_curve(data_loader, item_pop, head_items, tail_items, k=20, num_samples=500):
    """
    Generate Ideal Performance (Pareto Frontier) by controlling Head/Tail ratio.
    """
    sorted_items = item_pop.sort_values(ascending=False).index.values
    test_users = data_loader.test_df['user_id'].unique()
    if len(test_users) > num_samples:
        np.random.seed(42)
        test_users = np.random.choice(test_users, num_samples, replace=False)
    
    # Pre-compute GT for sample users
    user_gt = {}
    test_df_subset = data_loader.test_df[data_loader.test_df['user_id'].isin(test_users)]
    for u, group in test_df_subset.groupby('user_id'):
        user_gt[u] = set(group['item_id'].values)

    results = []
    # Target Tail Ratios in recommendation list
    target_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    for ratio in target_ratios:
        n_tail = int(k * ratio)
        n_head = k - n_tail
        
        all_rec_items = []
        ndcg_sum = 0
        
        for u in test_users:
            gt = user_gt.get(u, set())
            
            # 1. Select Head Items (Prioritize GT)
            gt_head = [i for i in gt if i in head_items]
            rec_head = gt_head[:n_head]
            # Fill remaining with popular head items
            if len(rec_head) < n_head:
                for i in sorted_items:
                    if len(rec_head) >= n_head: break
                    if i in head_items and i not in rec_head:
                        rec_head.append(i)
            
            # 2. Select Tail Items (Prioritize GT)
            gt_tail = [i for i in gt if i in tail_items]
            rec_tail = gt_tail[:n_tail]
            # Fill remaining with popular tail items
            # (Sort tail items by pop if needed, here assuming sorted_items order)
            if len(rec_tail) < n_tail:
                for i in sorted_items:
                    if len(rec_tail) >= n_tail: break
                    if i in tail_items and i not in rec_tail:
                        rec_tail.append(i)
            
            recs = rec_head + rec_tail
            all_rec_items.extend(recs)
            
            # Calculate NDCG
            dcg = 0.0
            idcg = 0.0
            num_pos = len(gt)
            for i in range(min(num_pos, k)):
                idcg += 1.0 / np.log2(i + 2)
            
            for i, r_item in enumerate(recs):
                if r_item in gt:
                    dcg += 1.0 / np.log2(i + 2)
            
            if idcg > 0:
                ndcg_sum += dcg / idcg
        
        avg_ndcg = ndcg_sum / len(test_users)
        
        # Calculate Coverage
        unique_recs = set(all_rec_items)
        tail_recs = unique_recs.intersection(tail_items)
        tail_cov = len(tail_recs) / len(tail_items) if len(tail_items) > 0 else 0
        
        results.append({
            'Target_Tail_Ratio': ratio,
            'NDCG': avg_ndcg,
            'Tail_Coverage': tail_cov
        })
        
    return pd.DataFrame(results)

def analyze_all_models(dataset_path, device='cpu'):
    dataset_path = Path(dataset_path)
    dataset_name = dataset_path.name
    
    results = []
    
    # 0. Load Dataset & Simulate Pareto Frontier (using the first valid config found)
    sim_df = None
    data_loader = None
    
    # Find all model directories
    for model_dir in dataset_path.iterdir():
        if not model_dir.is_dir(): continue
        # Check if it has config.yaml
        if (model_dir / 'config.yaml').exists() and (model_dir / 'best_model.pt').exists():
            print(f"Found model: {model_dir.name}")
            
            # Load model to get stats
            stats = analyze_model_bias(model_dir, device)
            if stats:
                results.append(stats)
                
                # Run Simulation once
                if sim_df is None:
                    print("Simulating Theoretical Trade-off Curve (Pareto Frontier)...")
                    # Need to construct data_loader from this config to get item_pop etc.
                    try:
                        _, dl, _ = load_trained_model(model_dir, device)
                        item_pop = get_item_popularity(dl)
                        groups = split_items_by_popularity(item_pop)
                        sim_df = simulate_tradeoff_curve(dl, item_pop, set(groups['head']), set(groups['tail']))
                    except Exception as e:
                        print(f"Simulation failed: {e}")

    if not results:
        print("No valid models found.")
        return

    # Aggregate & Save
    output_dir = Path("output") / dataset_name / "model_bias_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compare Table
    compare_data = []
    for r in results:
        row = {
            'Model': r['model_name'],
            'Run': r['run_name'],
            'NDCG': r['recommendation_bias']['NDCG'],
            'Tail_Coverage': r['recommendation_bias']['Tail_Coverage'],
            'Coverage': r['recommendation_bias']['Coverage'],
            'APR': r['recommendation_bias']['APR'],
            'Gini': r['recommendation_bias']['Rec_Gini'],
        }
        if 'norm_pop_correlation' in r['embedding_bias']:
             row['Emb_Pop_Corr'] = r['embedding_bias']['norm_pop_correlation']
        compare_data.append(row)
        
    df_compare = pd.DataFrame(compare_data)
    df_compare.to_csv(output_dir / "bias_comparison.csv", index=False)
    print(f"\nSaved comparison to {output_dir / 'bias_comparison.csv'}")
    
    # Visualization: Trade-off Plot (Tail Coverage vs NDCG)
    plt.figure(figsize=(10, 8))
    
    # 1. Models
    sns.scatterplot(data=df_compare, x='Tail_Coverage', y='NDCG', hue='Model', style='Model', s=150, zorder=10)
    
    # 2. Pareto Frontier (Simulation)
    if sim_df is not None:
        plt.plot(sim_df['Tail_Coverage'], sim_df['NDCG'], 'r--', label='Ideal Pareto Frontier (Simulated)', alpha=0.7)
        plt.scatter(sim_df['Tail_Coverage'], sim_df['NDCG'], c='red', s=30, alpha=0.7)

    # Add labels
    for i in range(df_compare.shape[0]):
        plt.text(
            df_compare['Tail_Coverage'][i], 
            df_compare['NDCG'][i]+0.001, 
            df_compare['Run'][i], 
            fontsize=9, 
            alpha=0.8
        )
        
    plt.title(f"NDCG vs Long-tail Coverage Trade-off ({dataset_name})")
    plt.xlabel("Long-tail Coverage (Reflects Item Diversity)")
    plt.ylabel("NDCG@20 (Overall Performance)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "ndcg_vs_tail_coverage.png", dpi=300)
    plt.close()
    
    print("Saved visualizations.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str, help='Path to trained dataset models (e.g., trained_model/ml-100k)')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    analyze_all_models(args.dataset_path, args.device)
