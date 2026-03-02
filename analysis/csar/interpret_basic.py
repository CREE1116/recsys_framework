import os
import argparse
import re
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.stats import chi2_contingency, pearsonr
from tqdm import tqdm

# 프로젝트 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models import get_model
from src.data_loader import DataLoader as RecSysDataLoader
from analysis.utils import load_item_metadata, get_analysis_output_path, AnalysisReport

def load_model_and_data(run_folder_path):
    """실험 폴더에서 모델과 데이터를 로드합니다."""
    # config.yaml 로드 (실제 경로는 실험 폴더 내의 config.yaml 활용)
    # 여기서는 간단하게 checkpoint 로드 로직을 구현하거나 기존 utils 활용
    # 기존 detail_interests.py의 load_model_from_run 기능을 활용한다고 가정
    from analysis.utils import load_model_from_run
    return load_model_from_run(run_folder_path)

def analyze_interest_correlations(interest_keys, num_interests):
    """관심사 간의 상관관계를 분석합니다."""
    # interest_keys: [K, D]
    keys_norm = torch.nn.functional.normalize(interest_keys, p=2, dim=1)
    corr_matrix = torch.matmul(keys_norm, keys_norm.t()).cpu().numpy()
    return corr_matrix

def extract_decade(title):
    """Extracts decade from title string like 'Toy Story (1995)'"""
    match = re.search(r'\((\d{4})\)', title)
    if match:
        year = int(match.group(1))
        return (year // 10) * 10
    return None

def run_enrichment_analysis(interest_items, metadata_df, global_counts, total_items, feature_name='genres', p_threshold=0.05):
    """특정 관심사에 모인 아이템들과 전체 아이템 간의 특징 분포 차이를 분석합니다."""
    interest_features = []
    for item_id in interest_items:
        if item_id in metadata_df.index:
            val = metadata_df.loc[item_id, feature_name]
            if feature_name == 'genres':
                if isinstance(val, str): interest_features.extend(val.split('|'))
                elif isinstance(val, list): interest_features.extend(val)
            else: # decade 등 단일 값
                if val is not None: interest_features.append(val)
                
    interest_feat_counts = Counter(interest_features)
    results = []
    
    unique_features = list(global_counts.keys())
    for feat in unique_features:
        count_in = interest_feat_counts.get(feat, 0)
        if count_in == 0: continue
            
        count_out = len(interest_items) - count_in
        global_count = global_counts[feat]
        global_out = total_items - global_count
        
        # 2x2 Contingency Table
        table = [[count_in, count_out], [global_count - count_in, global_out - count_out]]
        
        try:
            # 클러스터 크기가 작을 때는 Yates correction 적용 (default)
            chi2, p, dof, ex = chi2_contingency(table)
            ratio_in = count_in / len(interest_items)
            ratio_global = global_count / total_items
            lift = ratio_in / ratio_global if ratio_global > 0 else 0
            
            # 의미 있는 특징 포착을 위해 p-value 임계값 완화 및 Lift 기준 적용
            if (p < p_threshold or (lift > 2.0 and count_in >= 2)) and ratio_in > ratio_global:
                results.append({
                    'Feature': feat,
                    'Count': count_in,
                    'Ratio': f"{ratio_in:.1%}",
                    'Lift': f"{lift:.1f}x",
                    'p-value': f"{p:.4f}"
                })
        except:
            continue
            
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values('p-value')

def interpret_interests(run_folder_path, top_n_items=30):
    """핵심 해석 프로세스"""
    model, data_loader, config = load_model_and_data(run_folder_path)
    if not model:
        print("Model load failed.")
        return

    run_name = os.path.basename(run_folder_path)
    output_path = get_analysis_output_path(config['dataset_name'], run_name)
    report = AnalysisReport(f"Interest Interpretation Report: {run_name}", output_path)

    # 1. 원본 메타데이터 로드 및 가공
    metadata_df = load_item_metadata(config['dataset_name'], config['data_path'])
    metadata_df['decade'] = metadata_df['title'].apply(extract_decade)
    inv_item_map = {v: k for k, v in data_loader.item_map.items()}
    
    # 2. 관심사 키 및 상관관계
    interest_keys = model.get_interest_keys() # [K, D]
    num_interests = interest_keys.shape[0]
    corr_matrix = analyze_interest_correlations(interest_keys, num_interests)
    
    # 3. 전역 통계 (장르 & 시대)
    global_genres = []
    global_decades = []
    for idx, row in metadata_df.iterrows():
        gs = row['genres']
        if isinstance(gs, str): global_genres.extend(gs.split('|'))
        global_decades.append(row['decade'])
        
    global_genre_counts = Counter(global_genres)
    global_decade_counts = Counter([d for d in global_decades if d is not None])
    total_items = len(metadata_df)

    # 4. 멤버십 기반 아이템 분류
    with torch.no_grad():
        item_emb = model.item_embedding.weight
        m_i = model.model_layer.get_membership(item_emb) # [N, K]
        # Soft Assignment: 상위 10% 멤버십을 가진 아이템들을 해당 관심사로 간주
        thresholds = torch.quantile(m_i, 0.9, dim=0) # [K]
        is_member = (m_i >= thresholds).cpu().numpy() # [N, K] bool

    # --- 리포트 작성 ---
    report.add_section("1. Interest-Interest Correlation", level=2)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='RdBu_r', center=0, annot=False)
    plt.title("Interest Similarity (Cosine)")
    plot_file = "interest_corr.png"
    plt.savefig(os.path.join(output_path, plot_file))
    plt.close()
    report.add_figure(plot_file, "Interest key cosine similarity heatmap.")

    report.add_text("Top Related Interest Pairs:")
    pairs = []
    for i in range(num_interests):
        for j in range(i+1, num_interests):
            pairs.append((i, j, corr_matrix[i, j]))
    pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:10]
    pair_df = pd.DataFrame(pairs, columns=['Int A', 'Int B', 'Sim'])
    report.add_table(pair_df)

    report.add_section("2. Per-Interest Deep Dive (Enrichment Analysis)", level=2)
    
    for k in tqdm(range(num_interests), desc="Analyzing interests"):
        item_indices = np.where(is_member[:, k])[0]
        if len(item_indices) < 3:
            continue
            
        report.add_section(f"Interest #{k} (Member Items: {len(item_indices)})", level=3)
        
        # [A] Top Items (Highest Membership)
        k_weights = m_i[:, k].cpu().numpy()
        top_k_indices = item_indices[np.argsort(k_weights[item_indices])[::-1][:10]]
        
        top_items = []
        for idx in top_k_indices:
            orig_id = inv_item_map[idx]
            title = metadata_df.loc[orig_id, 'title'] if orig_id in metadata_df.index else "Unknown"
            top_items.append({'Title': title, 'Weight': f"{k_weights[idx]:.4f}"})
        
        report.add_text("**Top Items (Representative):**")
        report.add_table(pd.DataFrame(top_items))
        
        # [B] Enrichment Analysis (Genre & Decade)
        original_item_ids = [inv_item_map[idx] for idx in item_indices if inv_item_map[idx] in metadata_df.index]
        
        genre_df = run_enrichment_analysis(original_item_ids, metadata_df, global_genre_counts, total_items, 'genres', p_threshold=0.1)
        decade_df = run_enrichment_analysis(original_item_ids, metadata_df, global_decade_counts, total_items, 'decade', p_threshold=0.1)
        
        if not genre_df.empty:
            report.add_text("**Significant Genres:**")
            report.add_table(genre_df.head(5))
            
        if not decade_df.empty:
            report.add_text("**Temporal Characteristics (Decades):**")
            # 1990.0 -> 1990s 로 보기 좋게 변형
            decade_df['Feature'] = decade_df['Feature'].apply(lambda x: f"{int(x)}s")
            report.add_table(decade_df.head(5))
            
        if genre_df.empty and decade_df.empty:
            report.add_text("*No distinctive genre or decade concentration found.*")

    report.save("interest_interpretation.md")
    print(f"Interpretation complete. Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the trained model directory")
    args = parser.parse_args()
    
    interpret_interests(args.run_dir)
