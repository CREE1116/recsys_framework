"""
CSAR 관심사 유사도 상세 분석 스크립트

분석 항목:
1. Interest Keys 간 코사인 유사도
2. 각 관심사별 가장 유사한 관심사 Top-N
3. 유사한 관심사 쌍의 대표 콘텐츠 비교
"""
import sys
import os
from pathlib import Path

# 프로젝트 루트 추가
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import yaml

from analysis.utils import load_model_from_run, get_analysis_output_path, AnalysisReport, load_item_metadata


def load_augmented_metadata(data_dir, dataset_name='ml-1m'):
    """메타데이터 로드"""
    if os.path.isfile(data_dir):
        data_dir = os.path.dirname(data_dir)
    
    aug_path = os.path.join(data_dir, 'movies_augmented.csv')
    if os.path.exists(aug_path):
        print(f"Loading augmented metadata from {aug_path}")
        df = pd.read_csv(aug_path)
        df = df.fillna('')
        return df.set_index('item_id')
    
    dummy_path = os.path.join(data_dir, 'dummy_file')
    return load_item_metadata(dataset_name, dummy_path)


def compute_interest_similarity(model):
    """Interest Keys 간 코사인 유사도 계산"""
    keys = model.attention_layer.interest_keys.detach().cpu()
    keys_norm = torch.nn.functional.normalize(keys, p=2, dim=1)
    sim_matrix = (keys_norm @ keys_norm.t()).numpy()
    return sim_matrix


def get_top_items_for_interest(model, data_loader, interest_k, top_n=10):
    """관심사 k의 대표 아이템 추출"""
    model.eval()
    with torch.no_grad():
        item_embs = model.item_embedding.weight
        memberships = model.attention_layer.get_membership(item_embs).cpu()
    
    weights = memberships[:, interest_k]
    top_weights, top_indices = torch.topk(weights, min(top_n, len(weights)))
    
    return top_indices.numpy(), top_weights.numpy()


def generate_pair_report(report, k1, k2, similarity, 
                         items1, weights1, items2, weights2,
                         inv_item_map, metadata_df):
    """관심사 쌍 리포트 생성"""
    report.add_section(f"Interest {k1} ↔ {k2} (sim: {similarity:.3f})", level=3)
    
    # Interest k1 대표 아이템
    report.add_text(f"\n**Interest {k1} Top Items:**\n")
    for idx, (item_id, w) in enumerate(zip(items1[:5], weights1[:5])):
        orig_id = inv_item_map.get(item_id)
        if orig_id and orig_id in metadata_df.index:
            row = metadata_df.loc[orig_id]
            title = row.get('title', row.get('movie_title', f'Item {orig_id}'))
            genres = row.get('genres', '')
            report.add_text(f"- {title} ({genres}) [w={w:.2f}]")
        else:
            report.add_text(f"- Item {orig_id} [w={w:.2f}]")
    
    # Interest k2 대표 아이템
    report.add_text(f"\n**Interest {k2} Top Items:**\n")
    for idx, (item_id, w) in enumerate(zip(items2[:5], weights2[:5])):
        orig_id = inv_item_map.get(item_id)
        if orig_id and orig_id in metadata_df.index:
            row = metadata_df.loc[orig_id]
            title = row.get('title', row.get('movie_title', f'Item {orig_id}'))
            genres = row.get('genres', '')
            report.add_text(f"- {title} ({genres}) [w={w:.2f}]")
        else:
            report.add_text(f"- Item {orig_id} [w={w:.2f}]")


def find_top_similar_pairs(sim_matrix, top_n=10):
    """가장 유사한 관심사 쌍 찾기 (중복 제거)"""
    K = sim_matrix.shape[0]
    pairs = []
    
    for i in range(K):
        for j in range(i+1, K):
            pairs.append((i, j, sim_matrix[i, j]))
    
    # 유사도 내림차순 정렬
    pairs.sort(key=lambda x: -x[2])
    return pairs[:top_n]


def find_most_distant_pairs(sim_matrix, top_n=5):
    """가장 다른 관심사 쌍 찾기"""
    K = sim_matrix.shape[0]
    pairs = []
    
    for i in range(K):
        for j in range(i+1, K):
            pairs.append((i, j, sim_matrix[i, j]))
    
    pairs.sort(key=lambda x: x[2])
    return pairs[:top_n]


def visualize_similarity(sim_matrix, save_path):
    """유사도 히트맵 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    sns.heatmap(sim_matrix, ax=axes[0], cmap='coolwarm', center=0, 
                vmin=-1, vmax=1, annot=False)
    axes[0].set_title('Interest Cosine Similarity Matrix')
    axes[0].set_xlabel('Interest')
    axes[0].set_ylabel('Interest')
    
    K = sim_matrix.shape[0]
    off_diag = sim_matrix[~np.eye(K, dtype=bool)]
    axes[1].hist(off_diag, bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[1].axvline(x=off_diag.mean(), color='blue', linestyle='-', 
                     label=f'Mean: {off_diag.mean():.3f}')
    axes[1].set_xlabel('Cosine Similarity')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Off-diagonal Similarity Distribution')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# 기본 설정 (수정 가능)
# ============================================================
EXP_PATH = 'trained_model/ml-1m-10c/csar-rec2'
DEVICE = 'cpu'
TOP_PAIRS = 10
TOP_ITEMS = 5


def main():
    exp_path = EXP_PATH
    top_pairs = TOP_PAIRS
    top_items = TOP_ITEMS
    
    print(f"Loading model from {exp_path}...")
    model, data_loader, config = load_model_from_run(exp_path)
    
    # 메타데이터 로드
    data_dir = os.path.dirname(config['data_path'])
    metadata_df = load_augmented_metadata(data_dir, config['dataset_name'])
    inv_item_map = {v: k for k, v in data_loader.item_map.items()}
    
    # 출력 경로
    run_name = os.path.basename(exp_path)
    output_path = Path(get_analysis_output_path(config['dataset_name'], run_name))
    report = AnalysisReport("Interest Similarity Deep Analysis", output_path)
    
    print("\n" + "=" * 60)
    print("Interest Similarity Deep Analysis")
    print("=" * 60)
    
    # 유사도 계산
    print("\n[1] Computing similarity matrix...")
    sim_matrix = compute_interest_similarity(model)
    K = sim_matrix.shape[0]
    off_diag = sim_matrix[~np.eye(K, dtype=bool)]
    
    report.add_section("Summary", level=2)
    report.add_text(f"- **Number of Interests**: {K}")
    report.add_text(f"- **Mean Similarity**: {off_diag.mean():.4f}")
    report.add_text(f"- **Max Similarity**: {off_diag.max():.4f}")
    report.add_text(f"- **Min Similarity**: {off_diag.min():.4f}")
    
    # 가장 유사한 쌍 찾기
    print(f"\n[2] Finding top-{top_pairs} similar pairs...")
    similar_pairs = find_top_similar_pairs(sim_matrix, top_pairs)
    
    report.add_section("Most Similar Interest Pairs", level=2)
    
    for k1, k2, sim in similar_pairs:
        print(f"  Interest {k1} <-> {k2}: {sim:.4f}")
        
        # 각 관심사의 대표 아이템
        items1, weights1 = get_top_items_for_interest(model, data_loader, k1, top_items)
        items2, weights2 = get_top_items_for_interest(model, data_loader, k2, top_items)
        
        generate_pair_report(report, k1, k2, sim, 
                            items1, weights1, items2, weights2,
                            inv_item_map, metadata_df)
    
    # 가장 다른 쌍
    print(f"\n[3] Finding most distant pairs...")
    distant_pairs = find_most_distant_pairs(sim_matrix, 5)
    
    report.add_section("Most Distant Interest Pairs", level=2)
    
    for k1, k2, sim in distant_pairs:
        print(f"  Interest {k1} <-> {k2}: {sim:.4f}")
        
        items1, weights1 = get_top_items_for_interest(model, data_loader, k1, top_items)
        items2, weights2 = get_top_items_for_interest(model, data_loader, k2, top_items)
        
        generate_pair_report(report, k1, k2, sim,
                            items1, weights1, items2, weights2,
                            inv_item_map, metadata_df)
    
    # 시각화
    print("\n[4] Visualization...")
    fig_path = output_path / 'interest_similarity_matrix.png'
    visualize_similarity(sim_matrix, fig_path)
    print(f"Saved: {fig_path}")
    
    # 리포트 저장
    report.save()
    print(f"Saved: {output_path / 'interest_similarity_deep_report.md'}")
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

