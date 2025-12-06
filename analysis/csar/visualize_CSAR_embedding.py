import argparse
import os
import yaml
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys

# 프로젝트 루트 경로를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_loader import DataLoader
from src.models import get_model

def get_item_popularity(data_loader):
    """
    전체 학습 데이터에서 아이템별 등장 횟수(Popularity)를 계산합니다.
    """
    # train_df에 접근 가능하다고 가정 (BaseDataLoader 구조에 따라 다를 수 있음)
    if hasattr(data_loader, 'train_df'):
        df = data_loader.train_df
        # 혹은 data_loader 내부 구조에 맞춰 수정
        pop_counts = df['item_id'].value_counts().sort_index() # item_id가 0~N 인덱싱 되었다고 가정
        
        # 인덱스가 비어있는 아이템(학습에 안나온) 처리
        full_counts = np.zeros(data_loader.n_items)
        full_counts[pop_counts.index] = pop_counts.values
        return full_counts
    else:
        print("Warning: Could not access train_df to calculate popularity. Using random colors.")
        return None

def visualize_embeddings_advanced(experiment_dir, output_file=None):
    """
    향상된 시각화: Final Item Embedding + Interest Keys + Popularity Color
    """
    print(f"Starting Advanced Visualization: {experiment_dir}")

    # 1. 설정 및 모델 로드
    config_path = os.path.join(experiment_dir, 'config.yaml')
    model_path = os.path.join(experiment_dir, 'best_model.pt')

    if not os.path.exists(config_path) or not os.path.exists(model_path):
        print(f"Error: Files not found in {experiment_dir}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_loader = DataLoader(config)
    model = get_model(config['model']['name'], config, data_loader)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # 2. 데이터 추출
    # (1) 아이템 임베딩 (Final Interest Space)
    with torch.no_grad():
        # 모델마다 메서드 명이 다를 수 있어 안전장치 추가
        if hasattr(model, 'item_embedding'):
             # fallback: 기본 임베딩 사용 (CSAR 레이어 통과 전)
            item_embs = model.item_embedding.weight
            print("Warning: Using raw item embeddings (get_final_item_embeddings not found).")
        else:
            print("Error: Cannot extract item embeddings.")
            return
        
        item_embs = item_embs.cpu().numpy()

        # (2) Interest Keys (Cluster Centroids) 추출
        keys = None
        # 모델 구조에 따라 interest_keys 위치 찾기
        if hasattr(model, 'global_interest_keys'): # CSAR_R
            keys = model.global_interest_keys
        elif hasattr(model, 'attention_layer') and hasattr(model.attention_layer, 'interest_keys'): # CSAR_Hybrid
            keys = model.attention_layer.interest_keys
        elif hasattr(model, 'interest_keys'): # CSAR_Hybrid_BPR
            keys = model.interest_keys
        
        if keys is not None:
            keys = keys.cpu().numpy()
            print(f"Found {keys.shape[0]} Interest Keys.")
        else:
            print("Interest Keys not found in model attributes.")

    # 3. 인기도 정보 추출 (색상용)
    # DataLoader 구현체에 따라 train_df 접근 방식이 다를 수 있음. 
    # 여기서는 data_loader가 DataFrame을 가지고 있다고 가정.
    # 만약 없다면, 임의의 값이나 0으로 채움.
    try:
        # 일반적인 DataFrame 기반 DataLoader 가정
        pop_counts = data_loader.train_df['item_id'].value_counts()
        # item_id 매핑이 0부터 n_items-1까지 순차적이라고 가정
        popularities = np.zeros(data_loader.n_items)
        # 인덱스 매핑 확인 필요 (여기서는 raw item_id가 곧 인덱스라고 가정하거나, 매핑 로직 필요)
        # 간단히: 있는 것만 채움
        for iid, count in pop_counts.items():
            if iid < data_loader.n_items:
                popularities[iid] = count
        
        # 로그 스케일 적용 (인기도 분포가 롱테일이므로 로그를 취해야 색상 구분이 잘 됨)
        popularities = np.log1p(popularities)
    except:
        print("Could not calculate popularity from DataLoader. Calculating norm as proxy.")
        # 대안: 벡터의 크기(Norm)를 색상으로 사용 (Norm 이론 검증용)
        popularities = np.linalg.norm(item_embs, axis=1)

    # 4. t-SNE 수행 (Items + Keys 함께 투영)
    # 함께 fit해야 같은 공간에 매핑됨
    if keys is not None:
        combined_data = np.vstack([item_embs, keys])
        n_items = item_embs.shape[0]
        n_keys = keys.shape[0]
    else:
        combined_data = item_embs
        n_items = item_embs.shape[0]
        n_keys = 0

    print(f"Running t-SNE on {combined_data.shape[0]} vectors...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    embedded = tsne.fit_transform(combined_data)

    item_tsne = embedded[:n_items]
    key_tsne = embedded[n_items:] if n_keys > 0 else None

    # 5. 시각화
    plt.figure(figsize=(14, 12))

    # (1) 아이템 산점도 (인기도에 따른 색상)
    # cmap='viridis' (보라:비인기 -> 노랑:인기) or 'magma'
    scatter = plt.scatter(item_tsne[:, 0], item_tsne[:, 1], 
                          c=popularities, cmap='turbo', 
                          s=3, alpha=0.5, label='Items')
    
    # Colorbar 추가
    cbar = plt.colorbar(scatter)
    cbar.set_label('Log Popularity (or Norm)', rotation=270, labelpad=15)

    # (2) Interest Keys (별 모양, 검은 테두리)
    if key_tsne is not None:
        plt.scatter(key_tsne[:, 0], key_tsne[:, 1], 
                    c='white', marker='*', s=300, edgecolors='black', linewidths=1.5, 
                    label='Interest Keys', zorder=10) # zorder로 맨 위에 그림
        
        # 키 번호 매기기
        for i in range(n_keys):
            plt.text(key_tsne[i, 0], key_tsne[i, 1], str(i), fontsize=12, fontweight='bold')

    plt.title(f"Advanced Embedding Visualization\nModel: {config['model']['name']} | K={n_keys}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_file is None:
        output_file = os.path.join(experiment_dir, 'embedding_analysis_plot.png')
    
    plt.savefig(output_file, dpi=300)
    print(f"Saved to {output_file}")
    plt.close()

if __name__ == '__main__':
    # 실험 경로 설정
    target_dir = '/Users/leejongmin/code/recsys_framework/trained_model/amazon_books/csar-bpr-ce__temperature=0.8'
    
    if os.path.isdir(target_dir):
        visualize_embeddings_advanced(target_dir)
    else:
        print("Directory not found.")