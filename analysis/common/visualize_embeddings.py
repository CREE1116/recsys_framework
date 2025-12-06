import argparse
import os
import yaml
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sys

# 프로젝트 루트 경로를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_loader import DataLoader
from src.models import get_model

def visualize_embeddings(experiment_dir, embedding_type='final', output_file=None):
    """
    학습된 모델의 아이템 임베딩을 t-SNE를 사용하여 시각화합니다.
    'base' 또는 'final' 임베딩 타입을 선택할 수 있습니다.
    """
    print(f"Starting embedding visualization for experiment: {experiment_dir}")
    print(f"Embedding type: {embedding_type}")

    # 1. 설정 파일 및 모델 체크포인트 경로 확인
    config_path = os.path.join(experiment_dir, 'config.yaml')
    model_path = os.path.join(experiment_dir, 'best_model.pt')

    if not os.path.exists(config_path) or not os.path.exists(model_path):
        print(f"Error: 'config.yaml' or 'best_model.pt' not found in {experiment_dir}")
        return

    # 2. 설정 로드
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 3. 데이터 로더 및 모델 재구성
    data_loader = DataLoader(config)
    model = get_model(config['model']['name'], config, data_loader)
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully.")

    # 4. 아이템 임베딩 추출
    with torch.no_grad():
        if embedding_type == 'final':
            print("Extracting final (transformed) item embeddings...")
            item_embeddings = model.get_final_item_embeddings()
        else: # 'base'
            print("Extracting base item embeddings...")
            if hasattr(model, 'item_embedding'):
                item_embeddings = model.item_embedding.weight
            elif hasattr(model, 'base_model') and hasattr(model.base_model, 'item_embedding'): # For wrappers
                 item_embeddings = model.base_model.item_embedding.weight
            else:
                print("Error: Could not find base 'item_embedding' attribute in the model.")
                return
            
    item_embeddings = item_embeddings.cpu().detach().numpy()
    n_items = item_embeddings.shape[0]
    print(f"Extracted {n_items} item embeddings with dimension {item_embeddings.shape[1]}.")

    # 5. t-SNE 수행 (전체 데이터 사용)
    print(f"Performing t-SNE on {n_items} items. This may be slow for large datasets...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(item_embeddings)
    print("t-SNE transformation complete.")

    # 6. 시각화 및 저장
    plt.figure(figsize=(12, 12))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6,s=2)
    title = (f't-SNE Visualization of {embedding_type.capitalize()} Item Embeddings\n'
             f'(Experiment: {os.path.basename(experiment_dir)})')
    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    if output_file is None:
        output_filename = f'item_embedding_visualization_{embedding_type}.png'
        output_file = os.path.join(experiment_dir, output_filename)
        
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")
    plt.close()


if __name__ == '__main__':
    # 분석할 실험 디렉토리 목록
    experiment_dirs_to_visualize = [
        'csar-r-confidence__num_interests=4_scale=True_negative_sampling_strategy=popularity'
        # 여기에 다른 실험 디렉토리 경로를 추가할 수 있습니다.
    ]

    for exp_dir in experiment_dirs_to_visualize:
        if not os.path.isdir(exp_dir):
            print(f"Directory not found, skipping: {exp_dir}")
            continue
        
        # 'base'와 'final' 임베딩 모두에 대해 시각화 실행
        visualize_embeddings(exp_dir, embedding_type='base')
        visualize_embeddings(exp_dir, embedding_type='final')
        print("-" * 60)

    print("All visualizations completed.")
 