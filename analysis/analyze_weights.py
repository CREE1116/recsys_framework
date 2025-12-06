import argparse
import os
import yaml
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# 프로젝트 루트 경로를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DataLoader
from src.models import get_model

def analyze_csar_interests(experiment_dir, output_file=None):
    """
    학습된 CSAR 계열 모델의 Interest Key 간의 유사도를 분석하고 히트맵으로 시각화합니다.
    """
    print(f"Starting CSAR interest analysis for experiment: {experiment_dir}")

    # 1. 설정 파일 및 모델 체크포인트 경로 확인
    config_path = os.path.join(experiment_dir, 'config.yaml')
    model_path = os.path.join(experiment_dir, 'best_model.pt')

    if not os.path.exists(config_path) or not os.path.exists(model_path):
        print(f"Error: 'config.yaml' or 'best_model.pt' not found in {experiment_dir}")
        return

    # 2. 설정 로드
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_name = config['model']['name']
    if 'csar' not in model_name.lower():
        print(f"Error: This analysis script is designed for CSAR models, but found '{model_name}'.")
        return

    # 3. 데이터 로더 및 모델 재구성
    data_loader = DataLoader(config)
    model = get_model(model_name, config, data_loader)
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully.")

    # 4. Interest Key (self.C) 임베딩 추출
    # 4. Interest Key 추출
    # 모델 구조에 따라 저장 위치가 다를 수 있음
    if hasattr(model, 'attention_layer') and hasattr(model.attention_layer, 'interest_keys'):
        interest_keys_tensor = model.attention_layer.interest_keys
    elif hasattr(model, 'C') and hasattr(model.C, 'weight'):
        interest_keys_tensor = model.C.weight
    else:
        print("Error: Could not find interest keys (checked 'attention_layer.interest_keys' and 'C.weight').")
        return
        
    interest_keys = interest_keys_tensor.detach().cpu().numpy()
    n_keys, dim = interest_keys.shape
    print(f"Extracted {n_keys} interest keys with dimension {dim}.")

    # 5. Interest Key 간의 코사인 유사도 계산
    # 정규화
    norm = np.linalg.norm(interest_keys, axis=1, keepdims=True)
    normalized_keys = interest_keys / norm
    # 유사도 행렬 계산
    similarity_matrix = np.dot(normalized_keys, normalized_keys.T)

    # 6. 히트맵 시각화 및 저장
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap='viridis', annot=False)
    plt.title(f'Cosine Similarity of Interest Keys\n(Experiment: {os.path.basename(experiment_dir)})')
    plt.xlabel('Interest Key Index')
    plt.ylabel('Interest Key Index')
    
    if output_file is None:
        output_file = os.path.join(experiment_dir, 'interest_key_similarity.png')
        
    plt.savefig(output_file)
    print(f"Analysis saved to {output_file}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze interest keys from a trained CSAR model.")
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='Path to the experiment directory (containing config.yaml and best_model.pt).')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the output visualization PNG file. Defaults to saving in the experiment directory.')
    
    args = parser.parse_args()
    
    analyze_csar_interests(args.exp_dir, output_file=args.output)
