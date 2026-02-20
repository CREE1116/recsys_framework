"""
Unified CSAR Layer Analysis Script
모든 CSAR 모델 변형(Rec, Rec2, Rec7, Rec8 등)을 통합 분석합니다.

분석 항목:
1. Interest Keys 직교도 (Orthogonality)
2. Gram Matrix (G) 시각화
3. Interest Keys Norm 분포

사용법:
  python analysis/csar/analyze_csar_layer.py /path/to/trained_model/ml-100k
"""
import sys
from pathlib import Path

# 프로젝트 루트 추가
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import yaml
import pandas as pd


def find_csar_experiments(dataset_path: Path) -> list:
    """데이터셋 폴더 내 모든 CSAR 모델 경로 찾기"""
    exp_dirs = []
    for exp_dir in dataset_path.iterdir():
        if not exp_dir.is_dir():
            continue
        # csar-로 시작하는 폴더 찾기
        if exp_dir.name.startswith('csar'):
            config_path = exp_dir / 'config.yaml'
            model_path = exp_dir / 'best_model.pt'
            if config_path.exists() and model_path.exists():
                exp_dirs.append(exp_dir)
    return sorted(exp_dirs)


def load_model(exp_path: str, device='cpu'):
    """학습된 모델 로드"""
    from src.models import MODEL_REGISTRY
    from src.data_loader import DataLoader
    
    config_path = Path(exp_path) / 'config.yaml'
    model_path = Path(exp_path) / 'best_model.pt'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_loader = DataLoader(config)
    
    model_name = config['model']['name']
    if model_name not in MODEL_REGISTRY:
        print(f"Warning: Model {model_name} not registered. Skipping.")
        return None, None, None
        
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(config, data_loader)
    
    # strict=False for compatibility (some models might have extra/missing buffers)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    except Exception as e:
        print(f"Error loading weights for {model_name}: {e}")
        return None, None, None
        
    model.to(device)
    model.eval()
    
    return model, data_loader, config


def get_interest_keys(model):
    """모델에서 Interest Keys 추출 (자동 감지)"""
    if hasattr(model, 'attention_layer'):
        layer = model.attention_layer
        if hasattr(layer, 'interest_keys'):
            return layer.interest_keys.detach().cpu()
        # elif hasattr(layer, 'encoder') and hasattr(layer.encoder, 'interest_keys'): # For some complex layers
        #     return layer.encoder.interest_keys.detach().cpu()
    
    # Fallback: search all parameters
    for name, param in model.named_parameters():
        if 'interest_keys' in name:
            return param.detach().cpu()
            
    return None


def analyze_keys(keys, exp_name):
    """Interest Keys 통계 분석"""
    K, d = keys.shape
    
    # L2 Norm
    norms = torch.norm(keys, p=2, dim=1)
    
    # Normalized Gram Matrix (Cosine Similarity)
    keys_norm = torch.nn.functional.normalize(keys, p=2, dim=1)
    gram_norm = keys_norm @ keys_norm.t()
    
    # Raw Gram Matrix
    raw_gram = keys @ keys.t()
    
    # Off-diagonal 분석
    off_diag_mask = 1 - torch.eye(K)
    off_diag = gram_norm * off_diag_mask
    
    # 직교도 (1 - 평균 코사인 유사도)
    orthogonality = 1 - off_diag.abs().mean().item()
    
    stats = {
        'exp_name': exp_name,
        'K': K,
        'd': d,
        'orthogonality': orthogonality,
        'avg_cosine_sim': off_diag.abs().mean().item(),
        'max_cosine_sim': off_diag.abs().max().item(),
        'norm_mean': norms.mean().item(),
        'norm_std': norms.std().item(),
        'sparsity_0.1': (off_diag.abs() < 0.1).float().mean().item(),
    }
    
    return stats, gram_norm.numpy(), raw_gram.numpy()


def save_heatmap(matrix, save_path, title):
    """히트맵 저장"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap='coolwarm', center=0, xticklabels=False, yticklabels=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Unified CSAR Layer Analysis")
    parser.add_argument('dataset_path', type=str, help="Path to trained dataset folder (e.g. trained_model/ml-100k)")
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Path {dataset_path} does not exist.")
        return

    exp_dirs = find_csar_experiments(dataset_path)
    print(f"Found {len(exp_dirs)} CSAR experiments.")
    
    summary = []
    
    for exp_dir in exp_dirs:
        print(f"\nAnalyzing {exp_dir.name}...")
        try:
            model, _, config = load_model(exp_dir)
            if model is None:
                continue
                
            keys = get_interest_keys(model)
            if keys is None:
                print(f"  Skipping: Could not find 'interest_keys' in model.")
                continue
                
            # Key Analysis
            stats, gram_norm, raw_gram = analyze_keys(keys, exp_dir.name)
            print(f"  K={stats['K']}, Orthogonality={stats['orthogonality']:.4f}, Norm={stats['norm_mean']:.4f}")
            
            # Save Heatmaps
            save_heatmap(gram_norm, exp_dir / 'gram_cosine.png', f"Cosine Similarity (K={stats['K']})")
            save_heatmap(raw_gram, exp_dir / 'gram_raw.png', f"Gram Matrix G=K@K.T (K={stats['K']})")
            
            summary.append(stats)
            
            # Save individual JSON
            with open(exp_dir / 'layer_analysis.json', 'w') as f:
                json.dump(stats, f, indent=4)
                
        except Exception as e:
            print(f"Error analyzing {exp_dir.name}: {e}")
            # import traceback; traceback.print_exc()

    if summary:
        df = pd.DataFrame(summary)
        # 컬럼 순서 정리
        cols = ['exp_name', 'K', 'orthogonality', 'avg_cosine_sim', 'norm_mean', 'max_cosine_sim']
        print("\n=== Analysis Summary ===")
        print(df[cols].to_string(index=False))
        
        save_csv_path = dataset_path / 'csar_analysis_summary.csv'
        df.to_csv(save_csv_path, index=False)
        print(f"\nSummary saved to {save_csv_path}")
    else:
        print("\nNo analysis results generated.")

if __name__ == "__main__":
    main()
