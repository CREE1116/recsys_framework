"""
데이터셋 자체의 인기도 편향(Popularity Bias) 및 롱테일 분포 분석 스크립트.
"""
import sys
from pathlib import Path

# 프로젝트 루트 추가
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import json
from collections import Counter

from src.data_loader import DataLoader

def calculate_gini_coefficient(array):
    """Gini Index 계산 (0: 완전 평등, 1: 완전 불평등)"""
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((2 * index - n - 1) * array).sum() / (n * array.sum())

def analyze_dataset_bias(config_path):
    print(f"Loading dataset configuration from {config_path}...")
    with open(config_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    # DataLoader 초기화를 위한 최소 설정 구성
    config = {
        'dataset': dataset_config, # analyze_rec2_layer 등 다른 곳에서 참조하는 구조 호환
        **dataset_config # DataLoader는 root level 키를 주로 사용
    }
    
    # 필수 파라미터 기본값 채우기 (Dataset Config에 없을 수 있는 학습 관련 설정)
    defaults = {
        'device': 'cpu',
        'train': {
            'batch_size': 1024,
            'num_negatives': 1,
            'negative_sampling_strategy': 'uniform'
        },
        'evaluation': {
            'validation_method': 'sampled',
            'final_method': 'full'
        },
        'min_user_interactions': 5,
        'min_item_interactions': 5,
        'rating_threshold': None
    }
    
    # config에 없는 경우 default 값으로 채움
    for k, v in defaults.items():
        if k not in config:
            config[k] = v
        elif isinstance(v, dict): # 중첩된 딕셔너리 처리
             for sub_k, sub_v in v.items():
                 if sub_k not in config[k]:
                     config[k][sub_k] = sub_v

    data_loader = DataLoader(config)
    
    if hasattr(data_loader, 'df'):
        df = data_loader.df
    else:
        print("Warning: DataLoader did not expose `df`. Attempting to access train/valid/test dfs...")
        df = pd.concat([data_loader.train_df, data_loader.valid_df, data_loader.test_df])
        
    print(f"Total Interactions: {len(df)}")
    print(f"Total Users: {df['user_id'].nunique()}")
    print(f"Total Items: {df['item_id'].nunique()}")

    # 1. Item Popularity Analysis
    item_counts = df['item_id'].value_counts().values
    
    # Gini Index
    gini = calculate_gini_coefficient(item_counts)
    
    # Head/Tail Ratio (Pareto Principle check)
    num_items = len(item_counts)
    head_count = max(1, int(num_items * 0.2))
    head_interactions = item_counts[:head_count].sum()
    total_interactions = item_counts.sum()
    head_tail_ratio = head_interactions / total_interactions

    print("\n[Item Popularity Statistics]")
    print(f"  Gini Index: {gini:.4f}")
    print(f"  Head(20%) Interaction Ratio: {head_tail_ratio:.4f}")
    print(f"  Max Popularity: {item_counts.max()}")
    print(f"  Min Popularity: {item_counts.min()}")
    print(f"  Avg Popularity: {item_counts.mean():.2f}")
    
    sparsity = 1.0 - (len(df) / (data_loader.n_users * data_loader.n_items))
    print(f"  Sparsity: {sparsity:.6f}")

    # 2. Visualization & Saving
    output_dir = Path("output") / config['dataset_name'] / "dataset_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Long-tail Distribution (Log-Log Plot)
    plt.figure(figsize=(10, 6))
    rank = np.arange(1, len(item_counts) + 1)
    plt.loglog(rank, item_counts, marker='.', linestyle='none', alpha=0.5, color='blue')
    plt.title(f"Long-tail Distribution (Item Popularity) - {config['dataset_name']}")
    plt.xlabel("Item Rank (Log)")
    plt.ylabel("Interaction Count (Log)")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(output_dir / "long_tail_distribution.png", dpi=300)
    plt.close()

    # Lorenz Curve
    sorted_counts = np.sort(item_counts)
    cumulative_interactions = np.cumsum(sorted_counts) / sorted_counts.sum()
    cumulative_items = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    
    plt.figure(figsize=(6, 6))
    plt.plot(cumulative_items, cumulative_interactions, label=f"Lorenz Curve (Gini={gini:.2f})", color='red')
    plt.plot([0, 1], [0, 1], 'k--', label="Perfect Equality")
    plt.fill_between(cumulative_items, cumulative_items, cumulative_interactions, alpha=0.1, color='red')
    plt.title(f"Lorenz Curve - {config['dataset_name']}")
    plt.xlabel("Cumulative % of Items (Sorted by popularity)")
    plt.ylabel("Cumulative % of Interactions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "lorenz_curve.png", dpi=300)
    plt.close()

    # Save Stats JSON
    stats = {
        'dataset_name': config['dataset_name'],
        'gini_index': float(gini),
        'head_ratio_20': float(head_tail_ratio),
        'sparsity': float(sparsity),
        'num_users': int(data_loader.n_users),
        'num_items': int(data_loader.n_items),
        'num_interactions': int(len(df)),
        'avg_popularity': float(item_counts.mean()),
        'max_popularity': int(item_counts.max())
    }
    with open(output_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    # Generate Markdown Report
    markdown_content = f"""# Dataset Bias Analysis: {config['dataset_name']}

## Overview
- **Users**: {stats['num_users']:,}
- **Items**: {stats['num_items']:,}
- **Interactions**: {stats['num_interactions']:,}
- **Sparsity**: {stats['sparsity']:.6f}

## Popularity Bias Metrics
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Gini Index** | **{gini:.4f}** | 0 (Equal) ~ 1 (Inequal). Higher means more bias. |
| **Head (Top 20%) Ratio** | **{head_tail_ratio:.4f}** | Ratio of interactions covered by top 20% popular items. |
| **Max Popularity** | {stats['max_popularity']} | Hits of the most popular item. |
| **Avg Popularity** | {stats['avg_popularity']:.2f} | Average hits per item. |

## Visualizations

### Long-tail Distribution
![Long-tail Curve](long_tail_distribution.png)
*Log-Log plot of Item Rank vs Interaction Count. A straight line indicates a Power Law distribution.*

### Lorenz Curve
![Lorenz Curve](lorenz_curve.png)
*Cumulative distribution of items vs interactions. The area between the curve and the diagonal represents inequality (Gini).*
"""
    with open(output_dir / "README.md", "w") as f:
        f.write(markdown_content)
    
    print(f"\nAnalysis results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to dataset config yaml (e.g., configs/dataset/ml100k.yaml)')
    args = parser.parse_args()
    
    analyze_dataset_bias(args.config_path)
