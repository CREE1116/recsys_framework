import os
import sys
import yaml
import json
import numpy as np
import pandas as pd
from glob import glob

# Framework root path
sys.path.append(os.getcwd())

from src.data_loader import DataLoader

def generate_stats():
    config_dir = "configs/dataset"
    output_base_dir = "data/dataset_stats"
    os.makedirs(output_base_dir, exist_ok=True)
    
    config_files = sorted(glob(os.path.join(config_dir, "*.yaml")))
    
    all_summary = []
    
    for config_path in config_files:
        print(f"\nProcessing {config_path}...")
        try:
            with open(config_path, 'r') as f:
                dataset_config = yaml.safe_load(f)
            
            # YAML 파일에 없는 경우 파일명으로 이름 지정
            if 'dataset_name' not in dataset_config:
                dataset_config['dataset_name'] = os.path.basename(config_path).replace('.yaml', '')
            
            # DataLoader 초기화 (데이터 필터링 및 리매핑 수행)
            loader = DataLoader(dataset_config)
            
            n_users = loader.n_users
            n_items = loader.n_items
            n_interactions = len(loader.df)
            density = n_interactions / (n_users * n_items) if n_users * n_items > 0 else 0
            
            # 인터랙션 분계
            n_train = len(loader.train_df)
            n_valid = len(loader.valid_df)
            n_test = len(loader.test_df)
            
            # 차수(Degree) 통계
            user_counts = loader.df['user_id'].value_counts()
            item_counts = loader.df['item_id'].value_counts()
            
            stats = {
                "dataset_name": dataset_config['dataset_name'],
                "n_users": int(n_users),
                "n_items": int(n_items),
                "n_interactions": int(n_interactions),
                "density": float(density),
                "sparsity": float(1.0 - density),
                "split": {
                    "train": int(n_train),
                    "valid": int(n_valid),
                    "test": int(n_test)
                },
                "user_degree": {
                    "avg": float(user_counts.mean()),
                    "min": int(user_counts.min()),
                    "max": int(user_counts.max()),
                    "median": float(user_counts.median())
                },
                "item_degree": {
                    "avg": float(item_counts.mean()),
                    "min": int(item_counts.min()),
                    "max": int(item_counts.max()),
                    "median": float(item_counts.median())
                }
            }
            
            # 결과 저장 (데이터셋별 개별 파일)
            target_save_dir = os.path.dirname(dataset_config['data_path'])
            if os.path.isdir(target_save_dir):
                save_path = os.path.join(target_save_dir, "stats.json")
                with open(save_path, 'w') as f:
                    json.dump(stats, f, indent=4)
                print(f"  Saved stats to {save_path}")
            
            # 중앙 관리용 저장
            central_save_path = os.path.join(output_base_dir, f"{dataset_config['dataset_name']}.json")
            with open(central_save_path, 'w') as f:
                json.dump(stats, f, indent=4)
            
            all_summary.append(stats)
            
        except Exception as e:
            print(f"Error processing {config_path}: {e}")
            continue
            
    # 전체 요약 리포트 테이블 형식으로 출력 (Markdown)
    if all_summary:
        print("\n" + "="*80)
        print(f"{'Dataset':<20} | {'Users':<10} | {'Items':<10} | {'Interactions':<12} | {'Density':<10}")
        print("-"*80)
        for s in all_summary:
            print(f"{s['dataset_name']:<20} | {s['n_users']:<10} | {s['n_items']:<10} | {s['n_interactions']:<12} | {s['density']:.6f}")
        print("="*80)
        
        # 전체 요약 JSON 저장
        with open(os.path.join(output_base_dir, "all_datasets.json"), 'w') as f:
            json.dump(all_summary, f, indent=4)

if __name__ == "__main__":
    generate_stats()
