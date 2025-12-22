import argparse
import torch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import os
import sys

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from src.data_loader import RecSysDataLoader

def estimate_k_with_svd(data_path, dataset_name, max_k=200):
   """
   Interaction Matrix의 SVD를 통해 적정 K(Latent Dimension/Interest)를 추정합니다.
   """
   print(f"Loading Dataset: {dataset_name}...")
   
   # Data Loader 재사용 (경로 호환성 위해)
   # 실제로는 csv/txt 직접 읽는 게 더 빠를 수 있으나, 일관성을 위해 사용
   # 편의상 train/test 구분 없이 전체 데이터의 복잡도를 보는 것이 좋음 (여기서는 train만 사용)
   
   # Hardcoded for ML-1M based on config investigation
   # 실제로는 config를 로드해서 쓰는 게 좋지만, 퀵 툴이므로 직접 지정
   # data_path arg가 파일 경로를 가리키게 수정하거나, ml-1m이면 자동 처리
   
   file_path = data_path
   sep = "::"
   if dataset_name == 'ml-1m' and os.path.isdir(data_path):
       file_path = os.path.join(data_path, "ml1m", "ratings.dat")
   elif dataset_name == 'ml-1m' and not os.path.exists(data_path):
        # Fallback to absolute path found in config
        file_path = "/Users/leejongmin/code/recsys_framework/data/ml1m/ratings.dat"
       
   print(f"Reading data from: {file_path}")
   try:
       df = pd.read_csv(file_path, sep=sep, header=None, names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python')
   except Exception as e:
       print(f"Error reading file: {e}")
       return

   # Rating Threshold (Optional but good for noise reduction)
   # Config says threshold 4. Let's apply it to match training data
   if 'rating' in df.columns:
       print("Applying rating threshold >= 4 (Implicit feedback assumption)...")
       df = df[df['rating'] >= 4]
   
   n_users = df['user_id'].nunique()
   n_items = df['item_id'].nunique()
   
   print(f"Users: {n_users}, Items: {n_items}, Interactions: {len(df)}")
   
   # Sparse Matrix 생성
   row = df['user_id'].values
   col = df['item_id'].values
   data = np.ones(len(df))
   
   shape_u = row.max() + 1
   shape_i = col.max() + 1
   
   # CSR Matrix
   adj_mat = csr_matrix((data, (row, col)), shape=(shape_u, shape_i)) # 0-index safe
   
   print(f"Running SVD (k={max_k})... This might take a minute.")
   
   # SVD 수행
   # k개의 singular value만 구함 (Truncated SVD)
   # return: u, s, vt. s is singular values (ascending order usually in scipy, need check)
   u, s, vt = svds(adj_mat, k=max_k)
   
   # svds returns singular values in ascending order, so reverse it
   s = s[::-1]
   
   # Singular Value Spectrum 분석
   # s^2 (Eigenvalues of Covariance) represents explained variance (energy)
   energy = s**2
   total_energy = energy.sum() # NOTE: This is sum of top-k energy, not total matrix energy (which is hard to copmute for massive sparse)
   # But analyzing the shape of the curve is enough to find the elbow.
   
   # Normalize by top-1 (Relative drop)
   normalized_s = s / s[0]
   
   # find elbow (heuristic: curvature)
   # Or find k where cumulative energy reaches X% of top-MaxK energy
   
   print("\n[Estimated K Analysis]")
   print("-" * 30)
   print(f"Top-1 Singular Value: {s[0]:.4f}")
   print(f"Top-10 Avg: {s[:10].mean():.4f}")
   print(f"Top-{max_k} Avg: {s.mean():.4f}")
   
   # Thresholds
   thresholds = [0.9, 0.8, 0.5, 0.1] # relative to max
   
   recommendations = {}
   
   for t in thresholds:
       # Find first k where s[k] < t * s[0]
       idx = np.where(normalized_s < t)[0]
       if len(idx) > 0:
           recommendations[f"Drop to {int(t*100)}%"] = idx[0] + 1
   
   # Explained Variance (if we assume max_k covers significant portion)
   # Cumulative plot
   cumulative_energy = np.cumsum(energy) / total_energy
   idx_80 = np.where(cumulative_energy >= 0.80)[0]
   if len(idx_80) > 0:
        recommendations["80% Variance (of Top-K)"] = idx_80[0] + 1

   idx_90 = np.where(cumulative_energy >= 0.90)[0]
   if len(idx_90) > 0:
        recommendations["90% Variance (of Top-K)"] = idx_90[0] + 1
        
   print("Recommended K candidates:")
   for reason, k_val in recommendations.items():
       print(f"  - {reason}: K ≈ {k_val}")
       
   # Plot saving
   plt.figure(figsize=(10, 6))
   plt.plot(range(1, max_k+1), s, marker='o', markersize=2)
   plt.title(f"Singular Value Spectrum ({dataset_name})")
   plt.xlabel("K (Rank)")
   plt.ylabel("Singular Value")
   plt.grid(True)
   plt.savefig(f'analysis/svd_spectrum_{dataset_name}.png')
   print(f"Saved plot to analysis/svd_spectrum_{dataset_name}.png")

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('--dataset', type=str, default='ml-1m')
   parser.add_argument('--data_path', type=str, default='data')
   parser.add_argument('--max_k', type=int, default=150, help="Check up to this rank")
   args = parser.parse_args()
   
   estimate_k_with_svd(args.data_path, args.dataset, args.max_k)
