import os
import sys
import yaml
import torch
import numpy as np
from scipy.sparse import csr_matrix

# Framework root path
sys.path.append(os.getcwd())

from src.data_loader import DataLoader
from src.utils.gpu_accel import SVDCacheManager

def load_config(dataset_name):
    """YAML 파일을 로드하여 설정을 반환합니다."""
    # dataset_name can be a path or a name in configs/dataset/
    if not dataset_name.endswith('.yaml'):
        config_path = f"configs/dataset/{dataset_name}.yaml"
    else:
        config_path = dataset_name
        
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # model name is required by DataLoader
    if 'model' not in config:
        config['model'] = {'name': 'aspire'}
    
    return config

def get_loader_and_svd(dataset_name, k=None, target_energy=0.95):
    """DataLoader와 SVD 데이터를 초기화합니다."""
    config = load_config(dataset_name)
    loader = DataLoader(config)
    
    # R (Interaction Matrix) 생성
    rows = loader.train_df['user_id'].values
    cols = loader.train_df['item_id'].values
    vals = np.ones(len(rows))
    R = csr_matrix((vals, (rows, cols)), shape=(loader.n_users, loader.n_items))
    
    # SVD 계산 (target_energy 기준)
    svd_manager = SVDCacheManager()
    U, S, V, _ = svd_manager.get_svd(R, k=k, target_energy=target_energy, dataset_name=config["dataset_name"])
    
    return loader, R, S, V, config

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
