import torch
import yaml
import json
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path

from src.data_loader import DataLoader
from src.models import MODEL_REGISTRY

def load_trained_model(exp_path, device='cpu'):
    """
    학습된 모델을 로드하여 반환합니다.
    
    Args:
        exp_path (str or Path): 실험 결과 폴더 경로 (예: trained_model/ml-100k/csar-rec2...)
        device (str): 로드할 디바이스
        
    Returns:
        model: 학습된 모델 인스턴스 (eval 모드)
        data_loader: 해당 실험 설정으로 초기화된 DataLoader
        config: 로드된 설정 딕셔너리
    """
    exp_path = Path(exp_path)
    config_path = exp_path / 'config.yaml'
    model_path = exp_path / 'best_model.pt'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 디바이스 오버라이드 (분석 시에는 지정된 디바이스 사용)
    config['device'] = device
    
    # DataLoader 초기화
    # 학습 시와 동일한 설정을 사용하되, 배치 사이즈 등은 조정 가능
    data_loader = DataLoader(config)
    
    # 모델 초기화
    model_name = config['model']['name']
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}")
        
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(config, data_loader)
    
    # 가중치 로드
    if model_path.exists():
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model weights from {model_path}")
    else:
        print(f"Warning: Model weights not found at {model_path}. Using initialized weights.")
    
    model.to(device)
    model.eval()
    
    return model, data_loader, config

# Alias creation for compatibility
load_model_from_run = load_trained_model

def get_item_popularity(data_loader):
    """
    DataLoader에서 학습 데이터 기준 아이템 인기도를 반환합니다.
    
    Returns:
        pop_series (pd.Series): 인덱스는 item_id, 값은 interaction count
    """
    if hasattr(data_loader, 'item_popularity'):
         # Series (index: item_id, value: count)
        return data_loader.item_popularity
    else:
        # popularity가 없는 경우 계산
        train_df = data_loader.train_df
        return train_df['item_id'].value_counts().sort_index().reindex(range(data_loader.n_items), fill_value=0)

def split_items_by_popularity(pop_series, ratios={'head': 0.2, 'tail': 0.8}):
    """
    아이템을 Head/Tail (또는 더 세분화)로 그룹화합니다.
    """
    # 인기도 순으로 정렬된 아이템 ID
    sorted_items = pop_series.sort_values(ascending=False).index.values
    n_items = len(sorted_items)
    
    head_cutoff = int(n_items * ratios['head'])
    
    groups = {
        'head': sorted_items[:head_cutoff],
        'tail': sorted_items[head_cutoff:]
    }
    return groups

# ========================================================================================
# [ADDED] Missing Utilities for Analysis Scripts
# ========================================================================================

def get_analysis_output_path(dataset_name, run_name):
    """
    분석 결과 저장 경로를 생성하고 반환합니다.
    Structure: analysis_results/{dataset_name}/{run_name}/
    """
    # Project root (assumed to be 2 levels up from analysis/utils.py)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'analysis_results', dataset_name, run_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

class AnalysisReport:
    """
    Markdown 리포트를 생성하는 간단한 클래스.
    """
    def __init__(self, title, output_dir):
        self.title = title
        self.output_dir = output_dir
        self.content = [f"# {title}\n"]
        self.images_dir = os.path.join(output_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)

    def add_section(self, title, level=2):
        self.content.append(f"\n{'#' * level} {title}\n")

    def add_text(self, text):
        self.content.append(f"{text}\n")

    def add_table(self, df):
        self.content.append(f"\n{df.to_markdown(index=False)}\n")

    def add_figure(self, filename, caption=""):
        # The filename should be just the basename if it's saved in the output_dir
        # If the script saves it to output_dir, we just link it.
        # Markdown convention: ![Alt text](url)
        self.content.append(f"\n![{caption}]({filename})\n")
        if caption:
            self.content.append(f"*{caption}*\n")

    def save(self, filename="report.md"):
        full_path = os.path.join(self.output_dir, filename)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.content))
        print(f"Report generated at: {full_path}")

def load_item_metadata(dataset_name, data_path_or_dir):
    """
    데이터셋의 아이템 메타데이터(제목, 장르 등)를 로드합니다.
    기본적으로 ml-1m (movies.dat) 등을 지원합니다.
    
    Args:
        dataset_name (str): 데이터셋 이름 (예: ml-1m)
        data_path_or_dir (str): ratings.dat 파일 경로 또는 데이터 디렉토리
        
    Returns:
        pd.DataFrame: index가 item_id(str)이고 컬럼이 메타데이터인 DF
    """
    # 디렉토리 추론
    if os.path.isfile(data_path_or_dir):
        data_dir = os.path.dirname(data_path_or_dir)
    else:
        data_dir = data_path_or_dir
        
    # Dataset specific logic
    if dataset_name == 'ml-1m':
        # Try movies.dat in data_dir
        meta_path = os.path.join(data_dir, 'movies.dat')
        if not os.path.exists(meta_path):
             # Fallback: check project data dir structure
             # Try ../data/ml-1m/movies.dat logic if needed, but let's assume it's co-located
             print(f"[Warning] Metadata file not found at {meta_path}. Returning empty metadata.")
             return pd.DataFrame()
             
        # Load movies.dat (ISO-8859-1 usually)
        try:
            df = pd.read_csv(meta_path, sep='::', engine='python', encoding='iso-8859-1', header=None, names=['item_id', 'title', 'genres'])
            df['item_id'] = df['item_id'].astype(str) # Ensure string ID
            return df.set_index('item_id')
        except Exception as e:
            print(f"[Error] Failed to load metadata: {e}")
            return pd.DataFrame()
            
    elif dataset_name == 'ml-100k':
        # u.item
        meta_path = os.path.join(data_dir, 'u.item')
        if os.path.exists(meta_path):
            try:
                # 100k: id | title | date | ... | url | genre_flags...
                # Just take ID and Title for now
                df = pd.read_csv(meta_path, sep='|', engine='python', encoding='iso-8859-1', header=None, usecols=[0, 1], names=['item_id', 'title'])
                df['item_id'] = df['item_id'].astype(str)
                return df.set_index('item_id')
            except:
                pass

    print(f"[Info] No metadata loader implemented for {dataset_name}. Returning empty.")
    return pd.DataFrame()
