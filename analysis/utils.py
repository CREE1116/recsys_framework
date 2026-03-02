import os
import yaml
import torch
import pandas as pd
import numpy as np
from src.models import get_model
from src.data_loader import DataLoader

def load_model_from_run(run_folder_path):
    """
    실험 폴더에서 config와 best_model.pt를 로드하여 모델을 복원합니다.
    """
    config_path = os.path.join(run_folder_path, 'config.yaml')
    if not os.path.exists(config_path):
        print(f"[Error] Config not found: {config_path}")
        return None, None, None
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # 데이터 로더 초기화 (모델 초기화에 필요)
    data_loader = DataLoader(config)
    
    # 모델 생성
    model = get_model(config['model']['name'], config, data_loader)
    
    # 가중치 로드
    checkpoint_path = os.path.join(run_folder_path, 'best_model.pt')
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=model.device))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"[Warning] Checkpoint not found: {checkpoint_path}")
        
    model.eval()
    return model, data_loader, config

def load_item_metadata(dataset_name, data_path):
    """
    데이터셋별 아이템 메타데이터(제목, 장르 등)를 로드합니다.
    """
    data_dir = os.path.dirname(data_path)
    
    if 'ml-100k' in dataset_name:
        item_path = os.path.join(data_dir, 'u.item')
        if os.path.exists(item_path):
            cols = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + [f'genre_{i}' for i in range(19)]
            df = pd.read_csv(item_path, sep='|', names=cols, encoding='latin-1')
            
            # 장르 통합 (0/1 형태를 문자열 리스트로)
            genre_list = ['unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
            
            def get_genres(row):
                return '|'.join([genre_list[i] for i in range(19) if row[f'genre_{i}'] == 1])
                
            df['genres'] = df.apply(get_genres, axis=1)
            return df.set_index('item_id')
            
    elif 'ml-1m' in dataset_name:
        item_path = os.path.join(data_dir, 'movies.dat')
        if os.path.exists(item_path):
            df = pd.read_csv(item_path, sep='::', names=['item_id', 'title', 'genres'], engine='python', encoding='latin-1')
            return df.set_index('item_id')
            
    # Default / Amazon support
    meta_path = os.path.join(data_dir, 'item_metadata.csv')
    if os.path.exists(meta_path):
        return pd.read_csv(meta_path).set_index('item_id')
        
    return pd.DataFrame(columns=['title', 'genres'])

def get_analysis_output_path(dataset_name, run_name):
    """분석 결과 저장 경로를 생성합니다."""
    path = os.path.join('analysis_results', dataset_name, run_name)
    os.makedirs(path, exist_ok=True)
    return path

class AnalysisReport:
    """분석 결과를 Markdown 리포트로 작성하는 유틸리티."""
    def __init__(self, title, output_path):
        self.title = title
        self.output_path = output_path
        self.content = f"# {title}\n\n"

    def add_section(self, title, level=2):
        self.content += f"{'#' * level} {title}\n\n"

    def add_text(self, text):
        self.content += f"{text}\n\n"

    def add_table(self, df):
        if not df.empty:
            self.content += df.to_markdown(index=False) + "\n\n"
        else:
            self.content += "*Empty Table*\n\n"

    def add_figure(self, filename, caption):
        self.content += f"![{caption}]({filename})\n\n"

    def save(self, filename="analysis_report.md"):
        full_path = os.path.join(self.output_path, filename)
        with open(full_path, "w") as f:
            f.write(self.content)
        print(f"Report saved to {full_path}")
