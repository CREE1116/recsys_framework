import os
import yaml
import re
import pandas as pd
import torch
import sys
from contextlib import contextmanager
from tabulate import tabulate

# 프로젝트의 src 폴더를 python 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DataLoader
from src.models import get_model

class AnalysisReport:
    """
    분석 결과를 체계적인 Markdown 리포트로 생성하는 클래스.
    """
    def __init__(self, title, output_path):
        self.title = title
        self.output_path = output_path
        self.elements = [f"# {self.title}\n"]
        os.makedirs(self.output_path, exist_ok=True)

    def add_section(self, title, level=2):
        """리포트에 섹션 제목을 추가합니다."""
        self.elements.append(f"{'#' * level} {title}\n")

    def add_text(self, text):
        """리포트에 일반 텍스트를 추가합니다."""
        self.elements.append(f"{text}\n")

    def add_table(self, df):
        """리포트에 Pandas DataFrame을 Markdown 테이블로 추가합니다."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("add_table expects a Pandas DataFrame.")
        # 'tabulate'를 사용하여 DataFrame을 Markdown 테이블로 변환
        self.elements.append(tabulate(df, headers='keys', tablefmt='pipe', showindex=False) + "\n")
    
    def add_figure(self, figure_name, caption=""):
        """리포트에 이미지(피규어) 링크를 추가합니다."""
        # 경로는 리포트 파일 기준 상대 경로
        self.elements.append(f"![{caption}]({figure_name})\n")

    def save(self, filename="report.md"):
        """리포트를 Markdown 파일로 저장합니다."""
        report_path = os.path.join(self.output_path, filename)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.elements))
        print(f"Analysis report saved to: {report_path}")

def get_analysis_output_path(dataset_name, run_name=None):
    """
    분석 결과물이 저장될 경로를 생성하고 반환합니다.
    - 루트 경로: output/
    - 구조: output/{dataset_name}/{run_name}/
    """
    path_parts = [
        "output",
        dataset_name.replace('/', '_') # e.g., 'ml-1m'
    ]
    
    if run_name:
        path_parts.append(run_name)
        
    output_path = os.path.join(*path_parts)
    os.makedirs(output_path, exist_ok=True)
    
    return output_path

def load_item_metadata(dataset_name, data_path):
    """아이템 메타데이터(제목, 연도, 장르)를 로드합니다."""
    print(f"Loading item metadata for {dataset_name}...")
    
    metadata_path = ''
    if 'ml-1m' in dataset_name:
        metadata_path = os.path.join(os.path.dirname(data_path), 'movies.dat')
        movies_df = pd.read_csv(metadata_path, sep='::', header=None, names=['item_id', 'title', 'genres'], 
                                engine='python', encoding='ISO-8859-1')
        movies_df['year'] = movies_df['title'].apply(lambda x: int(re.search(r'\((\d{4})\)', x).group(1)) if re.search(r'\((\d{4})\)', x) else np.nan)
        movies_df['genres'] = movies_df['genres'].apply(lambda x: x.split('|'))
        movies_df.set_index('item_id', inplace=True)

    elif 'ml100k' in dataset_name:
        metadata_path = os.path.join(os.path.dirname(data_path), 'u.item')
        genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 
                      'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
                      'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        col_names = ['item_id', 'title', 'release_date'] + ['video_release_date', 'IMDb_URL'] + genre_cols
        movies_df = pd.read_csv(metadata_path, sep='|', header=None, names=col_names,
                                engine='python', encoding='ISO-8859-1')
        movies_df['year'] = pd.to_datetime(movies_df['release_date']).dt.year
        
        def get_genres(row, genre_cols):
            return [col for col in genre_cols if row[col] == 1]
        movies_df['genres'] = movies_df.apply(get_genres, axis=1, genre_cols=genre_cols)
        movies_df.set_index('item_id', inplace=True)

    else:
        # 다른 데이터셋에 대한 메타데이터 로딩 로직 추가 가능
        print(f"[Warning] Metadata loading not implemented for '{dataset_name}'. Returning empty DataFrame.")
        return pd.DataFrame()
        
    print(f"Loaded {len(movies_df)} items from {metadata_path}")
    return movies_df

def load_model_from_run(run_path):
    """
    학습된 모델의 run 경로에서 config와 checkpoint를 읽어와
    모델, 데이터로더, 설정을 반환합니다.
    
    Args:
        run_path (str): 학습된 모델의 개별 run 폴더 경로
                        (e.g., 'trained_model/ml-1m/CSAR__default')
                        
    Returns:
        tuple: (model, data_loader, config) or (None, None, None) on failure
    """
    config_path = os.path.join(run_path, 'config.yaml')
    model_path = os.path.join(run_path, 'best_model.pt')

    if not os.path.exists(config_path) or not os.path.exists(model_path):
        print(f"[Error] config.yaml or best_model.pt not found in {run_path}")
        return None, None, None

    # 1. 설정 로드
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. 장치 설정
    if config.get('device') == 'auto':
        if torch.backends.mps.is_available():
            config['device'] = 'mps'
        elif torch.cuda.is_available():
            config['device'] = 'cuda'
        else:
            config['device'] = 'cpu'
    device = torch.device(config['device'])
    
    # 3. 데이터로더, 모델 인스턴스화
    # print("Instantiating DataLoader and Model...")
    data_loader = DataLoader(config)
    model = get_model(config['model']['name'], config, data_loader)
    
    # 4. 학습된 가중치 로드
    # print(f"Loading trained model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, data_loader, config
