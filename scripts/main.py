import argparse
import yaml
import pprint
import torch
import numpy as np
import random
import collections
import json
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DataLoader
from src.models import get_model
from src.trainer import Trainer


def set_seed(seed=42):
    """재현성을 위한 전역 Seed 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Deterministic 설정 (속도가 느려질 수 있음)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(config):
    """
    메인 실행 함수
    """
    seed = config.get('seed', 42)
    
    # 0. 재현성을 위한 Seed 고정
    set_seed(seed)
    print(f"\n{'='*50}\nStarting run with Seed: {seed}\n{'='*50}")
        
    # 1. 데이터 로딩 및 전처리
    data_loader = DataLoader(config)

    # 2. 모델 생성
    model_name = config['model']['name']
    model = get_model(model_name, config, data_loader)
    
    # 3. 트레이너 생성 및 실행 (fit → train → evaluate 자동 분기)
    trainer = Trainer(config, model, data_loader)
    current_metrics = trainer.run()

if __name__ == '__main__':
    # 커맨드 라인 인자 파싱
    parser = argparse.ArgumentParser(description="Recommendation System Framework")
    parser.add_argument('--dataset_config', type=str, default='configs/dataset/ml100k.yaml',
                        help='Path to the dataset configuration file.')
    parser.add_argument('--model_config', type=str, default='configs/model/csar/csar.yaml',
                        help='Path to the model configuration file.')
    parser.add_argument('--eval_config', type=str, default=None,
                        help='Path to evaluation master config. Default: configs/evaluation.yaml')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Optional run name to identify the experiment.')
    args = parser.parse_args()

    # 설정 파일 로드
    with open(args.dataset_config, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)
    with open(args.model_config, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)

    # run_name 설정
    if args.run_name:
        model_config['run_name'] = args.run_name

    # 마스터 evaluation config 로드 및 3단계 병합
    # 우선순위: evaluation.yaml(기본) → dataset(데이터셋 특화) → model(최종 결정)
    from config_utils import merge_all_configs
    config = merge_all_configs(dataset_config, model_config, eval_config_path=args.eval_config)

    # MPS 장치 사용 설정
    if config.get('device', 'auto') == 'auto':
        if torch.backends.mps.is_available():
            print("MPS is available. Using MPS device.")
            config['device'] = 'mps'
        elif torch.cuda.is_available():
            print("Using CUDA device.")
            config['device'] = 'cuda'
        else:
            print("Using CPU device.")
            config['device'] = 'cpu'

    print("="*20, "Configuration", "="*20)
    pprint.pprint(config)
    print("="*55)

    main(config)

