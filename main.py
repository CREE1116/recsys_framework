import argparse
import yaml
import pprint
import torch
import numpy as np
import random

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
    # 0. 재현성을 위한 Seed 고정
    seed = config.get('seed', 42)
    set_seed(seed)
    print(f"Random seed set to: {seed}")
    
    # 1. 데이터 로딩 및 전처리
    data_loader = DataLoader(config)

    # 2. 모델 생성
    model_name = config['model']['name']
    model = get_model(model_name, config, data_loader)
    
    # 3. 트레이너 생성
    trainer = Trainer(config, model, data_loader)

    # 4. 모델 타입에 따라 학습 또는 바로 평가를 실행
    if 'train' in config:
        # 학습 가능한 모델
        print("Trainable model detected. Starting training process...")
        trainer.train()
    else:
        # 학습이 필요 없는 모델 (e.g., ItemKNN, MostPopular)
        print("Non-trainable model detected. Proceeding directly to final evaluation...")
        # ItemKNN과 같은 모델은 fit 과정이 필요
        if hasattr(model, 'fit'):
            print(f"Fitting model {model_name}...")
            # fit() 메소드는 data_loader를 사용할 수 있어야 함
            # base_model의 fit 메소드는 data_loader를 인자로 받지 않을 수 있으므로,
            # 모델 내부에서 data_loader에 접근하거나, fit 메소드를 적절히 정의해야 함
            model.fit(data_loader)
        trainer.evaluate(is_final_evaluation=True)

if __name__ == '__main__':
    # 커맨드 라인 인자 파싱
    parser = argparse.ArgumentParser(description="Recommendation System Framework")
    parser.add_argument('--dataset_config', type=str, default='configs/dataset/ml100k.yaml',
                        help='Path to the dataset configuration file.')
    parser.add_argument('--model_config', type=str, default='configs/model/csar/csar.yaml',
                        help='Path to the model configuration file.')
    args = parser.parse_args()

    # 설정 파일 로드
    with open(args.dataset_config, 'r') as f:
        dataset_config = yaml.safe_load(f)
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)

    # 두 설정을 병합 (모델 설정이 데이터셋 설정을 덮어쓰도록)
    config = dataset_config
    for key, value in model_config.items():
        if key in config and isinstance(config[key], dict) and isinstance(value, dict):
            config[key].update(value)
        else:
            config[key] = value

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
