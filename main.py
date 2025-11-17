import argparse
import yaml
import pprint
import torch # torch.set_default_device를 위해 추가

from src.data_loader import DataLoader
from src.models import get_model
from src.trainer import Trainer

def main(config):
    """
    메인 실행 함수
    """
    # 1. 데이터 로딩 및 전처리
    data_loader = DataLoader(config)

    # 2. 모델 생성
    # config의 모델 이름을 기반으로 src/models/__init__.py에서 모델을 동적으로 가져옴
    model_name = config['model']['name']
    model = get_model(model_name, config, data_loader)
    
    # 3. 트레이너 생성
    trainer = Trainer(config, model, data_loader)

    # 4. 학습 시작
    trainer.train()

if __name__ == '__main__':
    # 커맨드 라인 인자 파싱
    parser = argparse.ArgumentParser(description="Recommendation System Framework")
    parser.add_argument('--dataset_config', type=str, default='configs/dataset/ml100k.yaml',
                        help='Path to the dataset configuration file.')
    parser.add_argument('--model_config', type=str, default='configs/model/csar.yaml',
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
    if config.get('device') == 'auto':
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
