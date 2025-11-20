import argparse
import yaml
import itertools
import os
import copy
import pprint
import torch
from main import main as run_single_experiment

def _find_list_params_recursive(config_dict, path, list_params):
    """설정 딕셔너리를 재귀적으로 탐색하여 값이 리스트인 파라미터를 찾습니다."""
    for key, value in config_dict.items():
        new_path = f"{path}.{key}" if path else key
        
        if isinstance(value, list):
            list_params[new_path] = value
        elif isinstance(value, dict):
            _find_list_params_recursive(value, new_path, list_params)

def generate_hyperparameter_combinations(config):
    """
    설정 파일에서 리스트 형태의 하이퍼파라미터를 찾아 모든 조합을 생성합니다.
    'model'과 'train' 섹션 내에서만 하이퍼파라미터를 탐색합니다.
    리스트에 요소가 하나만 있는 경우(e.g., [[64, 32]])는 하이퍼파라미터로 취급하지 않고,
    내부 값을 실제 파라미터 값으로 사용합니다.
    """
    list_params = {}
    search_sections = {'model': config.get('model', {}), 'train': config.get('train', {})}
    
    for section_name, section_config in search_sections.items():
        _find_list_params_recursive(section_config, section_name, list_params)

    # 실제 하이퍼파라미터(요소 > 1)와 단일 값 리스트를 분리
    hyperparams_to_search = {}
    base_config_modifier = {}
    
    for name, values in list_params.items():
        if len(values) == 1:
            base_config_modifier[name] = values[0]
        else:
            hyperparams_to_search[name] = values

    # 단일 값 리스트는 기본 설정에 미리 적용
    for param_name, value in base_config_modifier.items():
        keys = param_name.split('.')
        temp_conf = config
        for key in keys[:-1]:
            temp_conf = temp_conf[key]
        temp_conf[keys[-1]] = value

    if not hyperparams_to_search:
        config['run_name'] = 'default'
        yield config
        return

    param_names = list(hyperparams_to_search.keys())
    param_values = list(hyperparams_to_search.values())
    
    for combo in itertools.product(*param_values):
        new_config = copy.deepcopy(config)
        combo_str_parts = []
        
        for i, param_name in enumerate(param_names):
            keys = param_name.split('.')
            temp_conf = new_config
            for key in keys[:-1]:
                temp_conf = temp_conf[key]
            
            temp_conf[keys[-1]] = combo[i]
            
            key_for_path = keys[-1]
            val_for_path = str(combo[i])
            combo_str_parts.append(f"{key_for_path}={val_for_path}")
        
        new_config['run_name'] = "_".join(combo_str_parts)
        
        yield new_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hyperparameter Grid Search for Recommendation System")
    parser.add_argument('--dataset_config', type=str, default='configs/dataset/ml100k.yaml',
                        help='Path to the dataset configuration file.')
    parser.add_argument('--model_config', type=str, default='configs/model/csar.yaml',
                        help='Path to the model configuration file for grid search.')
    args = parser.parse_args()

    # 설정 파일 로드
    with open(args.dataset_config, 'r') as f:
        dataset_config = yaml.safe_load(f)
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)

    # 두 설정을 병합
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

    # 하이퍼파라미터 조합 생성 및 실험 실행
    for i, run_config in enumerate(generate_hyperparameter_combinations(config)):
        print("="*60)
        print(f"Grid Search Run {i+1}: Starting experiment with run_name='{run_config['run_name']}'")
        print("="*60)
        pprint.pprint(run_config)
        print("---")
        
        # 단일 실험 실행
        run_single_experiment(run_config)
        
        print("\n" * 2)

    print("="*60)
    print("Grid search finished.")
    print("="*60)
