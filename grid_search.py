import argparse
import yaml
import itertools
import os
import copy
import pprint
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
    [수정] 'model'과 'train' 섹션 내에서만 하이퍼파라미터를 탐색합니다.
    """
    list_params = {}
    # 탐색할 섹션 지정
    search_sections = {'model': config.get('model', {}), 'train': config.get('train', {})}
    
    for section_name, section_config in search_sections.items():
        _find_list_params_recursive(section_config, section_name, list_params)

    if not list_params:
        # 리스트 파라미터가 없으면 원본 설정을 그대로 반환
        config['run_name'] = 'default'
        yield config
        return

    param_names = list(list_params.keys())
    param_values = list(list_params.values())
    
    for combo in itertools.product(*param_values):
        new_config = copy.deepcopy(config)
        combo_str_parts = []
        
        for i, param_name in enumerate(param_names):
            keys = param_name.split('.')
            temp_conf = new_config
            for key in keys[:-1]:
                temp_conf = temp_conf[key]
            
            # 원본 리스트를 현재 조합의 값으로 교체
            temp_conf[keys[-1]] = combo[i]
            
            # 실행 이름(run_name) 생성을 위한 문자열 조각
            key_for_path = keys[-1]
            val_for_path = str(combo[i])
            combo_str_parts.append(f"{key_for_path}={val_for_path}")
        
        # 최종 실행 이름 설정
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
