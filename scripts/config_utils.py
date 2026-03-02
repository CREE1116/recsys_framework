"""
Config 병합 유틸리티.
마스터 evaluation.yaml → dataset config → model config 순서로 deep merge.
"""
import os
import yaml
import copy


def deep_merge(base, override):
    """base에 override를 deep merge. override 값이 우선."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_eval_master_config(eval_config_path=None):
    """마스터 evaluation config 로드. 경로 미지정 시 기본 configs/evaluation.yaml."""
    if eval_config_path is None:
        # 프로젝트 루트 기준 기본 경로
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        default_path = os.path.join(project_root, 'configs', 'evaluation.yaml')
        if os.path.exists(default_path):
            eval_config_path = default_path

    if eval_config_path and os.path.exists(eval_config_path):
        with open(eval_config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}


def merge_all_configs(dataset_config, model_config, eval_config_path=None):
    """마스터 eval → dataset → model 순서로 3단계 deep merge."""
    eval_config = load_eval_master_config(eval_config_path)
    config = deep_merge(eval_config, dataset_config)
    config = deep_merge(config, model_config)
    return config
