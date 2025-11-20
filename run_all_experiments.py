import glob
import subprocess
import os
import json
import pandas as pd
import argparse
import re
import yaml

def run_all_searches(model_config_files, dataset_config_path):
    """
    지정된 모델 및 데이터셋 설정에 대해 그리드 서치를 실행합니다.
    """
    print("="*80)
    print(f"Starting grid search for dataset: {os.path.basename(dataset_config_path)}")
    print("="*80)

    for config_file in model_config_files:
        print(f"\n--- Running grid search for model: {os.path.basename(config_file)} ---\n")
        command = [
            '.venv/bin/python', 
            'grid_search.py', 
            '--model_config', config_file,
            '--dataset_config', dataset_config_path # 데이터셋 설정 전달
        ]
        
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running grid search for {config_file} with {dataset_config_path}: {e}")
            continue

    print("="*80)
    print("All specified grid searches completed.")
    print("="*80)

def parse_experiment_name(exp_name):
    """
    'model__param1=val1_param2=val2' 형식의 폴더 이름에서 파라미터를 파싱합니다.
    """
    if '__' in exp_name:
        model_name, params_str = exp_name.split('__', 1)
    else:
        model_name = exp_name
        params_str = ''
        
    params = {'model': model_name}
    
    if params_str:
        for part in params_str.split('_'):
            if '=' in part:
                key, value = part.split('=', 1)
                try:
                    # 값에 소수점이 있으면 float, 아니면 int, 둘 다 아니면 string
                    if '.' in value:
                        params[key] = float(value)
                    else:
                        params[key] = int(value)
                except ValueError:
                    params[key] = value
    return params

def aggregate_results(results_base_dir, output_csv):
    """
    모든 실험 결과를 취합하여 하나의 CSV 파일로 저장합니다.
    """
    print("\n" + "="*80)
    print(f"Aggregating results from: {results_base_dir}")
    print("="*80)

    metric_files = glob.glob(os.path.join(results_base_dir, '**', 'final_metrics.json'), recursive=True)
    
    if not metric_files:
        print("No 'final_metrics.json' files found. Skipping aggregation.")
        return

    all_results = []

    for metric_file in metric_files:
        try:
            exp_dir = os.path.dirname(metric_file)
            exp_name = os.path.basename(exp_dir)
            params = parse_experiment_name(exp_name)
            
            with open(metric_file, 'r') as f:
                metrics = json.load(f)
            
            row = {**params, **metrics}
            all_results.append(row)

        except Exception as e:
            print(f"Error processing file {metric_file}: {e}")

    df = pd.DataFrame(all_results)
    
    if not df.empty:
        param_cols = sorted([col for col in df.columns if '@' not in col and col != 'model'])
        metric_cols = sorted([col for col in df.columns if '@' in col])
        
        final_cols = ['model'] + param_cols + metric_cols
        final_cols = [col for col in final_cols if col in df.columns]
        
        df = df[final_cols]

    df.to_csv(output_csv, index=False)
    print(f"Successfully aggregated {len(all_results)} results into '{output_csv}'")


if __name__ == '__main__':
    # 실행할 모델 설정 파일 목록을 여기에 정의합니다.
    model_config_files_to_run = [
        # 'configs/model/csar_deep.yaml',
        # 'configs/model/rerank_lightgcn.yaml',
        # 'configs/model/neumf.yaml',
        'configs/model/item_knn.yaml',
        'configs/model/most_popular.yaml',
        'configs/model/csar_gate.yaml',
        'configs/model/csar_bpr.yaml',
        'configs/model/csar_r_bpr.yaml',
        'configs/model/csar_r_contrastive.yaml',
        # 'configs/model/csar_contrastive.yaml',
        # 'configs/model/csar.yaml',
        # 'configs/model/csar_r.yaml',
        # 'configs/model/csar_r_softmax.yaml',
        'configs/model/mf.yaml',
        # 'configs/model/lightgcn.yaml',
    ]

    parser = argparse.ArgumentParser(description="Run experiments for specified models and a dataset.")
    parser.add_argument('--dataset_config', type=str, required=True,
                        help='Path to the dataset configuration file (e.g., configs/dataset/ml-100k.yaml).')
    parser.add_argument('--results_dir', type=str, default='trained_model',
                        help='Base directory where experiment results are stored.')
    parser.add_argument('--output_file', type=str, default='results_summary.csv',
                        help='Path to the output CSV file for results summary.')
    
    args = parser.parse_args()

    # 1. 지정된 설정 파일들에 대해 그리드 서치 실행
    run_all_searches(model_config_files_to_run, args.dataset_config)
    
    # 2. 결과 취합
    # dataset_config 파일을 로드하여 dataset_name을 추출
    with open(args.dataset_config, 'r') as f:
        loaded_dataset_config = yaml.safe_load(f)
    dataset_name = loaded_dataset_config['dataset_name']

    results_path_for_dataset = os.path.join(args.results_dir, dataset_name)
    aggregate_results(results_path_for_dataset, args.output_file)
