import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict

def analyze_results(base_path, dataset='ml-100k'):
    target_dir = os.path.join(base_path, 'trained_model', dataset)
    if not os.path.exists(target_dir):
        print(f"Error: Path {target_dir} does not exist.")
        return

    # metrics[model_variant][metric_name] = [value1, value2, ...]
    metrics_by_group = defaultdict(lambda: defaultdict(list))
    
    # 1. 디렉토리 목록 스캔 (BEST_aspire_... 또는 BEST_cheby_aspire_...)
    subdirs = [d for d in os.listdir(target_dir) if d.startswith('BEST_')]
    
    for subdir in subdirs:
        # 모델명 및 변종 추출을 위한 파싱
        # 예: BEST_aspire_seed_42_auto_bias -> Model: aspire, Variant: auto_bias
        # 예: BEST_cheby_aspire_seed_42_auto_bias -> Model: cheby_aspire, Variant: auto_bias
        
        if 'aspire' not in subdir:
            continue
            
        parts = subdir.split('_')
        # 시드(seed_XX) 위치 찾기
        try:
            seed_idx = next(i for i, p in enumerate(parts) if p == 'seed')
            model_name = "_".join(parts[1:seed_idx])
            variant = "_".join(parts[seed_idx+2:])
            group_key = f"{model_name} [{variant}]"
        except (StopIteration, IndexError):
            continue
        
        metrics_file = os.path.join(target_dir, subdir, 'final_metrics.json')
        if not os.path.exists(metrics_file):
            continue
            
        with open(metrics_file, 'r') as f:
            try:
                data = json.load(f)
                for m_name, m_val in data.items():
                    metrics_by_group[group_key][m_name].append(m_val)
            except json.JSONDecodeError:
                continue

    if not metrics_by_group:
        print("No valid results found. Check if paths like 'BEST_aspire_...' exist.")
        return

    # 2. 결과 집계
    summary = []
    target_metrics = [
        'NDCG@10', 'Recall@10', 'Coverage@10',
        'NDCG@20', 'Recall@20', 'Coverage@20',
        'LongTailCoverage@20', 'Novelty@20'
    ]

    for group, metrics_dict in metrics_by_group.items():
        row = {'Group': group, 'Seeds': len(next(iter(metrics_dict.values())))}
        for m_name in target_metrics:
            if m_name in metrics_dict:
                vals = metrics_dict[m_name]
                row[f'{m_name}'] = np.mean(vals)
        summary.append(row)

    df = pd.DataFrame(summary).set_index('Group').sort_index()
    
    # 3. 출력
    print(f"\n[Model Variant Comparison: {dataset}]")
    print("=" * 120)
    print(df.to_string())
    
    # 4. 상대적 성능 차이 분석 (auto_bias vs auto_compromise)
    models = sorted(list(set(g.split(' [')[0] for g in metrics_by_group.keys())))
    
    for model in models:
        bias_key = f"{model} [auto_bias]"
        comp_key = f"{model} [auto_compromise]"
        
        if bias_key in df.index and comp_key in df.index:
            print(f"\n[Variant Comparison for {model}: auto_bias -> auto_compromise]")
            print("-" * 85)
            bias = df.loc[bias_key]
            comp = df.loc[comp_key]
            for m_name in target_metrics:
                if m_name in df.columns:
                    val_bias = bias[m_name]
                    val_comp = comp[m_name]
                    if val_bias != 0:
                        diff = (val_comp - val_bias) / val_bias * 100
                        status = "Gain" if diff >= 0 else "Loss"
                        print(f"{m_name:20}: {val_bias:.4f} -> {val_comp:.4f} ({diff:+.2f}%) [{status}]")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-100k', help='Dataset name')
    args = parser.parse_args()
    
    base_path = os.getcwd()
    analyze_results(base_path, dataset=args.dataset)
