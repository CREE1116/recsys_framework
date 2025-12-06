import os
import re
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.stats import chi2_contingency, pearsonr

import argparse

# 프로젝트의 src 및 analysis 폴더를 python 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from analysis.utils import (
    load_item_metadata, 
    load_model_from_run, 
    get_analysis_output_path, 
    AnalysisReport
)
from analysis.visualizers import generate_interest_report_section

def run_full_analysis(exp_config):
    run_folder_path = exp_config['run_folder_path']
    top_n = exp_config.get('top_n', 15)

    print(f"\nRunning FULL interest analysis for: {run_folder_path}")

    # 1. 모델 로드
    model, data_loader, config = load_model_from_run(run_folder_path)
    if not model: return
    if not hasattr(model, 'attention_layer'):
        print("[Error] Not a CSAR model. Skipping.")
        return

    # 2. 리포트 준비
    run_name = os.path.basename(run_folder_path)
    output_path = get_analysis_output_path(config['dataset_name'], run_name)
    report = AnalysisReport(f"Deep Interest Analysis: {run_name}", output_path)
    
    report.add_section("1. Overview", level=2)
    report.add_text(f"- **Dataset:** `{config['dataset_name']}`")
    num_interests = model.num_interests
    report.add_text(f"- **Number of Interests:** `{num_interests}`")

    # 3. 데이터 준비 (메타데이터, 인기도)
    item_metadata_df = load_item_metadata(config['dataset_name'], config['data_path'])
    inv_item_map = {v: k for k, v in data_loader.item_map.items()}
    
    # [중요] 전체 아이템 인기도 계산 (학습 데이터 기준)
    global_item_counts = data_loader.df['item_id'].value_counts().to_dict()

    # 4. Attention 계산
    with torch.no_grad():
        all_item_embs = model.item_embedding.weight
        item_interests = model.attention_layer(all_item_embs)
        # Key Norm 계산 (CPU로 이동)
        keys_norm = torch.norm(model.attention_layer.interest_keys, p=2, dim=1).cpu()

    # 5. 루프 실행 및 통계 수집
    report.add_section("2. Analysis per Interest", level=2)
    collected_stats = []

    for k in range(num_interests):
        # 위에서 정의한 함수 호출
        stats = generate_interest_report_section(
            report=report,
            interest_k=k,
            item_interests=item_interests,
            inv_item_map=inv_item_map,
            metadata_df=item_metadata_df,
            global_item_counts=global_item_counts, # 인기도 전달
            top_n=top_n,
            interest_key_norm=keys_norm[k]         # Norm 전달
        )
        if stats:
            collected_stats.append(stats)

    # --------------------------------------------------------------------------------
    # 6. 글로벌 상관관계 분석 (Key Norm vs Popularity)
    # --------------------------------------------------------------------------------
    if collected_stats:
        report.add_section("3. Global Analysis: Adaptive Scaling Verification", level=2)
        
        df_stats = pd.DataFrame(collected_stats)
        
        # 상관계수 계산
        corr, p_val = pearsonr(df_stats['avg_popularity'], df_stats['norm'])
        
        report.add_text(f"- **Pearson Correlation (Norm vs Log-Pop):** `{corr:.4f}`")
        report.add_text(f"- **P-value:** `{p_val:.4e}`")
        
        # 해석 자동 생성
        if corr < -0.3:
            interpretation = "✅ **Result: Strong Negative Correlation.**\n\nThis confirms the **Adaptive Scaling Hypothesis**. The model automatically reduces the norm of interest keys associated with popular items (high magnitude embeddings) to maintain non-linearity in the Softplus activation function."
        elif corr > 0.3:
            interpretation = "⚠️ **Result: Positive Correlation.**\nThe model assigns larger norms to popular interests, potentially amplifying popularity bias."
        else:
            interpretation = "⏺️ **Result: No Significant Correlation.**\nThe model maintains consistent norms across different popularity levels, suggesting a robust representation independent of item frequency."
        
        report.add_text(interpretation)
        
        # 그래프 그리기
        plt.figure(figsize=(8, 6))
        sns.regplot(data=df_stats, x='avg_popularity', y='norm', scatter_kws={'s': 100, 'alpha': 0.7}, line_kws={'color': 'red'})
        
        # 각 점에 ID 표시
        for i, row in df_stats.iterrows():
            plt.text(row['avg_popularity'], row['norm'], str(int(row['k'])), fontsize=9)
            
        plt.title(f"Interest Key Norm vs. Avg Item Popularity (r={corr:.3f})")
        plt.xlabel("Average Item Popularity (Log Scale)")
        plt.ylabel("Interest Key L2 Norm")
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plot_filename = "global_norm_vs_pop.png"
        plt.savefig(os.path.join(output_path, plot_filename))
        plt.close()
        
        report.add_figure(plot_filename, "Correlation Plot: Key Norm vs Popularity")
        
        # 데이터 CSV 저장 (나중에 논문용 그래프 다시 그릴 때 유용)
        df_stats.to_csv(os.path.join(output_path, "interest_stats_summary.csv"), index=False)

    # 7. 리포트 저장
    report.save(filename="interest_analysis_report.md")
    print(f"Analysis complete. Report saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze Item Interests')
    parser.add_argument('--run_path', type=str, required=True, help='Path to the trained model run folder')
    parser.add_argument('--top_n', type=int, default=20, help='Number of top items to analyze per interest')
    
    args = parser.parse_args()

    if os.path.exists(args.run_path):
        run_full_analysis({
            'run_folder_path': args.run_path,
            'top_n': args.top_n
        })
    else:
        print(f"[Error] Path not found: {args.run_path}")