import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# 프로젝트의 src 및 analysis 폴더를 python 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_loader import DataLoader
from analysis.utils import get_analysis_output_path, AnalysisReport

def run_dataset_distribution_analysis(dataset_config_path):
    """
    데이터셋의 아이템 상호작용에 대한 롱테일 분포를 분석하고 리포트를 생성합니다.
    """
    print(f"\nRunning Dataset Distribution Analysis for: {dataset_config_path}")
    
    with open(dataset_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # [FIX] DataLoader requires 'evaluation' and 'train' keys to be present
    if 'evaluation' not in config:
        config['evaluation'] = {'validation_method': 'uni99', 'final_method': 'full'}
    if 'train' not in config:
        config['train'] = {}
    
    data_loader = DataLoader(config)
    dataset_name = config['dataset_name']
    
    # 분석 결과 저장 경로 및 리포트 객체 생성
    output_path = get_analysis_output_path(dataset_name)
    report = AnalysisReport(
        title=f"Dataset Distribution Analysis for '{dataset_name}'",
        output_path=output_path
    )

    if data_loader.df is None or data_loader.df.empty:
        report.add_text("DataFrame not found in DataLoader. Cannot perform analysis.")
        report.save("dataset_distribution_report.md")
        return
        
    item_counts = data_loader.df['item_id'].value_counts()
    
    report.add_section("1. Dataset Overview", level=2)
    overview_data = {
        "Metric": ["Total Unique Items", "Total Interactions", "Min Interactions per Item", "Max Interactions per Item", "Mean Interactions per Item"],
        "Value": [len(item_counts), item_counts.sum(), item_counts.min(), item_counts.max(), f"{item_counts.mean():.2f}"]
    }
    report.add_table(pd.DataFrame(overview_data))

    # 시각화
    report.add_section("2. Popularity Distribution", level=2)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f'Popularity Distribution Analysis for {dataset_name.upper()}', fontsize=20, fontweight='bold')

    # 상호작용 수 히스토그램
    counts = item_counts.values
    axes[0].hist(counts, bins=max(50, int(np.sqrt(len(counts)))), color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_title('Item Interaction Count Distribution', fontsize=16)
    axes[0].set_xlabel('Number of Interactions per Item')
    axes[0].set_ylabel('Number of Items (Log Scale)')
    axes[0].set_yscale('log')

    # 로렌츠 커브 및 지니 계수
    sorted_counts = np.sort(item_counts.values)
    cum_counts = np.cumsum(sorted_counts)
    num_items = len(counts)
    cum_items_percent = np.arange(1, num_items + 1) / num_items
    cum_counts_percent = cum_counts / cum_counts[-1]
    gini_coefficient = 1 - 2 * np.trapz(cum_counts_percent, cum_items_percent)

    axes[1].plot(cum_items_percent, cum_counts_percent, label='Lorenz Curve', color='orange', linewidth=3)
    axes[1].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Line of Equality')
    axes[1].fill_between(cum_items_percent, cum_counts_percent, color='orange', alpha=0.2)
    axes[1].set_title('Lorenz Curve of Item Popularity', fontsize=16)
    axes[1].set_xlabel('Cumulative Share of Items')
    axes[1].set_ylabel('Cumulative Share of Interactions')
    axes[1].legend(loc='upper left')
    
    stats_text = f'Gini Coefficient: {gini_coefficient:.4f}'
    axes[1].text(0.95, 0.05, stats_text, transform=axes[1].transAxes, fontsize=14, 
                 fontweight='bold', va='bottom', ha='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 결과 저장
    figure_filename = "popularity_distribution.png"
    save_path = os.path.join(output_path, figure_filename)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    
    report.add_text(f"The Gini coefficient for this dataset is **{gini_coefficient:.4f}**. A value closer to 1 indicates higher inequality in item popularity (a more pronounced long-tail).")
    report.add_figure(figure_filename, "Item Popularity Distribution and Lorenz Curve")
    
    report.save("dataset_distribution_report.md")


if __name__ == '__main__':
    DATASETS_TO_ANALYZE = [
        'configs/dataset/ml1m.yaml',
        'configs/dataset/ml100k.yaml',
        'configs/dataset/amazon_books.yaml',
        'configs/dataset/ml20m.yaml',
        'configs/dataset/amazon_music.yaml',
    ]

    for d_config in DATASETS_TO_ANALYZE:
        if os.path.exists(d_config):
            run_dataset_distribution_analysis(d_config)
        else:
            print(f"[Warning] Dataset config not found, skipping: {d_config}")
