import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import argparse

# 프로젝트의 src 및 analysis 폴더를 python 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_loader import DataLoader
from analysis.utils import get_analysis_output_path, AnalysisReport

def split_items_three_groups(df, item_col='item_id', head_ratio=0.2, tail_ratio=0.2):
    """
    Split items into Head, Mid, Tail groups based on popularity.
    - Head: Top `head_ratio`
    - Tail: Bottom `tail_ratio`
    - Mid: The rest
    """
    item_counts = df[item_col].value_counts()
    sorted_items = item_counts.index.tolist()
    n_items = len(sorted_items)
    
    head_cutoff = int(n_items * head_ratio)
    tail_cutoff = int(n_items * (1 - tail_ratio))
    
    # Ensure logical bounds
    tail_cutoff = max(head_cutoff, tail_cutoff)
    
    head_items = set(sorted_items[:head_cutoff])
    mid_items = set(sorted_items[head_cutoff:tail_cutoff])
    tail_items = set(sorted_items[tail_cutoff:])
    
    return head_items, mid_items, tail_items, item_counts

def analyze_dataset_groups(df, item_sets, item_counts, rating_col=None):
    """
    Calculate statistics for groups.
    item_sets: {'Head': set(...), 'Mid': set(...), 'Tail': set(...)}
    """
    stats = {}
    
    for group_name, items in item_sets.items():
        if not items:
            stats[f'{group_name.lower()}_avg_exposure'] = 0
            stats[f'{group_name.lower()}_total_interactions'] = 0
            stats[f'{group_name.lower()}_avg_rating'] = None
            continue

        # 1. Exposure
        group_counts = item_counts[item_counts.index.isin(items)]
        stats[f'{group_name.lower()}_avg_exposure'] = group_counts.mean()
        stats[f'{group_name.lower()}_total_interactions'] = group_counts.sum()
        
        # 2. Ratings
        if rating_col and rating_col in df.columns:
            group_ratings = df[df['item_id'].isin(items)][rating_col]
            stats[f'{group_name.lower()}_avg_rating'] = group_ratings.mean()
        else:
            stats[f'{group_name.lower()}_avg_rating'] = None
            
    return stats

def visualize_comparison(stats, output_path, dataset_name):
    """
    Visualize Head vs Mid vs Tail stats.
    """
    # 1. Exposure Comparison
    plt.figure(figsize=(10, 6))
    groups = ['Head', 'Mid', 'Tail']
    exposures = [
        stats.get('head_avg_exposure', 0), 
        stats.get('mid_avg_exposure', 0), 
        stats.get('tail_avg_exposure', 0)
    ]
    
    # [Head: Red, Mid: Orange, Tail: Blue]
    colors = ['#e74c3c', '#f39c12', '#3498db']
    
    sns.barplot(x=groups, y=exposures, palette=colors)
    plt.title(f'Average Exposure (Interactions) - {dataset_name}', fontsize=16)
    plt.ylabel('Avg Interaction Count')
    plt.yscale('log') # Log scale helps typically
    
    # Add values on bars
    for i, v in enumerate(exposures):
        plt.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    plt.savefig(os.path.join(output_path, 'avg_exposure_comparison.png'), dpi=150)
    plt.close()
    
    # 2. Rating Comparison (if available)
    if stats.get('head_avg_rating') is not None:
        plt.figure(figsize=(10, 6))
        ratings = [
            stats.get('head_avg_rating', 0), 
            stats.get('mid_avg_rating', 0), 
            stats.get('tail_avg_rating', 0)
        ]
        
        # Scale y-axis narrowly if ratings are close (e.g., 3.0 ~ 5.0)
        valid_ratings = [r for r in ratings if r is not None and r > 0]
        if valid_ratings:
            min_r, max_r = min(valid_ratings), max(valid_ratings)
            plt.ylim(max(0, min_r - 0.5), min(5.0, max_r + 0.5))
        
        sns.barplot(x=groups, y=ratings, palette=colors)
        plt.title(f'Average Rating - {dataset_name}', fontsize=16)
        plt.ylabel('Avg Rating')
        
        for i, v in enumerate(ratings):
            if v is not None:
                plt.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
        plt.savefig(os.path.join(output_path, 'avg_rating_comparison.png'), dpi=150)
        plt.close()

def run_comprehensive_analysis(config_path, output_subfolder='long_tail_stats'):
    print(f"Running Comprehensive Long-Tail Analysis for: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    if 'train' not in config: config['train'] = {}
    if 'evaluation' not in config: config['evaluation'] = {}
    
    try:
        data_loader = DataLoader(config)
    except Exception as e:
        print(f"[Skipping] Failed to load data for {config_path}: {e}")
        return

    df = data_loader.df
    if df is None:
        print(f"[Skipping] No dataframe found for {config_path}")
        return
        
    dataset_name = config['dataset_name']
    
    # Analyze Output Path
    output_path = get_analysis_output_path(dataset_name, output_subfolder)
    report = AnalysisReport(f"Comprehensive Long-Tail Analysis (3-Group): {dataset_name}", output_path)
    
    # 1. Split Head/Mid/Tail
    # Default: Top 20% Head, Bottom 20% Tail, Middle 60% Mid
    head_ratio = 0.2
    tail_ratio = 0.2
    head_items, mid_items, tail_items, item_counts = split_items_three_groups(df, head_ratio=head_ratio, tail_ratio=tail_ratio)
    
    report.add_text(f"**Split Criteria**:")
    report.add_text(f"- **Head (Top {head_ratio*100:.0f}%)**: {len(head_items)} items")
    report.add_text(f"- **Mid (Middle {(1-head_ratio-tail_ratio)*100:.0f}%)**: {len(mid_items)} items")
    report.add_text(f"- **Tail (Bottom {tail_ratio*100:.0f}%)**: {len(tail_items)} items")
    
    # 2. Calculate Stats
    rating_col = None
    if 'rating' in df.columns:
        rating_col = 'rating'
    elif len(df.columns) >= 3 and pd.api.types.is_numeric_dtype(df.iloc[:, 2]):
        rating_col = df.columns[2]
        
    item_sets = {'Head': head_items, 'Mid': mid_items, 'Tail': tail_items}
    stats = analyze_dataset_groups(df, item_sets, item_counts, rating_col)
    
    # 3. Report Results
    report.add_section("1. Exposure (Interaction Count) Analysis")
    
    exposure_df = pd.DataFrame({
        "Group": ["Head", "Mid", "Tail"],
        "Avg Exposure": [stats['head_avg_exposure'], stats['mid_avg_exposure'], stats['tail_avg_exposure']],
        "Total Interactions": [stats['head_total_interactions'], stats['mid_total_interactions'], stats['tail_total_interactions']]
    })
    report.add_table(exposure_df)
    
    # Interaction Share logic
    total_interactions = exposure_df['Total Interactions'].sum()
    head_share = stats['head_total_interactions'] / total_interactions
    report.add_text(f"\n> **Insight**: Head items (Top {head_ratio*100:.0f}%) account for **{head_share*100:.1f}%** of all interactions.")

    # 4. Rating Analysis
    report.add_section("2. Rating Analysis")
    if stats['head_avg_rating'] is not None:
        rating_df = pd.DataFrame({
            "Group": ["Head", "Mid", "Tail"],
            "Avg Rating": [stats['head_avg_rating'], stats['mid_avg_rating'], stats['tail_avg_rating']]
        })
        report.add_table(rating_df)
    else:
        report.add_text("No rating column found in this dataset.")

    # 5. Visualize
    visualize_comparison(stats, output_path, dataset_name)
    report.add_figure("avg_exposure_comparison.png", "Average Exposure Comparison")
    if stats['head_avg_rating'] is not None:
        report.add_figure("avg_rating_comparison.png", "Average Rating Comparison")
        
    report.save("comprehensive_long_tail_report.md")
    print(f"Report saved at {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to dataset config')
    args = parser.parse_args()
    
    if args.config:
        if os.path.exists(args.config):
            run_comprehensive_analysis(args.config)
        else:
            print(f"Config not found: {args.config}")
    else:
        # Default run
        print("No config provided. Please implement bulk run logic outside or provide config.")

