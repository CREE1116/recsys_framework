import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy.stats import pearsonr
import sys

# 프로젝트의 src 및 analysis 폴더를 python 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.utils import (
    load_item_metadata, 
    load_model_from_run, 
    get_analysis_output_path, 
    AnalysisReport
)

def plot_user_long_tail_profile(output_path, user_idx, user_weights, target_item, best_item, worst_item):
    n_interests = len(user_weights)
    x = np.arange(n_interests)
    user_avg = user_weights.mean()

    fig, axes = plt.subplots(4, 1, figsize=(16, 20), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, n_interests))
    
    axes[0].bar(x, user_weights, color=colors, edgecolor='black', alpha=0.8)
    axes[0].set_title(f'User #{user_idx} - Interest Profile (Avg: {user_avg:.3f})', fontsize=16, fontweight='bold')
    axes[0].set_ylabel('Weight', fontsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    item_types = [
        {'data': target_item, 'type': 'Long-tail', 'color': '#e74c3c', 'ax_idx': 1},
        {'data': best_item, 'type': 'Best-Match', 'color': '#2ecc71', 'ax_idx': 2},
        {'data': worst_item, 'type': 'Worst-Match', 'color': '#95a5a6', 'ax_idx': 3},
    ]

    for item_info in item_types:
        ax = axes[item_info['ax_idx']]
        data = item_info['data']
        item_weights = data['weights']
        item_avg = item_weights.mean()
        
        ax.bar(x, item_weights, color=item_info['color'], edgecolor='black', alpha=0.7, label=f'Item (Avg: {item_avg:.3f})')
        ax.set_ylabel('Item Weight', fontsize=12, color=item_info['color'])
        ax.tick_params(axis='y', labelcolor=item_info['color'])
        ax_twin = ax.twinx()
        ax_twin.plot(x, user_weights, 'o-', color='#3498db', linewidth=2, markersize=8, label=f'User (Avg: {user_avg:.3f})', alpha=0.8)
        ax_twin.set_ylabel('User Weight', fontsize=12, color='#3498db')
        ax_twin.tick_params(axis='y', labelcolor='#3498db')
        ax.set_title(f'{item_info["type"]}: {data["title"][:60]} (Score: {data["score"]:.3f})', fontsize=14, fontweight='bold')
        ax.text(0.98, 0.98, f'r = {data["correlation"]:.3f}', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_twin.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=11, loc='upper left')

    axes[-1].set_xlabel('Interest Index', fontsize=14, fontweight='bold')
    axes[-1].set_xticks(x)
    fig.suptitle(f'User #{user_idx} Recommendation Profile vs. Long-tail Item', fontsize=18, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0.01, 1, 0.98])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

def run_long_tail_model_analysis(exp_config):
    run_folder_path = exp_config['run_folder_path']
    max_cases_to_find = exp_config.get('max_cases_to_find', 5)
    top_k_for_rec = exp_config.get('top_k_for_rec', 50)
    long_tail_percentage = exp_config.get('long_tail_percentage', 0.2)

    print(f"\nRunning Long-tail Model Analysis for: {run_folder_path}")
    
    model, data_loader, config = load_model_from_run(run_folder_path)
    if not model: return

    run_name = os.path.basename(run_folder_path)
    output_path = get_analysis_output_path(config['dataset_name'], run_name)
    report = AnalysisReport(f"Long-tail Recommendation Analysis: {run_name}", output_path)

    item_metadata_df = load_item_metadata(config['dataset_name'], config['data_path'])
    inv_item_map = {v: k for k, v in data_loader.item_map.items()}

    item_counts = data_loader.df['item_id'].value_counts()
    num_long_tail = int(data_loader.n_items * long_tail_percentage)
    long_tail_item_ids = set(item_counts.sort_values(ascending=True).head(num_long_tail).index.tolist())
    
    if not long_tail_item_ids:
        report.add_text("No long-tail items found based on the criteria. Skipping.")
        report.save("long_tail_analysis_report.md")
        return

    print("Pre-calculating all user/item weights and scores...")
    with torch.no_grad():
        if hasattr(model, 'attention_layer'):
            all_user_weights = model.attention_layer(model.user_embedding.weight).cpu()
            all_item_weights = model.attention_layer(model.item_embedding.weight).cpu()
        else:
            all_user_weights = model.user_embedding.weight.cpu()
            all_item_weights = model.item_embedding.weight.cpu()
        all_scores_matrix = torch.matmul(all_user_weights, all_item_weights.T)
    print("Pre-calculation complete.")
    
    report.add_section("Long-tail Recommendation Analysis Cases", level=2)
    report.add_text(f"Finding up to **{max_cases_to_find}** cases where a long-tail item (bottom {long_tail_percentage*100}%) appears in a user's Top-{top_k_for_rec} recommendations.")
    
    found_cases = 0
    # 유저를 랜덤하게 섞어서 매번 다른 케이스를 찾도록 함
    shuffled_user_indices = list(range(data_loader.n_users))
    random.shuffle(shuffled_user_indices)

    for user_idx in shuffled_user_indices:
        if found_cases >= max_cases_to_find:
            break

        scores_for_user = all_scores_matrix[user_idx, :]
        
        # 학습에 사용된 아이템은 추천에서 제외
        if user_idx in data_loader.user_history:
            scores_for_user[list(data_loader.user_history[user_idx])] = -torch.inf
            
        top_k_scores, top_k_indices = torch.topk(scores_for_user, k=top_k_for_rec)
        
        for rank, item_id in enumerate(top_k_indices.tolist()):
            if item_id in long_tail_item_ids:
                found_cases += 1
                
                original_item_id = inv_item_map[item_id]
                item_title = item_metadata_df.loc[original_item_id, 'title']
                
                report.add_section(f"Case #{found_cases}: User {user_idx} -> Item {original_item_id} ('{item_title}')", level=3)
                report.add_text(f"This long-tail item was found at **rank {rank+1}** in the user's Top-{top_k_for_rec} recommendation list.")

                # 상세 분석
                user_weights_np = all_user_weights[user_idx].numpy()
                best_score, best_item_id = torch.max(scores_for_user, dim=0)
                worst_score, worst_item_id = torch.min(scores_for_user, dim=0)
                
                target_item_data = {'title': item_title, 'weights': all_item_weights[item_id].numpy(), 'score': scores_for_user[item_id].item(), 'correlation': pearsonr(user_weights_np, all_item_weights[item_id].numpy())[0]}
                best_item_data = {'title': item_metadata_df.loc[inv_item_map[best_item_id.item()], 'title'], 'weights': all_item_weights[best_item_id.item()].numpy(), 'score': best_score.item(), 'correlation': pearsonr(user_weights_np, all_item_weights[best_item_id.item()].numpy())[0]}
                worst_item_data = {'title': item_metadata_df.loc[inv_item_map[worst_item_id.item()], 'title'], 'weights': all_item_weights[worst_item_id.item()].numpy(), 'score': worst_score.item(), 'correlation': pearsonr(user_weights_np, all_item_weights[worst_item_id.item()].numpy())[0]}

                figure_filename = f"case_{found_cases}_user_{user_idx}_item_{original_item_id}.png"
                plot_output_path = os.path.join(output_path, figure_filename)
                
                plot_user_long_tail_profile(plot_output_path, user_idx, user_weights_np, target_item_data, best_item_data, worst_item_data)
                report.add_figure(figure_filename, f"Detailed profile for Case #{found_cases}")

                if found_cases >= max_cases_to_find:
                    break

    if found_cases == 0:
        report.add_text("Could not find any cases of long-tail items in user recommendations within the searched user sample.")

    report.save("long_tail_model_analysis_report.md")


if __name__ == '__main__':
    EXPERIMENTS_TO_RUN = [
        {
            'run_folder_path': '/Users/leejongmin/code/recsys_framework/trained_model/amazon_books/csar-bpr-ce__temperature=0.8',
            'max_cases_to_find': 5,
            'top_k_for_rec': 50,
            'long_tail_percentage': 0.2,
        },
    ]

    for exp_config in EXPERIMENTS_TO_RUN:
        if os.path.exists(exp_config['run_folder_path']):
            run_long_tail_model_analysis(exp_config)
        else:
            print(f"[Error] Experiment folder not found: {exp_config['run_folder_path']}")
