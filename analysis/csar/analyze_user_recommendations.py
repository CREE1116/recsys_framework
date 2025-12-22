# Visualize User Recommendations with Coherence & Interest Matching

import os
import argparse
import sys
import torch
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import math

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from analysis.utils import load_model_from_run
from analysis.csar.detail_interests import load_augmented_metadata

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def math_ceil(x):
    return int(math.ceil(x))

def extract_genres(df):
    all_genres = []
    for genres_str in df['Genres']:
        if isinstance(genres_str, str):
            if genres_str.startswith('['):
                import ast
                try:
                    genres = ast.literal_eval(genres_str)
                    all_genres.extend(genres)
                except:
                    pass
            else:
                genres = genres_str.split('|')
                all_genres.extend(genres)
        elif isinstance(genres_str, list):
             all_genres.extend(genres_str)
    return Counter(all_genres)

def extract_years(df):
    all_years = []
    for year in df['Year']:
        try:
             y = str(year).strip()
             if y and y.isdigit():
                 all_years.append(int(y))
        except:
             pass
    return Counter(all_years)

def calculate_coherence(prof1, prof2):
    norm1 = np.linalg.norm(prof1)
    norm2 = np.linalg.norm(prof2)
    if norm1 > 0 and norm2 > 0:
        return np.dot(prof1, prof2) / (norm1 * norm2)
    return 0.0

def plot_user_profile(ax, user_prof, user_id, view_name, is_dual_concat=False):
    K = len(user_prof)
    x = np.arange(K)
    
    if is_dual_concat:
        half_k = K // 2
        colors = ['tab:blue'] * half_k + ['tab:red'] * math_ceil(K - half_k)
        colors = colors[:K]
        
        ax.bar(x, user_prof, color=colors, alpha=0.7, label='User Interest')
        ax.axvline(half_k - 0.5, color='black', linestyle='-', linewidth=1.5)
        
        y_max = np.max(np.abs(user_prof)) if np.max(np.abs(user_prof)) > 0 else 1.0
        ax.text(half_k * 0.5, y_max*1.1, "Like View (+)", ha='center', color='blue', fontweight='bold')
        ax.text(half_k * 1.5, y_max*1.1, "Dislike View (-)", ha='center', color='red', fontweight='bold')
    else:
        colors = ['tab:blue' if v >= 0 else 'tab:red' for v in user_prof]
        ax.bar(x, user_prof, color=colors, alpha=0.7, label='User Interest')

    ax.set_ylabel('Interest Weight')
    ax.set_title(f"User {user_id} Interest Profile ({view_name})")
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', linewidth=0.8)
    
    if K > 60:
        ax.set_xticks(x[::10])
    else:
        ax.set_xticks(x)

def plot_item_with_user_overlay(ax, item_prof, user_prof, title, metrics, item_color='tab:orange', is_contribution=False):
    K = len(item_prof)
    x = np.arange(K)
    
    # 1. Bar Chart
    if is_contribution:
        half_k = K // 2
        colors = ['tab:blue'] * half_k + ['tab:red'] * (K - half_k)
        
        ax.bar(x, item_prof, color=colors, alpha=0.6, label='Contribution')
        ax.axvline(half_k - 0.5, color='black', linestyle='-', linewidth=1.5)
        
        ylabel = 'Contrib (+Like, -Dislike)'
        overlay_label = 'User Vector (Ref)'
        
        info_text = (
            f"Total Score: {metrics.get('total_score', 0):.4f}\n"
            f"-----------------\n"
            f"Like Score: {metrics.get('score_like', 0):.4f}\n"
            f"Dislike Score: {metrics.get('score_dislike', 0):.4f}\n"
            f"-----------------\n"
            f"Like Coherence: {metrics.get('coh_like', 0):.4f}\n"
            f"Dislike Coherence: {metrics.get('coh_dislike', 0):.4f}"
        )
    else:
        item_norm = np.linalg.norm(item_prof)
        dot_product = np.dot(user_prof, item_prof)
        
        ax.bar(x, item_prof, color=item_color, alpha=0.6, label='Item Interest')
        ylabel = 'Item Weight'
        overlay_label = 'User Profile (Overlay)'
        
        info_text = (
            f"Coherence (Dir): {metrics.get('coherence', 0):.4f}\n"
            f"Item Norm (Mag): {item_norm:.4f}\n"
            f"Dot Product (Score): {dot_product:.4f}"
        )

    # 2. User Line Overlay
    ax2 = ax.twinx()
    ax2.plot(x, user_prof, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label=overlay_label)
    ax2.set_ylabel('User Weight', color='black')
    
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}")
    ax.set_xlabel('Interest ID (Concatenated)')
    ax.axhline(0, color='black', linewidth=0.8)

    # Annotation
    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    ax.text(0.98, 0.95, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    if K > 60:
        ax.set_xticks(x[::10])
    else:
        ax.set_xticks(x)
    
    # Legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left', fontsize='small')

def plot_best_worst_matching(user_prof, best_prof, worst_prof, best_meta, worst_meta, output_path, user_id, view_name="Interest", is_dual_net=False):
    """
    3-Row Plot.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
    
    # Row 1: User
    plot_user_profile(ax1, user_prof, user_id, view_name, is_dual_concat=is_dual_net)
    
    # Row 2: Best
    best_metrics = best_meta['metrics'] if is_dual_net else {'coherence': calculate_coherence(user_prof, best_prof)}
    best_title = f"[Best] {best_meta['title']}"
    plot_item_with_user_overlay(ax2, best_prof, user_prof, best_title, best_metrics, 
                                item_color='tab:orange', is_contribution=is_dual_net)
    
    # Row 3: Worst
    worst_metrics = worst_meta['metrics'] if is_dual_net else {'coherence': calculate_coherence(user_prof, worst_prof)}
    worst_title = f"[Worst] {worst_meta['title']}"
    plot_item_with_user_overlay(ax3, worst_prof, user_prof, worst_title, worst_metrics, 
                                item_color='tab:green', is_contribution=is_dual_net)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def report_diversity_diagnosis(user_profs, item_profs, best_item_ids, report_path):
    if len(user_profs) < 2: return
    user_matrix = np.array(user_profs)
    item_matrix = np.array(item_profs)
    
    # 1. Similarity
    user_sim = cosine_similarity(user_matrix)
    avg_user_sim = np.mean(user_sim[np.triu_indices_from(user_sim, k=1)])
    
    # 2. Unique Items
    unique_items = len(set(best_item_ids))
    total_users = len(best_item_ids)
    uniq_ratio = unique_items / total_users

    # 3. Dominant Interest (Feature Collapse)
    # Take absolute mean profile to find active dimensions
    mean_profile = np.mean(np.abs(user_matrix), axis=0)
    # Get top 3 indices
    top_indices = np.argsort(-mean_profile)[:3]
    top_values = mean_profile[top_indices]
    
    # 4. Norm Analysis
    norms = np.linalg.norm(user_matrix, axis=1)
    norm_mean = np.mean(norms)
    norm_std = np.std(norms)
    norm_cv = norm_std / norm_mean if norm_mean > 0 else 0

    with open(report_path, "a") as f:
        f.write("\n# Deep Collapse Diagnosis\n")
        f.write(f"- **Avg User Sim**: `{avg_user_sim:.4f}` (High = Collapse)\n")
        f.write(f"- **Unique Items**: `{unique_items}/{total_users}` ({uniq_ratio*100:.1f}%)\n")
        f.write(f"- **Dominant Interests**: Indices {top_indices} (Avg w: {top_values})\n")
        f.write(f"- **Norm CV (Magnitude Var)**: `{norm_cv:.4f}`\n\n")
        
        f.write("> **Interpretation**:\n")
        f.write("> - Low Unique Items + High Sim = **Population Collapse** (Everyone gets Star Wars).\n")
        f.write("> - Dominant indices with huge weights = **Feature Collapse** (One interest rules all).\n")
        f.write("> - High Norm CV + High Sim = **Norm-Driven Distinction** (Only magnitude matters).\n")

def analyze_user_recs(exp_config):
    run_folder_path = exp_config['run_folder_path']
    num_users = exp_config.get('num_users', 3)
    user_ids = exp_config.get('user_ids', None)

    print(f"Analyzing recommendations for {run_folder_path}...")
    exp_name = os.path.basename(run_folder_path)
    
    if "trained_model" in run_folder_path:
        output_base = os.path.abspath(run_folder_path.replace("trained_model", "output"))
    else:
        output_base = os.path.abspath(run_folder_path)
        
    print(f"Output Base: {output_base}")
    analysis_dir = ensure_dir(os.path.join(output_base, "user_analysis"))
    print(f"Analysis Dir: {analysis_dir}")
    if not os.path.exists(analysis_dir):
        print(f"WARNING: Directory {analysis_dir} was NOT created.")
        os.makedirs(analysis_dir, exist_ok=True)

    
    model, data_loader, config = load_model_from_run(run_folder_path)
    if not model: return
    model.eval()

    report_path = os.path.join(analysis_dir, "interest_match_report.md")
    with open(report_path, "w") as f:
        f.write(f"# Best vs Worst Interest Matching: {exp_name}\n\n")

    metadata_df = load_augmented_metadata(config['data_path'], config['dataset_name'])
    inv_user_map = {v: k for k, v in data_loader.user_map.items()}
    inv_item_map = {v: k for k, v in data_loader.item_map.items()}

    if user_ids:
        target_users = []
        for uid in user_ids:
            if str(uid) in data_loader.user_map:
                target_users.append(data_loader.user_map[str(uid)])
            elif int(uid) in data_loader.user_map:
                 target_users.append(data_loader.user_map[int(uid)])
    else:
        all_users = list(data_loader.user_map.values())
        target_users = random.sample(all_users, min(num_users, len(all_users)))

    collected_user_profs = []
    collected_item_profs = []
    collected_best_ids = []

    with torch.no_grad():
        for internal_uid in target_users:
            original_uid = inv_user_map[internal_uid]
            print(f"Processing User {original_uid}...")
            
            u_tensor = torch.tensor([internal_uid]).to(model.device)
            att_out = model.attention_layer(model.user_embedding(u_tensor))
            is_dual = isinstance(att_out, tuple)
            prediction = model(u_tensor)
            
            user_hist = []
            if hasattr(data_loader, 'df'):
                u_hist_df = data_loader.df[data_loader.df['user_id'] == str(original_uid)]
                if u_hist_df.empty: u_hist_df = data_loader.df[data_loader.df['user_id'] == int(original_uid)]
                user_hist = u_hist_df['item_id'].tolist()
            
            internal_hist_ids = [data_loader.item_map[str(iid)] for iid in user_hist if str(iid) in data_loader.item_map]
            prediction[0, internal_hist_ids] = -float('inf') 
            
            best_scores, best_indices = torch.topk(prediction, 1)
            best_idx = best_indices.item()
            collected_best_ids.append(best_idx)
            
            pred_for_min = prediction.clone()
            pred_for_min[0, internal_hist_ids] = float('inf')
            worst_scores, worst_indices = torch.topk(pred_for_min, 1, largest=False)
            worst_idx = worst_indices.item()
            
            def get_meta(idx, score, metrics=None):
                iid = inv_item_map.get(idx)
                title = f"Item {iid}"
                if iid in metadata_df.index:
                    title = metadata_df.loc[iid].get('title', title)
                return {'id': iid, 'title': title, 'score': score, 'metrics': metrics}
            
            i_best_out = model.attention_layer(model.item_embedding(torch.tensor([best_idx]).to(model.device)))
            i_worst_out = model.attention_layer(model.item_embedding(torch.tensor([worst_idx]).to(model.device)))

            if is_dual:
                u_like = att_out[0].cpu().numpy().flatten()
                u_dislike = att_out[1].cpu().numpy().flatten()
                u_concat = np.concatenate([u_like, -u_dislike])
                
                def calc_dual_metrics(u_l, u_d, i_l, i_d):
                    score_l = np.sum(u_l * i_l)
                    score_d = np.sum(u_d * i_d)
                    return {
                        'coh_like': calculate_coherence(u_l, i_l),
                        'coh_dislike': calculate_coherence(u_d, i_d),
                        'score_like': score_l,
                        'score_dislike': score_d,
                        'total_score': score_l - score_d
                    }
                
                ib_like = i_best_out[0].cpu().numpy().flatten()
                ib_dislike = i_best_out[1].cpu().numpy().flatten()
                iw_like = i_worst_out[0].cpu().numpy().flatten()
                iw_dislike = i_worst_out[1].cpu().numpy().flatten()
                
                best_contrib = np.concatenate([ (u_like * ib_like), -(u_dislike * ib_dislike) ])
                worst_contrib = np.concatenate([ (u_like * iw_like), -(u_dislike * iw_dislike) ])
                
                best_metrics = calc_dual_metrics(u_like, u_dislike, ib_like, ib_dislike)
                worst_metrics = calc_dual_metrics(u_like, u_dislike, iw_like, iw_dislike)
                
                best_meta = get_meta(best_idx, best_scores.item(), best_metrics)
                worst_meta = get_meta(worst_idx, worst_scores.item(), worst_metrics)

                plot_path = os.path.join(analysis_dir, f"user_{original_uid}_best_worst_dual.png")
                plot_best_worst_matching(
                    u_concat, best_contrib, worst_contrib, 
                    best_meta, worst_meta, plot_path, original_uid, "Dual View (L:Like, R:Dislike)", is_dual_net=True
                )
                
                collected_user_profs.append(u_concat)
                collected_item_profs.append(best_contrib) 

                with open(report_path, "a") as f:
                    f.write(f"## User {original_uid}\n")
                    f.write(f"**Best**: {best_meta['title']} ({best_meta['score']:.2f})\n")
                    f.write(f"**Worst**: {worst_meta['title']} ({worst_meta['score']:.2f})\n")
                    f.write(f"![Dual Analysis](user_{original_uid}_best_worst_dual.png)\n---\n")

            else:
                u_prof = att_out.cpu().numpy().flatten()
                i_best_prof = i_best_out.cpu().numpy().flatten()
                i_worst_prof = i_worst_out.cpu().numpy().flatten()
                best_meta = get_meta(best_idx, best_scores.item())
                worst_meta = get_meta(worst_idx, worst_scores.item())

                plot_path = os.path.join(analysis_dir, f"user_{original_uid}_best_worst.png")
                plot_best_worst_matching(
                    u_prof, i_best_prof, i_worst_prof, 
                    best_meta, worst_meta, plot_path, original_uid, "Interest", is_dual_net=False
                )
                collected_user_profs.append(u_prof)
                collected_item_profs.append(i_best_prof)
                
                with open(report_path, "a") as f:
                    f.write(f"## User {original_uid}\n")
                    f.write(f"**Best**: {best_meta['title']} ({best_meta['score']:.2f})\n")
                    f.write(f"**Worst**: {worst_meta['title']} ({worst_meta['score']:.2f})\n")
                    f.write(f"![Analysis](user_{original_uid}_best_worst.png)\n---\n")

    report_diversity_diagnosis(collected_user_profs, collected_item_profs, collected_best_ids, report_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze User-Item Interest Matching")
    parser.add_argument('--exp_dir', type=str, required=True, help='Path to experiment directory')
    parser.add_argument('--num_users', type=int, default=3, help='Number of random users to analyze')
    parser.add_argument('--user_ids', type=str, default=None, help='Comma-separated specific user IDs (original)')
    
    args = parser.parse_args()
    
    user_ids = args.user_ids.split(',') if args.user_ids else None
    
    config = {
        'run_folder_path': args.exp_dir,
        'num_users': args.num_users,
        'user_ids': user_ids
    }
    
    analyze_user_recs(config)
