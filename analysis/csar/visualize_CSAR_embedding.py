import argparse
import os
import yaml
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as pe
import matplotlib.cm as cm
import sys

# 프로젝트 루트 경로를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_loader import DataLoader
from src.models import get_model

def get_item_popularity(data_loader):
    """
    전체 학습 데이터에서 아이템별 등장 횟수(Popularity)를 계산합니다.
    """
    # train_df에 접근 가능하다고 가정 (BaseDataLoader 구조에 따라 다를 수 있음)
    if hasattr(data_loader, 'train_df'):
        df = data_loader.train_df
        # 혹은 data_loader 내부 구조에 맞춰 수정
        pop_counts = df['item_id'].value_counts().sort_index() # item_id가 0~N 인덱싱 되었다고 가정
        
        # 인덱스가 비어있는 아이템(학습에 안나온) 처리
        full_counts = np.zeros(data_loader.n_items)
        full_counts[pop_counts.index] = pop_counts.values
        return full_counts
    else:
        print("Warning: Could not access train_df to calculate popularity. Using random colors.")
        return None

def visualize_embeddings_advanced(experiment_dir, output_file=None, perplexity=30):
    """
    Refactored Visualization: Joint User-Item Plots (Raw vs CSAR).
    """
    print(f"Starting Joint Visualization: {experiment_dir} (Perplexity={perplexity})")

    # 1. Config & Model Load
    config_path = os.path.join(experiment_dir, 'config.yaml')
    model_path = os.path.join(experiment_dir, 'best_model.pt')

    if not os.path.exists(config_path) or not os.path.exists(model_path):
        print(f"Error: Files not found in {experiment_dir}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_loader = DataLoader(config)
    model = get_model(config['model']['name'], config, data_loader)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # 2. Extract Data (Sampled for speed if needed, but let's try full first or large sample)
    # Sampling Users to avoid clutter (e.g., 2000 users)
    n_sample_users = 2000
    if model.user_embedding.weight.shape[0] > n_sample_users:
        indices = torch.randperm(model.user_embedding.weight.shape[0])[:n_sample_users]
        user_ids = indices
        print(f"Sampling {n_sample_users} users for visualization...")
    else:
        user_ids = torch.arange(model.user_embedding.weight.shape[0])

    with torch.no_grad():
        # --- RAW SPACE (D-dim) ---
        raw_user_embs = model.user_embedding(user_ids).cpu()
        raw_item_embs = model.item_embedding.weight.cpu()
        
        # --- CSAR SPACE (K-dim) ---
        # "Transform" = Interest Intensities
        if hasattr(model, 'attention_layer'):
            # CSAR / CSAR_Hard
            csar_user_embs = model.attention_layer(model.user_embedding(user_ids)).cpu()
            csar_item_embs = model.attention_layer(model.item_embedding.weight).cpu()
            
            # Keys
            keys = None
            if hasattr(model.attention_layer, 'interest_keys'):
                keys = model.attention_layer.interest_keys.cpu()
        else:
            # Fallback for MF/ProtoMF (No attention layer transformation usually, or different)
            print("Model does not have attention_layer. Using Raw as Transformed (Identity).")
            csar_user_embs = raw_user_embs
            csar_item_embs = raw_item_embs
            keys = None
            if hasattr(model, 'user_prototypes'): # ProtoMF
                keys = model.user_prototypes.cpu() # Treat prototypes as keys

    # 3. Generate Plots (Only 2 as requested)
    if output_file is None:
        base_dir = experiment_dir
    else:
        base_dir = os.path.dirname(output_file)

    # (A) RAW Joint Plot
    print("Generating RAW Joint Plot...")
    plot_joint_space(
        user_embs=raw_user_embs,
        item_embs=raw_item_embs,
        keys=keys if keys is not None and keys.shape[1] == raw_item_embs.shape[1] else None, # Keys might be in Raw space (ProtoMF) or not
        output_path=os.path.join(base_dir, f'joint_RAW_perp{perplexity}.png'),
        title=f"Joint RAW Space (D={raw_user_embs.shape[1]})",
        perplexity=perplexity
    )

    # (B) CSAR Transformed Joint Plot
    print("Generating CSAR Transformed Joint Plot...")
    plot_joint_space(
        user_embs=csar_user_embs,
        item_embs=csar_item_embs,
        keys=keys if keys is not None and keys.shape[1] == csar_item_embs.shape[1] else None, # Keys usually in hidden space
        output_path=os.path.join(base_dir, f'joint_CSAR_perp{perplexity}.png'),
        title=f"Joint CSAR Space (K={csar_user_embs.shape[1]})",
        perplexity=perplexity
    )
    
    # Calculate Proximity for CSAR Space (Verification)
    if keys is not None:
        # Similarity should be calculated in the RAW Embedding space (D-dim), 
        # because Keys are D-dim vectors. csar_item_embs are K-dim intensities.
        calculate_key_item_similarity(keys, raw_item_embs, output_dir=base_dir, top_k=50)


def plot_joint_space(user_embs, item_embs, keys, output_path, title, perplexity=30):
    """
    Plots Users, Items, and (Optional) Keys in the same t-SNE space.
    Style:
    - Users: Grid points (Gray/Black), small alpha
    - Items: Blue
    - Keys: Red Stars
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    # Concatenate for reduced space
    # [Users, Items, Keys]
    u_np = user_embs.detach().cpu().numpy()
    i_np = item_embs.detach().cpu().numpy()
    
    data_list = [u_np, i_np]
    labels_list = ['Users', 'Items']
    
    if keys is not None:
        k_np = keys.detach().cpu().numpy() if isinstance(keys, torch.Tensor) else keys
        data_list.append(k_np)
        labels_list.append('Keys')
        
    combined_data = np.vstack(data_list)
    n_u = u_np.shape[0]
    n_i = i_np.shape[0]
    n_k = k_np.shape[0] if keys is not None else 0

    # --- Separation Analysis ---
    u_center = np.mean(u_np, axis=0)
    i_center = np.mean(i_np, axis=0)
    centroid_dist = np.linalg.norm(u_center - i_center)
    
    u_norm = np.mean(np.linalg.norm(u_np, axis=1))
    i_norm = np.mean(np.linalg.norm(i_np, axis=1))
    
    analysis_path = os.path.join(os.path.dirname(output_path), 'separation_analysis.txt')
    with open(analysis_path, 'a') as f:
        f.write(f"\n--- {title} Analysis ---\n")
        f.write(f"User Center vs Item Center Dist: {centroid_dist:.4f}\n")
        f.write(f"Avg User Norm: {u_norm:.4f}\n")
        f.write(f"Avg Item Norm: {i_norm:.4f}\n")
        f.write(f"Norm Ratio (Item/User): {i_norm/u_norm:.4f}\n")
        if keys is not None:
             k_norm = np.mean(np.linalg.norm(k_np, axis=1))
             f.write(f"Avg Key Norm: {k_norm:.4f}\n")
             
             # Key Bridge Analysis
             k_center = np.mean(k_np, axis=0)
             dist_k_u = np.linalg.norm(k_center - u_center)
             dist_k_i = np.linalg.norm(k_center - i_center)
             
             f.write(f"Key-User Dist: {dist_k_u:.4f}\n")
             f.write(f"Key-Item Dist: {dist_k_i:.4f}\n")
             
             # Ideal Bridge: K should be between U and I
             # If Dist(U,I) ~ Dist(U,K) + Dist(K,I), then K is on the line.
             triangle_ineq = dist_k_u + dist_k_i
             f.write(f"Bridge Deviation: {triangle_ineq - centroid_dist:.4f} (0 is perfect line)\n")
             
    print(f"  -> Analysis appended to {analysis_path}")
    # ---------------------------
    
    print(f"  -> t-SNE on {combined_data.shape[0]} vectors...")
    tsne = TSNE(n_components=2, perplexity=perplexity, metric='cosine', init='pca', learning_rate='auto', random_state=42)
    embedded = tsne.fit_transform(combined_data)
    
    # Slice
    u_tsne = embedded[:n_u]
    i_tsne = embedded[n_u:n_u+n_i]
    k_tsne = embedded[n_u+n_i:] if keys is not None else None
    
    plt.figure(figsize=(12, 12))
    
    # Plot Users (Background, Gray)
    plt.scatter(u_tsne[:, 0], u_tsne[:, 1], c='lightgray', s=10, alpha=0.3, label='Users', rasterized=True)
    
    # Plot Items (Blue)
    plt.scatter(i_tsne[:, 0], i_tsne[:, 1], c='dodgerblue', s=15, alpha=0.6, label='Items', linewidth=0)
    
    # Plot Keys (Red Stars)
    if k_tsne is not None:
        plt.scatter(k_tsne[:, 0], k_tsne[:, 1], c='red', marker='*', s=200, edgecolors='black', linewidths=1.0, label='Keys/Protos', zorder=10)
        # Annotate
        import matplotlib.patheffects as pe
        for idx in range(n_k):
             plt.text(k_tsne[idx, 0], k_tsne[idx, 1], str(idx), fontsize=8, color='white', ha='center', va='center', fontweight='bold',
                      path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    plt.title(f"{title} (t-SNE perp={perplexity})")
    plt.legend(loc='upper right')
    plt.axis('off')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {output_path}")
    plt.close()


def plot_embeddings(embeddings, keys, output_path, title_prefix, config, include_keys=False):
    """
    Generic function to plot embeddings with RGB Blending coloring (Top-3 Interests).
    """
    item_embs_np = embeddings.numpy()
    n_items = item_embs_np.shape[0]
    
    # [USER REQUEST] Spectrum-Cluster Mixing
    # 1. Assign Colors to Keys based on Spectrum (PCA)
    #    Similar keys get similar colors.
    if keys is not None:
         k_tensor = keys if isinstance(keys, torch.Tensor) else torch.tensor(keys)
         e_tensor = embeddings if isinstance(embeddings, torch.Tensor) else torch.tensor(embeddings)
         k_tensor = k_tensor.to(e_tensor.device)
         n_keys = k_tensor.shape[0]

         from sklearn.decomposition import PCA
         from sklearn.preprocessing import MinMaxScaler
         
         # PCA on KEYS Only -> Get 3D Color Palette
         print("Generating Key Color Palette (PCA-3D on Keys)...")
         pca_color = PCA(n_components=3)
         key_rgb_raw = pca_color.fit_transform(k_tensor.detach().cpu().numpy())
         
         # [Fix: Gray] Boost Saturation!
         # PCA output -> MinMax(0,1) -> HSV -> Set S=1, V=1 -> RGB
         # This keeps the PCA "Hue" (Semantic) but ensures vibrant colors.
         scaler = MinMaxScaler()
         key_rgb_norm = scaler.fit_transform(key_rgb_raw)
         key_rgb_norm = np.clip(key_rgb_norm, 0.0, 1.0) # Fix floating point precision
         
         from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
         key_hsv = rgb_to_hsv(key_rgb_norm)
         key_hsv[:, 1] = 0.8  # Saturation: High but not eyes-bleeding (was 1.0)
         key_hsv[:, 2] = 0.95 # Value: Bright
         key_colors = hsv_to_rgb(key_hsv) # [K, 3]

         # 2. Color Items by Mixing Top-3 Key Colors
         #    Item Color = w1*C1 + w2*C2 + w3*C3
         print("Mixing Item Colors (Top-3, Sharpened)...")
         scores = torch.matmul(e_tensor, k_tensor.t()) # [N, K]
         
         # Top-3
         topk_vals, topk_inds = torch.topk(scores, k=3, dim=1) # [N, 3]
         
         # [Fix: Gray] Sharpen Weights!
         # If we just softmax, probabilities might be [0.34, 0.33, 0.33] -> Gray.
         # We want [0.8, 0.1, 0.1] -> Mostly Primary Color.
         # Apply Temperature or Power.
         topk_probs = torch.softmax(topk_vals * 3.0, dim=1) # Sharpening factor 3.0
         
         # Gather Colors
         selected_key_colors = torch.tensor(key_colors)[topk_inds.cpu()] # [N, 3, 3]
         
         # Weighted Sum
         item_rgb = (selected_key_colors * topk_probs.unsqueeze(-1).cpu()).sum(dim=1).numpy()
         item_rgb = np.clip(item_rgb, 0.0, 1.0)
         
         num_clusters = n_keys
    else:
         item_rgb = 'blue'
         num_clusters = 1
         key_colors = None

    # Prepare data for Dimension Reduction (Plotting Position)
    if include_keys and keys is not None:
        key_np = keys.numpy() if hasattr(keys, 'numpy') else keys
        combined_data = np.vstack([item_embs_np, key_np])
    else:
        combined_data = item_embs_np

    print(f"Running UMAP on {combined_data.shape[0]} vectors ({title_prefix})...")
    import umap
    
    # 1. Dimension Reduction for Visualization (2D) - UMAP
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
    embedded = reducer.fit_transform(combined_data)

    item_tsne = embedded[:n_items]
    key_tsne = embedded[n_items:] if keys is not None and include_keys else None

    # Plot
    plt.figure(figsize=(15, 12))
    
    # Scatter (Items) - Mixed Colors
    plt.scatter(item_tsne[:, 0], item_tsne[:, 1], 
                c=item_rgb, 
                s=30, alpha=0.7, linewidth=0, label='Items') # s=30
    
    # Scatter (Keys) - PCA Colors
    if key_tsne is not None:
        plt.scatter(key_tsne[:, 0], key_tsne[:, 1], 
                    c=key_colors, marker='*', s=200, 
                    edgecolors='black', linewidths=1.0, 
                    label='Interest Keys', zorder=10)
        
        for i in range(n_keys):
            plt.text(key_tsne[i, 0], key_tsne[i, 1], str(i), 
                     fontsize=9, fontweight='bold', color='white', 
                     ha='center', va='center',
                     path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    plt.title(f"{title_prefix} (Spectrum-Cluster Mixing)\nColors: PCA on Keys -> Mixed Top-3 per Item", fontsize=16)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Items (Mixed RGB)',
                              markerfacecolor='gray', markersize=10)]
    
    if include_keys and key_tsne is not None:
        legend_elements.append(Line2D([0], [0], marker='*', color='w', label='Interest Keys (PCA Spectrum)',
                                      markerfacecolor='red', markersize=15))

    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.savefig(output_path, dpi=300)
    print(f"Saved {title_prefix} plot to {output_path}")
    plt.close()


def plot_clustering_style(embeddings, keys, output_path, title_prefix, config, perplexity=30):
    """
    [User Request] Replicate Reference Image Style.
    Settings: t-SNE, metric=cosine.
    Style: Items (Blue Dots), Prototypes/Keys (Red Dots).
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    # Data Prep
    item_embs_np = embeddings.numpy() if hasattr(embeddings, 'numpy') else embeddings
    key_np = keys.numpy() if hasattr(keys, 'numpy') else keys
    
    n_items = item_embs_np.shape[0]
    
    # Combine for joint reduction
    if keys is not None:
        combined_data = np.vstack([item_embs_np, key_np])
    else:
        combined_data = item_embs_np
        
    print(f"Running t-SNE (Perp={perplexity}) on {combined_data.shape[0]} vectors ({title_prefix})...")
    
    # t-SNE Configuration
    # [USER REQUEST] "Increase perplexity", "Larger/Darker dots"
    tsne = TSNE(n_components=2, perplexity=perplexity, metric='cosine', init='pca', 
                learning_rate='auto', random_state=42)
    embedded = tsne.fit_transform(combined_data)
    
    item_tsne = embedded[:n_items]
    key_tsne = embedded[n_items:] if keys is not None else None
    
    # Plot
    plt.figure(figsize=(12, 12))
    
    # Items: Dodgerblue, Larger dots
    plt.scatter(item_tsne[:, 0], item_tsne[:, 1], 
                c='dodgerblue', s=20, alpha=0.8, linewidth=0, label='Items')
    
    # Keys: Red, Larger dots (Prototypes)
    if key_tsne is not None:
        plt.scatter(key_tsne[:, 0], key_tsne[:, 1], 
                    c='red', s=100, marker='o', 
                    edgecolors='white', linewidths=1.0, 
                    label='Prototypes', zorder=10)

    plt.title(f"{title_prefix} (Reference Style: t-SNE perp={perplexity})", fontsize=16)
    plt.legend(loc='upper right')
    plt.axis('off') # Reference image has no axis
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Reference Style plot to {output_path}")
    plt.close()


def plot_orthogonality_heatmap(keys, output_dir):
    """
    Visualizes the Gram Matrix of Interest Keys.
    Target: Identity Matrix (if orthogonal).
    """
    if keys is None: return
    
    # keys: [K, D]
    # Normalize
    k_tensor = torch.tensor(keys) if not isinstance(keys, torch.Tensor) else keys
    k_norm = torch.nn.functional.normalize(k_tensor, p=2, dim=1)
    
    # Gram Matrix: [K, K]
    gram = torch.matmul(k_norm, k_norm.t()).numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(gram, cmap="RdBu_r", center=0, vmin=-1, vmax=1)
    plt.title("Interest Key Orthogonality (Gram Matrix)")
    plt.xlabel("Key Index")
    plt.ylabel("Key Index")
    
    save_path = os.path.join(output_dir, "orthogonality_heatmap.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Orthogonality Heatmap to {save_path}")

def plot_interest_distribution(model, train_loader, output_dir):
    """
    Visualizes which Interest Keys are most active across Users.
    Checks for Mode Collapse (e.g. everyone using Key #0).
    """
    # Sample a batch of users
    model.eval()
    all_max_interests = []
    
    print("Calculating User Interest Distribution...")
    with torch.no_grad():
        # Iterate minimal amount to get statistical significance
        for i, batch in enumerate(train_loader):
            if i > 50: break # Check 50 batches (~50k users/interactions)
            users = batch['user_id']
            # Ensure users are on same device as model
            # But model is on CPU here (map_location='cpu'). 
            # Loader produces tensors depending on config? Usually CPU.
            
            # Simple check if model is on CUDA (it's loaded 'cpu' in current script but check to be safe)
            device = next(model.parameters()).device
            users = users.to(device)

            if hasattr(model, 'user_embedding'):
                user_embs = model.user_embedding(users)
                if hasattr(model, 'attention_layer'):
                    user_interests = model.attention_layer(user_embs) # [B, K]
                    
                    # Get Max Interest Index for each user
                    # user_interests is [B, K] (weights)
                    max_vals, max_indices = torch.max(user_interests, dim=1)
                    all_max_interests.extend(max_indices.cpu().numpy())
            else:
                 pass # Cannot compute interest distribution if no user embedding/attention logic standard
            
    if not all_max_interests:
        print("Skipping Interest Distribution (No interests found)")
        return

    plt.figure(figsize=(12, 6))
    sns.histplot(all_max_interests, bins=model.num_interests, kde=False)
    plt.title("User Interest Distribution (Primary Interest)")
    plt.xlabel("Interest Key Index")
    plt.ylabel("Count (Number of Users)")
    
    save_path = os.path.join(output_dir, "interest_distribution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Interest Distribution to {save_path}")


def calculate_key_item_similarity(keys, items, output_dir, top_k=50):
    """
    Calculates the average Cosine Similarity between each Key and its top-K nearest Items.
    Verifies if Keys are actually close to items in high-dimensional space.
    """
    if keys is None or items is None: return

    import torch
    import numpy as np

    # Ensure tensors
    k_tensor = torch.tensor(keys) if not isinstance(keys, torch.Tensor) else keys
    i_tensor = torch.tensor(items) if not isinstance(items, torch.Tensor) else items
    
    # Normalize for Cosine Similarity
    k_norm = torch.nn.functional.normalize(k_tensor, p=2, dim=1) # [K, D]
    i_norm = torch.nn.functional.normalize(i_tensor, p=2, dim=1) # [N, D]
    
    # Compute Similarity Matrix [K, N]
    sim_matrix = torch.matmul(k_norm, i_norm.t()) # Range [-1, 1]
    
    # Get Top-K similarity for each Key
    # "How close are the closest items to this key?"
    topk_sims, _ = torch.topk(sim_matrix, k=top_k, dim=1) # [K, top_k]
    
    avg_sim_per_key = topk_sims.mean(dim=1) # [K]
    global_avg_sim = avg_sim_per_key.mean().item()
    
    report_lines = []
    report_lines.append("\n" + "="*50)
    report_lines.append(f" [Analysis] Key-Item Proximity (Top-{top_k} Items per Key)")
    report_lines.append("="*50)
    report_lines.append(f"Global Average Cosine Sim: {global_avg_sim:.4f}")
    report_lines.append(f"Min Key Sim: {avg_sim_per_key.min().item():.4f}")
    report_lines.append(f"Max Key Sim: {avg_sim_per_key.max().item():.4f}")
    
    if global_avg_sim > 0.5:
        report_lines.append(">> Diagnosis: Keys are CLOSE to items. (Visual separation is likely a t-SNE artifact/Density issue)")
    elif global_avg_sim < 0.2:
        report_lines.append(">> Diagnosis: Keys are FAR from items. (Model has not learned valid clusters yet)")
    else:
        report_lines.append(">> Diagnosis: Keys are somewhat related but not tightly clustered.")
    report_lines.append("="*50 + "\n")
    
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # Save to file
    with open(os.path.join(output_dir, "analysis_report.txt"), "w") as f:
        f.write(report_text)



if __name__ == '__main__':
    # 실험 경로 설정
    target_dir = '/Users/leejongmin/code/recsys_framework/trained_model/ml-1m/csar-hard'
    
    # [USER REQUEST] Single Config Variable for Perplexity
    TARGET_PERPLEXITY = 5
    
    if os.path.isdir(target_dir):
        visualize_embeddings_advanced(target_dir, perplexity=TARGET_PERPLEXITY)
    else:
        print("Directory not found.")