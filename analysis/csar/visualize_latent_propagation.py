import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import numpy as np
import sys

# Add project root to sys.path for robust imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models import get_model
from src.data_loader import DataLoader
from yaml import safe_load

def compute_matrix_stats(M):
    """Computes various matrix metrics for analysis."""
    diag = np.diag(M)
    off_diag = M[~np.eye(M.shape[0], dtype=bool)]
    
    # 1. Diagonal Energy Ratio (Intensity)
    # How much of the matrix 'weight' is on the diagonal
    total_abs_sum = np.sum(np.abs(M))
    diag_abs_sum = np.sum(np.abs(diag))
    diag_ratio = diag_abs_sum / (total_abs_sum + 1e-9)
    
    # 2. Orthogonality (Independency) Measure
    # For a kernel/correlation matrix, how close is it to Identity?
    # Measure: 1 - mean absolute off-diagonal correlation
    orthogonality = 1.0 - np.mean(np.abs(off_diag))
    
    # 3. Sparsity (Effective)
    sparsity = (np.abs(off_diag) < 0.01).sum() / (off_diag.size + 1e-9)
    
    return {
        "diag_ratio": float(diag_ratio),
        "orthogonality": float(orthogonality),
        "sparsity": float(sparsity),
        "mean_off": float(np.mean(off_diag)),
        "max_off": float(np.max(off_diag)),
        "std_off": float(np.std(off_diag))
    }

def visualize_single_model(model_path, dataset_config=None, force_data_loader=None):
    # 1. Load Config
    run_dir = os.path.dirname(model_path)
    config_path = os.path.join(run_dir, 'config.yaml')
    
    if not os.path.exists(config_path):
        print(f"Skipping {run_dir}: No config.yaml found.")
        return
        
    with open(config_path, 'r') as f:
        config = safe_load(f)
    
    # Update with dataset config if explicitly provided
    if dataset_config and os.path.exists(dataset_config):
        with open(dataset_config, 'r') as f:
            ds_config = safe_load(f)
            config.update(ds_config)
    
    config['device'] = 'cpu'
    
    # 2. Load Data (Reuse loader if provided for speed)
    if force_data_loader is None:
        print(f"Loading data for {run_dir}...")
        data_loader = DataLoader(config)
    else:
        data_loader = force_data_loader
    
    # 3. Load Model
    print(f"Extracting kernels from {model_path}...")
    try:
        model = get_model(config['model']['name'], config, data_loader)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return
    
    # 4. Extract Kernels
    G = None
    H_t = None
    is_mixer = False
    
    # Try searching on the model directly (new structure)
    if hasattr(model, 'Mixer'):
        H_t = model.Mixer.detach().cpu().numpy()
        is_mixer = True
    elif hasattr(model, 'Inf_Kernel'):
        H_t = model.Inf_Kernel.detach().cpu().numpy()
    elif hasattr(model, 'H_t'):
        H_t = model.H_t.detach().cpu().numpy()
    
    if hasattr(model, 'G'):
        G = model.G.detach().cpu().numpy()
        
    # Legacy search (inside prop_layer)
    if G is None and H_t is None and hasattr(model, 'prop_layer'):
        if hasattr(model.prop_layer, 'G'):
            G = model.prop_layer.G.detach().cpu().numpy()
        if hasattr(model.prop_layer, 'H_t'):
            H_t = model.prop_layer.H_t.detach().cpu().numpy()
    
    if G is None and H_t is None:
        print(f"Warning: No kernels found in {model_path}")
        return

    # 5. Plotting
    n_plots = (1 if G is not None else 0) + (1 if H_t is not None else 0)
    if n_plots == 0: return
    
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]
    
    idx = 0
    if G is not None:
        stats = compute_matrix_stats(G)
        ax = axes[idx]
        G_plot = G.copy()
        np.fill_diagonal(G_plot, 0)
        vmax = np.percentile(np.abs(G_plot), 99.5) * 1.5
        vmax = max(vmax, 0.01)
        sns.heatmap(G_plot, ax=ax, cmap='coolwarm', center=0, vmax=vmax, vmin=-vmax, square=True)
        
        title = (f"Pearson (Off-diag)\n"
                 f"K={G.shape[0]}\n"
                 f"Diag Energy: {stats['diag_ratio']:.1%}\n"
                 f"Orthogonality: {stats['orthogonality']:.3f}")
        ax.set_title(title)
        idx += 1
        
    if H_t is not None:
        stats = compute_matrix_stats(H_t)
        ax = axes[idx]
        H_plot = H_t.copy()
        np.fill_diagonal(H_plot, 0)
        vmax_h = np.percentile(np.abs(H_plot), 99.5) * 2.0
        vmax_h = max(vmax_h, 1e-4)
        sns.heatmap(H_plot, ax=ax, cmap='viridis', vmax=vmax_h, square=True)
        
        main_type = "Mixer" if is_mixer else "Propagation"
        param_name = "alpha" if not is_mixer else "mix"
        param_val = config['model'].get('alpha') if not is_mixer else "0.2"
        
        title = (f"Energy {main_type} (Off-diag)\n"
                 f"{param_name}={param_val}\n"
                 f"Diag Energy: {stats['diag_ratio']:.1%}\n"
                 f"Sparsity: {stats['sparsity']:.1%}")
        ax.set_title(title)
    
    plt.tight_layout()
    save_path = os.path.join(run_dir, 'kernel_visualization.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    # 6. Save Stats to JSON
    import json
    combined_stats = {
        "model_name": config['model']['name'],
        "heat_t": config['model'].get('heat_t'),
        "num_interests": G.shape[0] if G is not None else (H_t.shape[0] if H_t is not None else 0),
        "pearson_stats": compute_matrix_stats(G) if G is not None else None,
        "heat_kernel_stats": compute_matrix_stats(H_t) if H_t is not None else None
    }
    
    json_path = os.path.join(run_dir, 'kernel_stats.json')
    with open(json_path, 'w') as f:
        json.dump(combined_stats, f, indent=4)
        
    print(f"  ✓ Image: {save_path}")
    print(f"  ✓ JSON:  {json_path}")

def batch_visualize(base_dir, dataset_config):
    print(f"Searching for LatentPropagation models in {base_dir}...")
    
    # Loader cache: dataset_name -> DataLoader
    loader_cache = {}
    
    model_count = 0
    for root, dirs, files in os.walk(base_dir):
        if 'best_model.pt' in files and 'latent_propagation' in root:
            model_path = os.path.join(root, 'best_model.pt')
            config_path = os.path.join(root, 'config.yaml')
            
            if not os.path.exists(config_path): continue
            
            with open(config_path, 'r') as f:
                config = safe_load(f)
            
            # Determine which dataset this model belongs to
            dataset_name = config.get('dataset_name', 'unknown')
            
            if dataset_name not in loader_cache:
                print(f"\n[Batch] Initializing DataLoader for dataset: {dataset_name}")
                if dataset_config:
                    with open(dataset_config, 'r') as f:
                        ds_config = safe_load(f)
                        config.update(ds_config)
                # Ensure device is cpu for visualization extraction
                config['device'] = 'cpu'
                loader_cache[dataset_name] = DataLoader(config)
            
            visualize_single_model(model_path, dataset_config, force_data_loader=loader_cache[dataset_name])
            model_count += 1
            
    print(f"\nBatch processing complete. Total models visualized: {model_count}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to a specific best_model.pt')
    parser.add_argument('--base_dir', type=str, help='Base directory to search for models')
    parser.add_argument('--dataset_config', type=str, help='Force a specific dataset config (optional)')
    
    args = parser.parse_args()
    
    if args.model_path:
        visualize_single_model(args.model_path, args.dataset_config)
    elif args.base_dir:
        batch_visualize(args.base_dir, args.dataset_config)
    else:
        print("Please provide --model_path or --base_dir")
