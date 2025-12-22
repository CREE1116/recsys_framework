import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import sys
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

# 프로젝트 루트 경로 설정
# 프로젝트 루트 경로 설정 (analysis/csar/ -> ../../)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.csar.CSAR_DualView import CSAR_DualView
from src.models.csar.CSAR_Sampled import CSAR_Sampled
from src.data_loader import DataLoader

def load_model(run_folder_path, device='cpu'):
    config_path = os.path.join(run_folder_path, 'config.yaml')
    model_path = os.path.join(run_folder_path, 'best_model.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(run_folder_path, 'best_model.pt')
    
    if not os.path.exists(config_path) or not os.path.exists(model_path):
        print(f"[Error] Missing config or model file in {run_folder_path}")
        return None, None
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Standard DataLoader
    data_loader = DataLoader(config)

    
    model_name = config['model']['name']
    if model_name == 'csar-dualview':
        model = CSAR_DualView(config, data_loader)
    elif model_name == 'csar-sampled':
        model = CSAR_Sampled(config, data_loader)
    else:
        print(f"Unknown model name: {model_name}")
        return None, None
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, config

def analyze_orthogonality(exp_dir):
    print(f"Analyzing Orthogonality for: {exp_dir}")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    model, config = load_model(exp_dir, device)
    if model is None:
        return

    keys_dict = {}
    
    # Extract keys based on model type
    if isinstance(model, CSAR_DualView):
        keys_dict['Pos Keys'] = model.attention_layer.pos_keys.detach()
        keys_dict['Neg Keys'] = model.attention_layer.neg_keys.detach()
        # All keys concatenated
        keys_dict['All Keys'] = torch.cat([model.attention_layer.pos_keys, model.attention_layer.neg_keys], dim=0).detach()
        
    elif isinstance(model, CSAR_Sampled):
        keys_dict['Interest Keys'] = model.attention_layer.interest_keys.detach()
        
    output_dir = os.path.join(exp_dir.replace("trained_model", "output"), "orthogonality_analysis")
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "orthogonality_report.md")
    
    with open(report_path, "w") as f:
        f.write(f"# Orthogonality Analysis Report\n")
        f.write(f"- Model: {config['model']['name']}\n")
        f.write(f"- Path: {exp_dir}\n\n")

        for name, keys in keys_dict.items():
            K = keys.size(0)
            keys_norm = F.normalize(keys, p=2, dim=1)
            
            # Compute Cosine Similarity Matrix
            sim_matrix = torch.matmul(keys_norm, keys_norm.t()) # [K, K]
            
            # Off-diagonal elements
            mask = ~torch.eye(K, dtype=bool, device=device)
            off_diag_sims = sim_matrix[mask]
            
            mean_sim = off_diag_sims.abs().mean().item()
            max_sim = off_diag_sims.max().item()
            min_sim = off_diag_sims.min().item()
            
            print(f"[{name}] K={K}")
            print(f"  Mean Abs Sim (Off-Diag): {mean_sim:.4f}")
            print(f"  Max Sim: {max_sim:.4f}")
            
            f.write(f"## {name} (K={K})\n")
            f.write(f"- **Mean Abs Similarity (Off-Diagonal)**: `{mean_sim:.4f}` (Lower is better, 0 = Perfect Orthogonal)\n")
            f.write(f"- **Max Similarity**: `{max_sim:.4f}`\n")
            f.write(f"- **Min Similarity**: `{min_sim:.4f}`\n\n")
            
            # Add histogram of correlations
            f.write("### Similarity Distribution\n")
            counts, bins = np.histogram(off_diag_sims.cpu().numpy(), bins=20, range=(-1, 1))
            
            # Log histogram as text chart
            hist_str = ""
            for i in range(len(counts)):
                if counts[i] > 0:
                    bar = '#' * int(counts[i] / len(off_diag_sims) * 50)
                    if not bar and counts[i] > 0: bar = "."
                    hist_str += f"{bins[i]:.2f} ~ {bins[i+1]:.2f}: {counts[i]} {bar}\n"
            
            f.write("```\n" + hist_str + "```\n\n")

            # --- Visualization ---
            plt.figure(figsize=(10, 8))
            sns.heatmap(sim_matrix.cpu().numpy(), cmap='coolwarm', vmin=-1, vmax=1)
            plt.title(f"{name} Orthogonality (Avg Abs Sim: {mean_sim:.3f})")
            img_path = os.path.join(output_dir, f"{name.replace(' ', '_').lower()}_heatmap.png")
            plt.savefig(img_path)
            plt.close()
            
            f.write(f"![Heatmap]({img_path})\n\n")
            
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True, help="Path to experiment directory")
    args = parser.parse_args()
    
    analyze_orthogonality(args.exp_dir)
