import sys
from pathlib import Path

# 프로젝트 루트 추가
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from src.models import get_model

def find_apollo_experiments(dataset_path: Path) -> list:
    """데이터셋 폴더 내 APOLLO 모델 경로 찾기"""
    exp_dirs = []
    for exp_dir in dataset_path.iterdir():
        if not exp_dir.is_dir():
            continue
        config_path = exp_dir / 'config.yaml'
        model_path = exp_dir / 'best_model.pt'
        if config_path.exists() and model_path.exists():
            with open(config_path, 'r') as f:
                try:
                    config = yaml.safe_load(f)
                    if config['model']['name'].lower().startswith('apollo'):
                        exp_dirs.append(exp_dir)
                except:
                    continue
    return sorted(exp_dirs)

def load_apollo(exp_path: str, device='cpu'):
    config_path = Path(exp_path) / 'config.yaml'
    model_path = Path(exp_path) / 'best_model.pt'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Dummy data loader for initialization
    from src.data_loader import DataLoader
    data_loader = DataLoader(config)
    
    from src.models import MODEL_REGISTRY
    model_class = MODEL_REGISTRY[config['model']['name']]
    model = model_class(config, data_loader)
    
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    
    return model, config

def analyze_apollo(exp_dir: str):
    print(f"\n> Analyzing APOLLO: {Path(exp_dir).name}")
    exp_path = Path(exp_dir)
    try:
        model, config = load_apollo(exp_dir)
    except Exception as e:
        print(f"  [ERROR] Load failed: {e}")
        return
    
    # 1. Extract Parameters
    S_curr = model.S.detach().cpu()
    S_init = getattr(model, 'S_init', None)
    if S_init is not None: S_init = S_init.detach().cpu()
    
    V_curr = model.item_embedding.weight.detach().cpu()
    # Try V_init (old) or I_init (new symmetric)
    V_init = getattr(model, 'V_init', getattr(model, 'I_init', None))
    if V_init is not None: V_init = V_init.detach().cpu()
    
    U_curr = model.user_embedding.weight.detach().cpu()
    U_init = getattr(model, 'U_init', None)
    if U_init is not None: U_init = U_init.detach().cpu()
    
    # 2. Calculate Stats
    diff_s = (S_curr - S_init).abs().mean().item() if S_init is not None else 0
    diff_v = (V_curr - V_init).abs().mean().item() if V_init is not None else 0
    diff_u = (U_curr - U_init).abs().mean().item() if U_init is not None else 0
    
    print(f"  Mean S change: {diff_s:.6f}")
    print(f"  Mean V change: {diff_v:.6f}")
    print(f"  Mean U change: {diff_u:.6f}")
    
    # 3. Visualization
    fig, axes = plt.subplots(1, 6, figsize=(30, 5))
    
    # [1] S_init (Initial Resonance)
    if S_init is not None:
        sns.heatmap(S_init.numpy(), ax=axes[0], cmap='coolwarm', center=0, cbar=False)
        axes[0].set_title('Initial Resonance (S_init)')
    else:
        axes[0].set_title('S_init not available')

    # [2] S_curr (Learned Resonance)
    sns.heatmap(S_curr.numpy(), ax=axes[1], cmap='coolwarm', center=0, cbar=False)
    axes[1].set_title('Learned Resonance (S_learned)')
    
    # [3] Change Matrix (S_curr - S_init)
    if S_init is not None:
        sns.heatmap((S_curr - S_init).numpy(), ax=axes[2], cmap='PRGn', center=0, cbar=False)
        axes[2].set_title('Change (S_learned - S_init)')
    else:
        axes[2].set_title('Change n/a')

    # [4] Diagonal Component (Identity check)
    s_diag = torch.diag(S_curr).numpy()
    axes[3].bar(range(len(s_diag)), s_diag, color='skyblue')
    axes[3].axhline(y=0, color='black', linewidth=0.5)
    axes[3].set_title('Diagonal of S_learned')
    axes[3].set_ylim(min(-0.1, s_diag.min()*1.1), max(1.1, s_diag.max()*1.1))

    # [5] Singular Values (Check for rank collapse)
    s_vals_curr = torch.linalg.svdvals(S_curr).numpy()
    if S_init is not None:
        s_vals_init = torch.linalg.svdvals(S_init).numpy()
        axes[4].plot(s_vals_init, label='Initial', alpha=0.5, linestyle='--')
    
    axes[4].plot(s_vals_curr, label='Learned', color='red', linewidth=2)
    axes[4].set_yscale('log')
    axes[4].set_title('Singular Values (Log)')
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)

    # [6] User Embedding Drift (Dist shift)
    if U_init is not None:
        drift = torch.norm(U_curr - U_init, p=2, dim=1).numpy()
        axes[5].hist(drift, bins=50, color='gold', alpha=0.7)
        axes[5].set_title('User Embedding Drift (|U-U0|)')
        axes[5].set_xlabel('L2 Distance')
    else:
        axes[5].set_title('U_init n/a')

    plt.tight_layout()
    save_path = exp_path / 'apollo_resonance_analysis.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [Output] Saved analysis plot to {save_path.name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to dataset result folder or experiment folder')
    args = parser.parse_args()
    
    path = Path(args.path)
    if (path / 'config.yaml').exists():
        analyze_apollo(str(path))
    else:
        exps = find_apollo_experiments(path)
        if not exps:
            print(f"No APOLLO experiments found in {path}")
        else:
            for exp in exps:
                analyze_apollo(str(exp))
