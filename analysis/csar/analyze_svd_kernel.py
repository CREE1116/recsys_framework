"""
CSAR_Kernel / CSAR_Proto / CSAR_SVD 내부 파라미터 및 SVD 구조 분석 스크립트

분석 항목:
1. Kernel G vs Prototype-Gram (PP.T) 비교 (Difference 포함)
2. Singular Value Distribution
3. Membership 분포 및 엔트로피
4. SVD Backbone 정렬도 측정

사용법:
  python analysis/csar/analyze_svd_kernel.py trained_model/ml-1m
"""
import sys
from pathlib import Path

# 프로젝트 루트 추가
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import yaml


def find_svd_experiments(dataset_path: Path) -> list:
    """데이터셋 폴더 내 SVD 계열 모델 경로 찾기"""
    exp_dirs = []
    supported_patterns = ['csar_kernel', 'csar_proto', 'csar_svd', 'csar_kl', 'csar_simple_cov']
    
    for exp_dir in dataset_path.iterdir():
        if not exp_dir.is_dir():
            continue
        if any(exp_dir.name.startswith(p) for p in supported_patterns):
            config_path = exp_dir / 'config.yaml'
            model_path = exp_dir / 'best_model.pt'
            if config_path.exists() and model_path.exists():
                exp_dirs.append(exp_dir)
    return sorted(exp_dirs)


def load_model(exp_path: str, device='cpu'):
    """학습된 모델 로드"""
    from src.models import MODEL_REGISTRY
    from src.data_loader import DataLoader
    
    config_path = Path(exp_path) / 'config.yaml'
    model_path = Path(exp_path) / 'best_model.pt'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_loader = DataLoader(config)
    
    model_name = config['model']['name']
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(config, data_loader)
    
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    
    return model, data_loader, config


def extract_structures(model):
    """
    모든 CSAR 모델에서 P와 G 구조를 추출합니다.
    - P: Prototypes / Interest Keys (K, D)
    - PP: P @ P.T (K, K) - Learned Prototype Correlation
    - G: Kernel (K, K) - Actual Propagation matrix used in scores
    """
    P = None
    G = None
    
    # 1. Prototype (P) 추출
    if hasattr(model, 'prototypes'): # CSAR_Proto
        P = model.prototypes.detach().cpu()
    elif hasattr(model, 'model_layer'):
        layer = model.model_layer
        if hasattr(layer, 'P'): # CSAR_KL, CSAR_SVD
            P = layer.P.detach().cpu()
        elif hasattr(layer, 'interest_keys'): # CSAR_Basic, CSAR_SimpleCov
            P = layer.interest_keys.detach().cpu()
            
    # 2. Kernel (G) 추출 - 실제 추론(forward)에 쓰는 행렬
    if hasattr(model, 'kernel'): # CSAR_Kernel 명시적 커널
        G = model.kernel.detach().cpu()
    elif hasattr(model, 'model_layer'):
        layer = model.model_layer
        # CSAR_KL 계열 (SVD 포함): kernel_type에 따른 동적 생성
        if hasattr(layer, 'kernel_type'):
            # 모델 내부의 커널 생성 로직 재현
            P_current = layer.P.detach()
            P_centered = P_current - P_current.mean(dim=1, keepdim=True)
            P_norm = F.normalize(P_centered, p=2, dim=1)
            G_intrinsic = torch.matmul(P_norm, P_norm.T)
            
            if layer.kernel_type == 'partial':
                K = layer.K
                I = torch.eye(K, device=G_intrinsic.device)
                G_reg = G_intrinsic + 1e-4 * I
                Precision = torch.linalg.inv(G_reg)
                prec_diag = torch.diag(Precision)
                D_inv_sqrt = torch.diag(torch.pow(prec_diag, -0.5))
                G_partial = - torch.matmul(torch.matmul(D_inv_sqrt, Precision), D_inv_sqrt)
                G_partial.fill_diagonal_(1.0)
                G = G_partial.cpu()
            else: # 'raw'
                G = G_intrinsic.cpu()
        elif hasattr(layer, 'ema_G'): # CSAR_SimpleCov
            G = layer.ema_G.detach().cpu()
        elif hasattr(layer, 'get_gram_matrix'): # CSAR_Basic
            G = layer.get_gram_matrix().detach().cpu()
            
    # Fallback: G가 없으면 PP로 대체
    P_norm = None
    PP = None
    if P is not None:
        P_norm = F.normalize(P, p=2, dim=-1)
        PP = torch.matmul(P_norm, P_norm.t())
    
    if G is None:
        G = PP
        
    return P, PP, G


def analyze_kernel_stats(G):
    """G 행렬 통계 분석"""
    if G is None: return {}
    
    K = G.shape[0]
    # Singular Values
    s = torch.linalg.svdvals(G)
    s_norm = s / (s.sum() + 1e-8)
    
    sparsity = (G.abs() < 0.01).float().mean().item()
    I = torch.eye(K, device=G.device)
    cos_sim_I = F.cosine_similarity(G.view(-1), I.view(-1), dim=0).item()
    
    return {
        'K': K,
        'G_trace': torch.trace(G).item(),
        'G_sparsity': sparsity,
        'G_identity_sim': cos_sim_I,
        'G_effective_rank': torch.exp(-(s_norm * torch.log(s_norm + 1e-8)).sum()).item(),
    }, s.numpy()


def analyze_memberships(model, data_loader, device, n_samples=1000):
    """Membership 분포 분석"""
    # Initialize with default/failure values
    empty_stats = {
        'user_m_entropy': 0.0,
        'user_m_max': 0.0,
        'user_m_sparsity': 0.0,
    }
    
    if not hasattr(model, 'get_membership') and not hasattr(model, 'model_layer'):
        return None, None, None
        
    model.eval()
    with torch.no_grad():
        try:
            user_ids = torch.arange(min(n_samples, data_loader.n_users)).to(device)
            u_emb = model.user_embedding(user_ids)
            
            if hasattr(model, 'get_membership'):
                m_u = model.get_membership(u_emb)
            elif hasattr(model.model_layer, 'get_membership'):
                m_u = model.model_layer.get_membership(u_emb)
            elif hasattr(model.model_layer, 'get_mem'):
                m_u = model.model_layer.get_mem(u_emb)
            else:
                return None, None, None
            
            m_u = m_u.cpu()
        except Exception as e:
            print(f"  [DEBUG] Membership extraction failed: {e}")
            return None, None, None
        
    # Stats calculation
    m_soft = F.softmax(m_u, dim=-1)
    entropy = -torch.sum(m_soft * torch.log(m_soft + 1e-8), dim=-1).mean().item()
    
    # Empirical G: Correlation across samples
    m_u_norm = F.normalize(m_u - m_u.mean(dim=0), p=2, dim=0, eps=1e-8)
    G_empirical = torch.matmul(m_u_norm.t(), m_u_norm)
    
    stats = {
        'user_m_entropy': entropy,
        'user_m_max': m_u.max(dim=-1).values.mean().item(),
        'user_m_sparsity': (m_u < 0.1).float().mean().item(),
    }
    
    return stats, m_u.numpy(), G_empirical.numpy()


def visualize(PP, G, s, m_u, G_emp, save_path, name):
    """시각화: PP vs G vs Diff"""
    n_cols = 4
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle(f'Structural Analysis: {name}', fontsize=14)
    
    # Row 1: Structures (Model Defined)
    # 1. Prototype Gram (PP)
    sns.heatmap(PP, ax=axes[0, 0], cmap='coolwarm', center=0)
    axes[0, 0].set_title('Prototype Gram (PP^T)')
    
    # 2. Kernel (G)
    sns.heatmap(G, ax=axes[0, 1], cmap='coolwarm', center=0)
    axes[0, 1].set_title('Kernel (G)')
    
    # 3. Difference (G - PP)
    diff = G - PP
    sns.heatmap(diff, ax=axes[0, 2], cmap='PRGn', center=0)
    axes[0, 2].set_title('Difference (G - PP)')
    
    # 4. Singular Values of G
    axes[0, 3].plot(s, marker='o', markersize=3)
    axes[0, 3].set_yscale('log')
    axes[0, 3].set_title('Singular Values of G')
    axes[0, 3].grid(True, alpha=0.3)
    
    # Row 2: Empirical Results (Data Emerging)
    if m_u is not None:
        # 5. Empirical G (Actual Interaction Pattern)
        sns.heatmap(G_emp, ax=axes[1, 0], cmap='YlOrRd')
        axes[1, 0].set_title('Empirical G (Membership Corr)')
        
        # 6. User Membership Dist (All)
        axes[1, 1].hist(m_u.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Membership Distribution')
        
        # 6. User Membership Entropy Distribution (per user)
        m_prob = F.softmax(torch.from_numpy(m_u), dim=-1)
        user_entropy = -torch.sum(m_prob * torch.log(m_prob + 1e-8), dim=-1).numpy()
        axes[1, 1].hist(user_entropy, bins=30, color='green', alpha=0.6)
        axes[1, 1].set_title('User Entropy Dist')
        
    # Remove empty plots
    for i in range(n_cols):
        for j in range(n_rows):
            if not axes[j, i].has_data() and len(axes[j, i].get_images()) == 0:
                axes[j, i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def analyze_single(exp_path: Path, device: str):
    print(f"\n> Analyzing: {exp_path.name}")
    try:
        model, data_loader, config = load_model(str(exp_path), device)
    except Exception as e:
        print(f"  [ERROR] Load failed: {e}")
        return
        
    all_stats = {}
    
    # 1. Structure Extraction
    P, PP, G = extract_structures(model)
    
    # 2. Kernel Stats
    g_stats, s = analyze_kernel_stats(G)
    all_stats.update(g_stats)
    
    # 3. Membership Stats
    m_stats, m_u, G_emp = analyze_memberships(model, data_loader, device)
    if m_stats:
        all_stats.update(m_stats)
        
    # Stats Logging
    for k, v in all_stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        
    # Visualization
    save_path = exp_path / 'svd_kernel_analysis_v2.png'
    
    # Defensive numpy conversion
    PP_np = PP.numpy() if PP is not None else np.eye(G.shape[0])
    G_np = G.numpy() if G is not None else np.eye(PP.shape[0])
    G_emp_np = G_emp if G_emp is not None else None
    
    visualize(PP_np, G_np, s, m_u, G_emp_np, save_path, exp_path.name)
    print(f"  [Output] Saved plot to {save_path.name}")
    
    # Save JSON
    with open(exp_path / 'svd_kernel_stats_v2.json', 'w') as f:
        json.dump(all_stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to dataset result folder')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    path = Path(args.path)
    exps = find_svd_experiments(path)
    
    if not exps:
        print(f"No SVD/CSAR experiments found in {path}")
        return
        
    print(f"### SVD Structure Analysis: {path.name} ###")
    for exp in exps:
        analyze_single(exp, args.device)


if __name__ == '__main__':
    main()
