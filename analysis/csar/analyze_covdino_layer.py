"""
CSAR_CovDino / CSAR_Cov 내부 파라미터 분석 스크립트

분석 항목:
1. Interest Keys 직교도 (Orthogonality)
2. PP (Prototype-Prototype) vs EMA_G 비교
3. 멤버십 분포 (유저/아이템)
4. DINO 유사도 (PP와 G의 구조적 유사성)

사용법:
  python analysis/csar/analyze_covdino_layer.py /path/to/trained_model/ml-100k
  => ml-100k 폴더 내 모든 csar_cov/csar_covdino 모델을 찾아서 분석
"""
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
import json
import yaml
import torch.nn.functional as F


def find_covdino_experiments(dataset_path: Path) -> list:
    """데이터셋 폴더 내 csar_cov/csar_covdino 모델 경로 찾기"""
    rec_dirs = []
    supported_patterns = ['csar_cov', 'csar_covdino', 'csar_simple_cov', 'csar_batchnorm', 'csar_geometry', 'csar_kl', 'csar_dualproto', 
                          'csar-cov', 'csar-covdino', 'csar-simple-cov', 'csar-batchnorm', 'csar-geometry', 'csar-kl', 'csar-dualproto']
    
    for exp_dir in dataset_path.iterdir():
        if not exp_dir.is_dir():
            continue
        if any(exp_dir.name.startswith(p) for p in supported_patterns):
            config_path = exp_dir / 'config.yaml'
            model_path = exp_dir / 'best_model.pt'
            if config_path.exists() and model_path.exists():
                rec_dirs.append(exp_dir)
    return sorted(rec_dirs)


def load_model(exp_path: str, device='cpu', cached_data_loader=None):
    """학습된 모델 로드 (DataLoader 재사용 가능)"""
    from src.models import MODEL_REGISTRY
    from src.data_loader import DataLoader
    
    config_path = Path(exp_path) / 'config.yaml'
    model_path = Path(exp_path) / 'best_model.pt'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # DataLoader 재사용 또는 새로 생성
    if cached_data_loader is not None:
        data_loader = cached_data_loader
    else:
        # 데이터 로딩 속도 최적화 (num_workers 등은 내부 설정 따름)
        data_loader = DataLoader(config)
    
    model_name = config['model']['name']
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(config, data_loader)
    
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    
    return model, data_loader, config


def get_csar_layer(model):
    """CSAR 레이어 가져오기"""
    if hasattr(model, 'model_layer'):
        return model.model_layer
    elif hasattr(model, 'attention_layer'):
        return model.attention_layer
    else:
        raise AttributeError("모델에 model_layer 또는 attention_layer가 없습니다.")


def analyze_interest_keys(model):
    """Interest Keys 분석"""
    layer = get_csar_layer(model)
    # Check for P (BatchNorm) or interest_keys (SimpleCov)
    if hasattr(model, 'P_u'): # CSAR_DualProto
        keys = model.P_u.detach().cpu()
    elif hasattr(layer, 'P'):
        keys = layer.P.detach().cpu()
    else:
        keys = layer.interest_keys.detach().cpu()
        
    K, d = keys.shape
    
    norms = torch.norm(keys, p=2, dim=1)
    
    # Cosine Similarity (Normalized Gram)
    keys_norm = F.normalize(keys, p=2, dim=1)
    gram = keys_norm @ keys_norm.t()
    
    off_diag_mask = 1 - torch.eye(K)
    off_diag = gram * off_diag_mask
    orthogonality = 1 - off_diag.abs().mean().item()
    
    stats = {
        'K': K,
        'd': d,
        'norm_mean': norms.mean().item(),
        'norm_std': norms.std().item(),
        'orthogonality': orthogonality,
        'avg_cosine_sim': off_diag.abs().mean().item(),
        'max_cosine_sim': off_diag.abs().max().item(),
    }
    
    # Raw Gram (PP)
    PP = keys @ keys.t()
    
    return stats, gram.numpy(), PP.numpy()


def analyze_gram_matrices(model):
    """PP vs EMA_G 분석 (CovDino 핵심)"""
    layer = get_csar_layer(model)
    
    with torch.no_grad():
        # 1. Get PP (Prototype Structure)
        if hasattr(layer, 'get_pp_matrix'):
            PP = layer.get_pp_matrix().cpu()
        elif hasattr(model, 'P_u'): # CSAR_DualProto
             # Use P_u intrinsic as "PP"
             keys = model.P_u
             keys_norm = F.normalize(keys, p=2, dim=-1)
             PP = torch.matmul(keys_norm, keys_norm.t()).cpu()
        else:
            # Fallback: Compute Cosine PP (since CSAR_SimpleCov aligns Cosine P to G)
            if hasattr(layer, 'P'):
                 keys = layer.P
            else:
                 keys = layer.interest_keys
                 
            keys_norm = F.normalize(keys, p=2, dim=-1)
            PP = torch.matmul(keys_norm, keys_norm.t()).cpu()

        # 2. Get G (Statistical Structure)
        # 2. Get G (Statistical Structure)
        if hasattr(model, 'P_u'): # CSAR_DualProto
            # DualProto doesn't has EMA_G, but aims to align P_u, P_i, and Cross.
            # We use Cross-Correlation (P_u @ P_i.T) as 'G' to visualize the shared structure.
            P_u = model.P_u
            P_i = model.P_i
            G = layer.get_intrinsic_correlation(P_u, P_i).cpu()
            
            # Re-define PP as P_u intrinsic for consistency
            PP = layer.get_intrinsic_correlation(P_u).cpu()

        elif hasattr(layer, 'get_gram_matrix'):
            G = layer.get_gram_matrix().cpu()
        elif hasattr(layer, 'ema_G'):
            G = layer.ema_G.cpu()
        elif hasattr(layer, 'running_PP'): # BatchNorm Style
            G = layer.running_PP.cpu()
        else:
             # Just use PP as G if no other G available (e.g. basic models)
             G = PP 
    
    K = G.shape[0]
    
    # PP 통계
    PP_diag = torch.diag(PP)
    PP_off = PP * (1 - torch.eye(K))
    PP_off_flat = PP_off[PP_off != 0]
    
    # G 통계
    G_diag = torch.diag(G)
    G_off = G * (1 - torch.eye(K))
    G_off_flat = G_off[G_off != 0]
    
    # PP vs G 유사도 (코사인)
    PP_norm = F.normalize(PP, p=2, dim=-1)
    G_norm = F.normalize(G, p=2, dim=-1)
    cosine_sim = (PP_norm * G_norm).sum(dim=-1).mean().item()
    
    # Frobenius distance
    frob_dist = torch.norm(PP - G, p='fro').item()
    
    stats = {
        'PP_diag_mean': PP_diag.mean().item(),
        'PP_off_mean': PP_off_flat.mean().item() if len(PP_off_flat) > 0 else 0,
        'PP_trace': torch.trace(PP).item(),
        'G_diag_mean': G_diag.mean().item(),
        'G_off_mean': G_off_flat.mean().item() if len(G_off_flat) > 0 else 0,
        'G_trace': torch.trace(G).item(),
        'PP_G_cosine_sim': cosine_sim,
        'PP_G_frob_dist': frob_dist,
        'dino_loss_approx': 1 - cosine_sim,  # DINO loss 근사
    }
    
    return stats, PP.numpy(), G.numpy()


def analyze_membership_distribution(model, data_loader, device, n_samples=1000):
    """멤버십 분포 분석 (Robust Entropy)"""
    model.eval()
    
    with torch.no_grad():
        user_ids = torch.arange(min(n_samples, data_loader.n_users)).to(device)
        user_embs = model.user_embedding(user_ids)
        
        if hasattr(model, 'P_u'): # CSAR_DualProto
            user_memberships = get_csar_layer(model).get_mem(user_embs, model.P_u).detach().cpu()
        else:
            user_memberships = get_csar_layer(model).get_membership(user_embs).detach().cpu()
        
        item_embs = model.item_embedding.weight[:min(n_samples, data_loader.n_items)]
        
        if hasattr(model, 'P_i'): # CSAR_DualProto
             item_memberships = get_csar_layer(model).get_mem(item_embs, model.P_i).detach().cpu()
        else:
             item_memberships = get_csar_layer(model).get_membership(item_embs).detach().cpu()
    
    def membership_stats(m, prefix):
        # 1. Negative Handling (Cosine Similarity -> Probability)
        if m.min() < 0:
            # Shift [-1, 1] -> [0, 1]
            m_prob = (m + 1.0) / 2.0
            sparsity = (m < 0.1).float().mean().item() # Sparsity based on original value (<0.1)
        else:
            m_prob = m
            sparsity = (m < 1e-3).float().mean().item()

        # 2. Normalize for Entropy
        m_prob_sum = m_prob.sum(dim=-1, keepdim=True) + 1e-10
        m_norm = m_prob / m_prob_sum
        
        # 3. Robust Entropy
        # 0 * log(0) 이슈 방지용 clamping
        m_norm_clamped = torch.clamp(m_norm, min=1e-10)
        entropy = -torch.sum(m_norm * torch.log(m_norm_clamped), dim=-1).mean().item()
        
        active = (m > 0.5).float().sum(dim=-1).mean().item()
        
        return {
            f'{prefix}_entropy': entropy,
            f'{prefix}_sparsity': sparsity,
            f'{prefix}_max_mean': m.max(dim=-1).values.mean().item(),
            f'{prefix}_active_interests': active,
        }
    
    stats = {}
    stats.update(membership_stats(user_memberships, 'user'))
    stats.update(membership_stats(item_memberships, 'item'))
    
    return stats, user_memberships.numpy(), item_memberships.numpy()


def visualize_all(keys_gram, PP, G, user_m, item_m, save_path):
    """시각화"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Keys Cosine Similarity
    sns.heatmap(keys_gram, ax=axes[0, 0], cmap='coolwarm', center=0, vmin=-1, vmax=1)
    axes[0, 0].set_title('Keys Cosine Similarity')
    
    # 2. PP (Prototype-Prototype)
    sns.heatmap(PP, ax=axes[0, 1], cmap='YlOrRd')
    axes[0, 1].set_title('PP (Keys @ Keys.T)')
    
    # 3. EMA_G (Covariance-based)
    sns.heatmap(G, ax=axes[0, 2], cmap='YlOrRd')
    axes[0, 2].set_title('EMA_G (Covariance)')
    
    # 4. User Membership Distribution
    axes[1, 0].hist(user_m.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('User Membership Distribution')
    axes[1, 0].set_xlabel('Membership Value')
    
    # 5. Item Membership Distribution
    axes[1, 1].hist(item_m.flatten(), bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[1, 1].set_title('Item Membership Distribution')
    axes[1, 1].set_xlabel('Membership Value')
    
    # 6. PP vs G Difference
    diff = PP - G
    sns.heatmap(diff, ax=axes[1, 2], cmap='coolwarm', center=0)
    axes[1, 2].set_title('PP - G (Difference)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_single_experiment(exp_path: Path, device: str = 'cpu', cached_data_loader=None):
    """단일 실험 분석"""
    print(f"\n{'='*60}")
    print(f"  {exp_path.name}")
    print(f"{'='*60}")
    
    try:
        model, data_loader, config = load_model(str(exp_path), device, cached_data_loader)
    except Exception as e:
        print(f"  [ERROR] 모델 로드 실패: {e}")
        return None, None
    
    # 1. Interest Keys
    print("\n[1] Interest Keys Analysis")
    print("-" * 40)
    keys_stats, keys_gram, PP_raw = analyze_interest_keys(model)
    for k, v in keys_stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # 2. PP vs EMA_G (핵심!)
    print("\n[2] PP vs EMA_G Analysis (CovDino Core)")
    print("-" * 40)
    gram_stats, PP, G = analyze_gram_matrices(model)
    for k, v in gram_stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # 3. Membership Distribution
    print("\n[3] Membership Distribution Analysis")
    print("-" * 40)
    mem_stats, user_m, item_m = analyze_membership_distribution(
        model, data_loader, device
    )
    for k, v in mem_stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # 4. Visualization
    save_path = exp_path / 'covdino_layer_analysis.png'
    visualize_all(keys_gram, PP, G, user_m, item_m, save_path)
    print(f"\n[4] Saved: {save_path}")
    
    # 5. Save stats
    all_stats = {
        'keys': keys_stats,
        'gram': gram_stats,
        'membership': mem_stats,
    }
    stats_path = exp_path / 'covdino_layer_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"    Saved: {stats_path}")
    
    return all_stats, data_loader


def main():
    parser = argparse.ArgumentParser(description='CSAR_CovDino 레이어 분석')
    parser.add_argument('dataset_path', type=str, 
                        help='데이터셋 결과 폴더 (예: trained_model/ml-100k)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='디바이스 (cpu/cuda/mps)')
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"[ERROR] 경로가 존재하지 않습니다: {dataset_path}")
        return
    
    rec_dirs = find_covdino_experiments(dataset_path)
    
    if not rec_dirs:
        print(f"[WARNING] csar_cov/csar_covdino 모델을 찾을 수 없습니다: {dataset_path}")
        return
    
    print(f"\n{'#'*60}")
    print(f"# CSAR_CovDino 분석: {dataset_path.name}")
    print(f"# 발견된 모델 수: {len(rec_dirs)}")
    print(f"{'#'*60}")
    
    cached_loader = None
    for exp_dir in rec_dirs:
        # 캐시된 로더 사용, 첫 번째 실험에서 로더가 생성됨
        stats, cached_loader = analyze_single_experiment(exp_dir, args.device, cached_loader)
    
    print(f"\n{'='*60}")
    print("모든 분석 완료!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
