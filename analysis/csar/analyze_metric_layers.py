"""
CSAR Metric Layers 내부 파라미터 분석 스크립트
(Subspace, Bilateral, Crossturm Metric 분석)

분석 항목:
1. Prototype P 분석 (크기, 직교도)
2. Delta 분포 분석 (유저/아이템)
3. Score 분해 (Global vs Local)

사용법:
  python analysis/csar/analyze_metric_layers.py trained_model/ml-1m
  => csar_subspace_metric, csar_bilateral_metric, csar_crossturm_metric 분석
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


METRIC_LAYER_PREFIXES = [
    'csar_subspace_metric',
    'csar_bilateral_metric',
    'csar_crossturm_metric',
]


def find_metric_experiments(dataset_path: Path) -> list:
    """데이터셋 폴더 내 metric 모델 경로 찾기"""
    exp_dirs = []
    for exp_dir in dataset_path.iterdir():
        if not exp_dir.is_dir():
            continue
        for prefix in METRIC_LAYER_PREFIXES:
            if exp_dir.name.startswith(prefix):
                config_path = exp_dir / 'config.yaml'
                model_path = exp_dir / 'best_model.pt'
                if config_path.exists() and model_path.exists():
                    exp_dirs.append(exp_dir)
                break
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


def analyze_prototypes(model):
    """Prototype P 분석"""
    P = model.model_layer.P.detach().cpu()
    K, D = P.shape
    
    # L2 Norms
    norms = torch.norm(P, p=2, dim=1)
    
    # Normalized Gram Matrix (Cosine Similarity)
    P_norm = F.normalize(P, p=2, dim=1)
    gram = P_norm @ P_norm.t()
    
    # Off-diagonal 분석
    off_diag_mask = 1 - torch.eye(K)
    off_diag = gram * off_diag_mask
    
    # 직교도 (1에 가까울수록 직교)
    orthogonality = 1 - off_diag.abs().mean().item()
    
    stats = {
        'K': K,
        'D': D,
        'norm_mean': norms.mean().item(),
        'norm_std': norms.std().item(),
        'norm_min': norms.min().item(),
        'norm_max': norms.max().item(),
        'orthogonality': orthogonality,
        'avg_cosine_sim': off_diag.abs().mean().item(),
        'max_cosine_sim': off_diag.abs().max().item(),
    }
    
    # Raw Gram Matrix (not normalized)
    raw_gram = P @ P.t()
    
    return stats, gram.numpy(), raw_gram.numpy(), norms.numpy()


def get_orthogonal_delta(embs, P, scale):
    """Gram-Schmidt로 delta 계산 (레이어와 동일한 로직)"""
    logits = torch.matmul(embs, P.t()) / scale
    w = F.softmax(logits, dim=-1)
    p_avg = torch.matmul(w, P)
    
    u_dot_p = torch.sum(embs * p_avg, dim=-1, keepdim=True)
    p_dot_p = torch.sum(p_avg * p_avg, dim=-1, keepdim=True) + 1e-8
    proj = (u_dot_p / p_dot_p) * p_avg
    
    delta = embs - proj
    return delta


def analyze_delta_distribution(model, data_loader, device, n_samples=1000):
    """Delta 분포 분석"""
    model.eval()
    
    P = model.model_layer.P.detach()
    D = P.shape[1]
    scale = D ** 0.5
    
    # 유저 샘플
    n_users = min(n_samples, data_loader.n_users)
    user_ids = torch.arange(n_users).to(device)
    user_embs = model.user_embedding(user_ids)
    with torch.no_grad():
        user_deltas = get_orthogonal_delta(user_embs, P, scale).cpu()
    
    # 아이템 샘플
    n_items = min(n_samples, data_loader.n_items)
    item_embs = model.item_embedding.weight[:n_items]
    with torch.no_grad():
        item_deltas = get_orthogonal_delta(item_embs, P, scale).cpu()
    
    def delta_stats(delta, prefix):
        norms = torch.norm(delta, p=2, dim=-1)
        return {
            f'{prefix}_delta_norm_mean': norms.mean().item(),
            f'{prefix}_delta_norm_std': norms.std().item(),
            f'{prefix}_delta_norm_min': norms.min().item(),
            f'{prefix}_delta_norm_max': norms.max().item(),
            # 원본 임베딩 대비 delta 비율
            f'{prefix}_delta_emb_ratio': norms.mean().item() / (D ** 0.5),
        }
    
    stats = {}
    stats.update(delta_stats(user_deltas, 'user'))
    stats.update(delta_stats(item_deltas, 'item'))
    
    user_delta_norms = torch.norm(user_deltas, p=2, dim=-1).numpy()
    item_delta_norms = torch.norm(item_deltas, p=2, dim=-1).numpy()
    
    return stats, user_delta_norms, item_delta_norms, user_deltas.numpy(), item_deltas.numpy()


def analyze_score_decomposition(model, data_loader, device, n_pairs=500):
    """Score 분해 분석 (Global vs Local 비중)"""
    model.eval()
    
    P = model.model_layer.P.detach()
    D = P.shape[1]
    scale = D ** 0.5
    
    # 모델의 learnable alpha 가져오기 (없으면 scale 사용)
    if hasattr(model.model_layer, 'alpha'):
        alpha = model.model_layer.alpha.detach().item()
    else:
        alpha = scale
    
    # 랜덤 유저-아이템 쌍
    n_users = min(n_pairs, data_loader.n_users)
    n_items = min(n_pairs, data_loader.n_items)
    
    user_ids = torch.randint(0, data_loader.n_users, (n_pairs,)).to(device)
    item_ids = torch.randint(0, data_loader.n_items, (n_pairs,)).to(device)
    
    user_embs = model.user_embedding(user_ids)
    item_embs = model.item_embedding(item_ids)
    
    with torch.no_grad():
        # Global Score
        u_proj = torch.matmul(user_embs, P.t())
        i_proj = torch.matmul(item_embs, P.t())
        global_scores = torch.sum(u_proj * i_proj, dim=-1).cpu()
        
        # Local Score (using model's alpha)
        delta_u = get_orthogonal_delta(user_embs, P, scale)
        u_intensity = torch.sum(user_embs * delta_u, dim=-1)
        i_conformity = torch.sum(item_embs * delta_u, dim=-1)
        local_scores = (u_intensity * i_conformity * alpha).cpu()
        
        # Total
        total_scores = (global_scores + local_scores)
    
    # 비중 분석
    global_abs = global_scores.abs()
    local_abs = local_scores.abs()
    total_abs = global_abs + local_abs + 1e-8
    
    global_ratio = (global_abs / total_abs).mean().item()
    local_ratio = (local_abs / total_abs).mean().item()
    
    stats = {
        'global_score_mean': global_scores.mean().item(),
        'global_score_std': global_scores.std().item(),
        'local_score_mean': local_scores.mean().item(),
        'local_score_std': local_scores.std().item(),
        'total_score_mean': total_scores.mean().item(),
        'global_contribution_ratio': global_ratio,
        'local_contribution_ratio': local_ratio,
        'learned_alpha': alpha,  # 학습된 alpha 값 저장
    }
    
    return stats, global_scores.numpy(), local_scores.numpy()


def compute_implicit_G(model, data_loader, device, n_samples=100):
    """
    암묵적 G 행렬 계산: G = P @ P.T + avg(δ @ δ.T)
    샘플 유저들의 δ를 평균하여 대표적인 G를 시각화
    """
    model.eval()
    
    P = model.model_layer.P.detach().cpu()
    D = P.shape[1]
    scale = D ** 0.5
    
    # Global component: P @ P.T
    G_global = P @ P.t()  # [D, D]
    
    # Local component: avg(δ @ δ.T)
    n_users = min(n_samples, data_loader.n_users)
    user_ids = torch.arange(n_users).to(device)
    user_embs = model.user_embedding(user_ids)
    
    with torch.no_grad():
        deltas = get_orthogonal_delta(user_embs, model.model_layer.P.detach(), scale).cpu()
    
    # 평균 delta outer product
    delta_outer_sum = torch.zeros(D, D)
    for delta in deltas:
        delta_outer_sum += torch.outer(delta, delta)
    G_local_avg = delta_outer_sum / n_users * scale  # sqrt(D) 스케일 적용
    
    # 합산
    G_implicit = G_global + G_local_avg
    
    stats = {
        'G_global_trace': torch.trace(G_global).item(),
        'G_local_trace': torch.trace(G_local_avg).item(),
        'G_global_frobenius': torch.norm(G_global, 'fro').item(),
        'G_local_frobenius': torch.norm(G_local_avg, 'fro').item(),
        'local_to_global_ratio': torch.norm(G_local_avg, 'fro').item() / (torch.norm(G_global, 'fro').item() + 1e-8),
    }
    
    return stats, G_global.numpy(), G_local_avg.numpy(), G_implicit.numpy()


def visualize_all(gram, raw_gram, norms, user_delta_norms, item_delta_norms, 
                  global_scores, local_scores, G_global, G_local, G_implicit,
                  save_path, model_name):
    """시각화"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'{model_name} Layer Analysis', fontsize=14)
    
    # Row 1: Prototype Analysis
    # 1. Prototype Cosine Similarity
    sns.heatmap(gram, ax=axes[0, 0], cmap='coolwarm', center=0, vmin=-1, vmax=1)
    axes[0, 0].set_title('P Cosine Similarity')
    
    # 2. Prototype Norms
    axes[0, 1].bar(range(len(norms)), norms)
    axes[0, 1].axhline(y=norms.mean(), color='r', linestyle='--', label=f'Mean: {norms.mean():.2f}')
    axes[0, 1].set_title('Prototype L2 Norms')
    axes[0, 1].set_xlabel('Prototype Index')
    axes[0, 1].legend()
    
    # 3. Delta Norm Distribution
    axes[0, 2].hist(user_delta_norms, bins=50, alpha=0.7, edgecolor='black', color='blue', label='User')
    axes[0, 2].hist(item_delta_norms, bins=50, alpha=0.5, edgecolor='black', color='orange', label='Item')
    axes[0, 2].set_title('Delta L2 Norm Distribution')
    axes[0, 2].set_xlabel('||δ||')
    axes[0, 2].legend()
    
    # Row 2: Score Analysis
    # 4. Global vs Local Score Scatter
    axes[1, 0].scatter(global_scores, local_scores, alpha=0.3, s=10)
    axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Global Score')
    axes[1, 0].set_ylabel('Local Score')
    axes[1, 0].set_title('Global vs Local Score')
    
    # 5. Score Distributions
    axes[1, 1].hist(global_scores, bins=50, alpha=0.7, label='Global', edgecolor='black')
    axes[1, 1].hist(local_scores, bins=50, alpha=0.5, label='Local', edgecolor='black')
    axes[1, 1].set_title('Score Distributions')
    axes[1, 1].legend()
    
    # 6. Raw Gram Matrix (P @ P.T)
    sns.heatmap(raw_gram, ax=axes[1, 2], cmap='coolwarm', center=0)
    axes[1, 2].set_title('Raw Gram Matrix (P @ P.T)')
    
    # Row 3: Implicit G Analysis
    # 7. G_global (P @ P.T in D x D)
    sns.heatmap(G_global, ax=axes[2, 0], cmap='coolwarm', center=0)
    axes[2, 0].set_title('G_global (P @ P.T)')
    
    # 8. G_local (avg δ @ δ.T)
    sns.heatmap(G_local, ax=axes[2, 1], cmap='coolwarm', center=0)
    axes[2, 1].set_title('G_local (avg δ @ δ.T)')
    
    # 9. G_implicit (Total)
    sns.heatmap(G_implicit, ax=axes[2, 2], cmap='coolwarm', center=0)
    axes[2, 2].set_title('G_implicit (Global + Local)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_single_experiment(exp_path: Path, device: str = 'cpu'):
    """단일 실험 분석"""
    print(f"\n{'='*60}")
    print(f"  {exp_path.name}")
    print(f"{'='*60}")
    
    try:
        model, data_loader, config = load_model(str(exp_path), device)
    except Exception as e:
        print(f"  [ERROR] 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 1. Prototype P 분석
    print("\n[1] Prototype P Analysis")
    print("-" * 40)
    p_stats, gram, raw_gram, norms = analyze_prototypes(model)
    for k, v in p_stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # 2. Delta 분포
    print("\n[2] Delta Distribution Analysis")
    print("-" * 40)
    delta_stats, user_delta_norms, item_delta_norms, _, _ = analyze_delta_distribution(
        model, data_loader, device
    )
    for k, v in delta_stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # 3. Score 분해
    print("\n[3] Score Decomposition Analysis")
    print("-" * 40)
    score_stats, global_scores, local_scores = analyze_score_decomposition(
        model, data_loader, device
    )
    for k, v in score_stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # 4. Implicit G 계산
    print("\n[4] Implicit G Analysis")
    print("-" * 40)
    g_stats, G_global, G_local, G_implicit = compute_implicit_G(
        model, data_loader, device
    )
    for k, v in g_stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # 5. Visualization
    save_path = exp_path / 'metric_layer_analysis.png'
    visualize_all(gram, raw_gram, norms, user_delta_norms, item_delta_norms,
                  global_scores, local_scores, G_global, G_local, G_implicit,
                  save_path, exp_path.name)
    print(f"\n[5] Saved: {save_path}")
    
    # 6. Save stats
    all_stats = {
        'prototypes': p_stats,
        'delta': delta_stats,
        'score_decomposition': score_stats,
        'implicit_G': g_stats,
    }
    stats_path = exp_path / 'metric_layer_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"    Saved: {stats_path}")
    
    return all_stats


def main():
    parser = argparse.ArgumentParser(description='CSAR Metric Layers 분석')
    parser.add_argument('dataset_path', type=str, 
                        help='데이터셋 결과 폴더 (예: trained_model/ml-1m)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='디바이스 (cpu/cuda/mps)')
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"[ERROR] 경로가 존재하지 않습니다: {dataset_path}")
        return
    
    # metric 실험들 찾기
    exp_dirs = find_metric_experiments(dataset_path)
    
    if not exp_dirs:
        print(f"[WARNING] Metric 모델을 찾을 수 없습니다: {dataset_path}")
        return
    
    print(f"\n{'#'*60}")
    print(f"# CSAR Metric Layers 분석: {dataset_path.name}")
    print(f"# 발견된 모델 수: {len(exp_dirs)}")
    print(f"{'#'*60}")
    
    for exp_dir in exp_dirs:
        analyze_single_experiment(exp_dir, args.device)
    
    print(f"\n{'='*60}")
    print("모든 분석 완료!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
