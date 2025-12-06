import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 프로젝트 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.utils import load_model_from_run, get_analysis_output_path

def calculate_gini(array):
    """ 지니 계수 계산 """
    array = np.sort(array)
    # 0이 포함되면 계산 오류가 날 수 있으므로 아주 작은 값 더함
    array = array + 1e-9
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((2 * index - n - 1) * array).sum() / (n * array.sum())

def calculate_entropy(probs):
    """ 섀넌 엔트로피 계산 (확률 분포 입력 가정) """
    # 정규화 (Sum=1)
    probs = probs / np.sum(probs)
    # log(0) 방지
    probs = probs[probs > 0] 
    return -np.sum(probs * np.log2(probs))

def analyze_soft_cluster_balance(exp_config):
    run_folder_path = exp_config['run_folder_path']
    print(f"\nRunning Soft Cluster Balance Analysis for: {run_folder_path}")

    # 1. 모델 로드
    model, data_loader, config = load_model_from_run(run_folder_path)
    if not model: return

    # 2. 관심사 가중치 계산 (Soft Weights)
    # (Num_Items, Num_Interests)
    if hasattr(model, 'get_final_item_embeddings'):
        item_interests = model.get_final_item_embeddings().detach().cpu()
    else:
        item_interests = model.attention_layer(model.item_embedding.weight).detach().cpu()
    
    # 3. [핵심 수정] 가중치 총합 계산 (Soft Volume)
    # 각 관심사가 전체 아이템 공간에서 차지하는 "총 에너지(Total Energy)"
    interest_volumes = torch.sum(item_interests, dim=0).numpy() # (Num_Interests,)
    
    # 4. Hard Assignment Calculation (User Request)
    num_interests = model.num_interests  # Moved up
    
    # 아이템별로 가장 강한 관심사(argmax)를 카운트
    # (Num_Items,)
    hard_assignments = torch.argmax(item_interests, dim=1).numpy()
    hard_counts = np.bincount(hard_assignments, minlength=num_interests)
    
    # 5. 지표 계산
    total_volume = np.sum(interest_volumes)
    total_items = len(hard_assignments)
    
    # 정규화된 분포 (Probability Distribution)
    interest_probs = interest_volumes / total_volume
    
    gini_soft = calculate_gini(interest_volumes)
    entropy_soft = calculate_entropy(interest_volumes)
    
    gini_hard = calculate_gini(hard_counts)
    entropy_hard = calculate_entropy(hard_counts)
    
    max_entropy = np.log2(num_interests)
    utilization_rate = entropy_soft / max_entropy * 100

    print(f"\n[Soft Cluster Balance Stats]")
    print(f"- Total Weight Volume: {total_volume:.2f}")
    print(f"- Gini Index (Soft): {gini_soft:.4f}")
    print(f"- Entropy (Soft): {entropy_soft:.4f} (Max: {max_entropy:.4f})")
    print(f"- Soft Utilization: {utilization_rate:.2f}%")
    print(f"- Min Volume: {np.min(interest_volumes):.2f}")
    print(f"- Max Volume: {np.max(interest_volumes):.2f}")
    
    print(f"\n[Hard Cluster Balance Stats]")
    print(f"- Total Items: {total_items}")
    print(f"- Gini Index (Hard): {gini_hard:.4f}")
    print(f"- Entropy (Hard): {entropy_hard:.4f}")
    print(f"- Min Count: {np.min(hard_counts)}")
    print(f"- Max Count: {np.max(hard_counts)}")
    print(f"- Empty Clusters: {np.sum(hard_counts == 0)}")

    # 6. 시각화 (Dual Axis Chart)
    output_path = get_analysis_output_path(config['dataset_name'], os.path.basename(run_folder_path))
    save_file = os.path.join(output_path, "soft_vs_hard_balance.png")
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Axis 1: Soft Volume (Bar)
    colors = plt.cm.viridis(interest_probs / np.max(interest_probs))
    bars = ax1.bar(range(num_interests), interest_volumes, color=colors, alpha=0.7, label='Soft Volume (Sum)')
    
    ax1.set_xlabel("Interest Key ID", fontsize=12)
    ax1.set_ylabel("Soft Volume (Total Weight)", fontsize=12, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_title(f"Interest Balance: Soft Sum vs Hard Count\n(Soft Gini: {gini_soft:.2f}, Hard Gini: {gini_hard:.2f})", fontsize=14)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    # Axis 2: Hard Count (Line/Marker)
    ax2 = ax1.twinx()
    ax2.plot(range(num_interests), hard_counts, color='tab:red', marker='o', linestyle='-', linewidth=2, markersize=6, label='Hard Count (Items)')
    ax2.set_ylabel("Hard Count (Num Items)", fontsize=12, color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # 레전드 통합
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    
    # X축 틱 (너무 많으면 생략)
    if num_interests <= 50:
        ax1.set_xticks(range(num_interests))
    else:
        ax1.set_xticks(range(0, num_interests, 5))

    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    print(f"Saved plot to: {save_file}")

if __name__ == '__main__':
    EXPERIMENTS = [{'run_folder_path': '/Users/leejongmin/code/recsys_framework/trained_model/ml-1m/csar-bpr-ce__temperature=0.8'}]
    for exp in EXPERIMENTS:
        if os.path.exists(exp['run_folder_path']): analyze_soft_cluster_balance(exp)