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
    
    # 4. 지표 계산
    num_interests = model.num_interests
    total_volume = np.sum(interest_volumes)
    
    # 정규화된 분포 (Probability Distribution)
    interest_probs = interest_volumes / total_volume
    
    gini = calculate_gini(interest_volumes)
    entropy = calculate_entropy(interest_volumes)
    max_entropy = np.log2(num_interests)
    utilization_rate = entropy / max_entropy * 100

    print(f"\n[Soft Cluster Balance Stats]")
    print(f"- Total Weight Volume: {total_volume:.2f}")
    print(f"- Gini Index (Soft): {gini:.4f}")
    print(f"- Entropy (Soft): {entropy:.4f} (Max: {max_entropy:.4f})")
    print(f"- Soft Utilization: {utilization_rate:.2f}%")
    print(f"- Min Volume: {np.min(interest_volumes):.2f}")
    print(f"- Max Volume: {np.max(interest_volumes):.2f}")

    # 5. 시각화 (Bar Chart)
    output_path = get_analysis_output_path(config['dataset_name'], os.path.basename(run_folder_path))
    save_file = os.path.join(output_path, "soft_cluster_balance.png")
    
    plt.figure(figsize=(12, 6))
    
    # 막대 그래프 (Soft Volume)
    colors = plt.cm.viridis(interest_probs / np.max(interest_probs)) # 비중에 따라 색상 진하게
    bars = plt.bar(range(num_interests), interest_volumes, color=colors, edgecolor='black', alpha=0.8)
    
    # 평균선
    avg_vol = np.mean(interest_volumes)
    plt.axhline(y=avg_vol, color='red', linestyle='--', label=f'Average Volume ({avg_vol:.1f})')
    
    plt.title(f"Soft Interest Distribution (Total Weight Sum)\n(Entropy: {entropy:.2f}, Gini: {gini:.2f})", fontsize=14)
    plt.xlabel("Interest Key ID", fontsize=12)
    plt.ylabel("Total Attention Weight (Sum)", fontsize=12)
    plt.xticks(range(num_interests))
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 비율 표시
    for bar, prob in zip(bars, interest_probs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{prob*100:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    print(f"Saved plot to: {save_file}")

if __name__ == '__main__':
    EXPERIMENTS = [{'run_folder_path': 'trained_model/amazon_music/csar-bpr__negative_sampling_strategy=uniform'}]
    for exp in EXPERIMENTS:
        if os.path.exists(exp['run_folder_path']): analyze_soft_cluster_balance(exp)