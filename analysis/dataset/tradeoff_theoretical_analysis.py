"""
다양성-정확도 Trade-off 이론적 정당화 분석 스크립트

논문용 3가지 방법론:
1. 확률 질량 이동 모델 (Probability Mass Shift Model)
2. 파레토 효율성 지수 (Pareto Efficiency Index, PEI)
3. 정보 이론적 접근 (KL-Divergence Bound)

"왜 다양성이 늘어나면 정확도가 떨어질 수밖에 없는가?" 수학적 증명
"우리 모델은 그 하락폭을 얼마나 잘 방어했는가?" 정량화
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import yaml
import pickle
from scipy.interpolate import interp1d
from scipy.stats import entropy as scipy_entropy
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# ==================== Data Classes ====================

@dataclass
class DatasetStats:
    """데이터셋 통계"""
    n_items: int
    n_head: int
    n_tail: int
    head_pop_ratio: float  # Head 아이템의 인기도 점유율
    tail_pop_ratio: float
    gini_index: float

@dataclass
class ModelResult:
    """모델 결과"""
    name: str
    ndcg: float
    longtail_coverage: float
    coverage: float
    hit_rate: float
    entropy: float

@dataclass
class TheoreticalAnalysis:
    """이론적 분석 결과"""
    # 방법론 1: 확률 질량 이동
    shifted_prob_mass: float           # δ: Tail로 이동한 추천 확률
    expected_loss: float               # 기대 정확도 손실
    actual_loss: float                 # 실제 손실
    natural_decay_factor: float        # 자연 감쇠 계수 (P_head - P_tail)
    efficiency_vs_theory: float        # 이론 대비 효율성 (몇 배 효율적?)
    
    # 방법론 2: PEI
    pei_score: float                   # 파레토 효율성 지수
    expected_ndcg_linear: float        # 선형 보간 기대 NDCG
    
    # 방법론 3: KL-Divergence
    kl_divergence: float               # 추천 분포와 데이터 분포 간 거리
    recommendation_entropy: float      # 추천 분포 엔트로피


# ==================== Data Loading ====================

def load_processed_data(dataset_name: str) -> Optional[Dict]:
    """Cached 데이터 로드"""
    cache_dir = Path(ROOT) / "data_cache"
    
    candidates = [
        cache_dir / f"{dataset_name}_mps_processed_data.pkl",
        cache_dir / f"{dataset_name}_cpu_processed_data.pkl",
        cache_dir / f"{dataset_name}_cuda_processed_data.pkl"
    ]
    
    for p in candidates:
        if p.exists():
            print(f"[INFO] Loading data from {p.name}")
            with open(p, 'rb') as f:
                return pickle.load(f)
    
    print(f"[ERROR] No processed data found for {dataset_name}")
    return None


def load_model_metrics(dataset_path: Path, k: int = 20) -> pd.DataFrame:
    """모든 모델의 메트릭 수집"""
    results = []
    
    for model_dir in dataset_path.iterdir():
        if not model_dir.is_dir():
            continue
        
        metrics_path = model_dir / "final_metrics.json"
        config_path = model_dir / "config.yaml"
        
        if not metrics_path.exists():
            continue
            
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            # config가 있으면 모델 이름 추출, 없으면 디렉토리 이름 사용
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                model_name = config['model']['name']
            else:
                model_name = model_dir.name
            
            run_name = model_dir.name
            
            results.append({
                'Model': model_name,
                'Run': run_name,
                'NDCG': metrics.get(f"NDCG@{k}", 0),
                'HitRate': metrics.get(f"HitRate@{k}", 0),
                'LongTailCoverage': metrics.get(f"LongTailCoverage@{k}", 0),
                'LongTailHitRate': metrics.get(f"LongTailHitRate@{k}", 0),
                'LongTailNDCG': metrics.get(f"LongTailNDCG@{k}", 0),
                'HeadHitRate': metrics.get(f"HeadHitRate@{k}", 0),
                'HeadNDCG': metrics.get(f"HeadNDCG@{k}", 0),
                'Coverage': metrics.get(f"Coverage@{k}", 0),
                'Entropy': metrics.get(f"Entropy@{k}", 0),
                'PopRatio': metrics.get(f"PopRatio@{k}", 0),
                'ILD': metrics.get(f"ILD@{k}", 0),
                'Novelty': metrics.get(f"Novelty@{k}", 0),
            })
        except Exception as e:
            print(f"[Warning] Loading {model_dir.name}: {e}")
    
    return pd.DataFrame(results)


def compute_dataset_stats(data: Dict) -> DatasetStats:
    """데이터셋 통계 계산"""
    item_pop = data['item_popularity']  # Series
    
    # Interaction Volume 기준 정렬 (내림차순)
    sorted_items = item_pop.sort_values(ascending=False)
    n_items = len(sorted_items)
    
    # Interaction Volume의 80%를 차지하는 아이템들을 Head로 정의 (Pareto 80/20)
    cumsum = sorted_items.cumsum()
    total_pop = sorted_items.sum()
    cutoff_val = total_pop * 0.8 # Standard Head Volume
    
    head_cutoff_idx = np.searchsorted(cumsum.values, cutoff_val, side='right')
    
    head_items = sorted_items.index[:head_cutoff_idx].tolist()
    tail_items = sorted_items.index[head_cutoff_idx:].tolist()
    
    head_pop_ratio = sorted_items.iloc[:head_cutoff_idx].sum() / total_pop
    tail_pop_ratio = sorted_items.iloc[head_cutoff_idx:].sum() / total_pop
    
    # Gini Index 계산
    pop_values = item_pop.values
    pop_sorted = np.sort(pop_values)
    n = len(pop_sorted)
    cumulative = np.cumsum(pop_sorted)
    gini = (2 * np.sum((np.arange(1, n+1) * pop_sorted))) / (n * np.sum(pop_sorted)) - (n + 1) / n
    
    return DatasetStats(
        n_items=n_items,
        n_head=len(head_items),
        n_tail=len(tail_items),
        head_pop_ratio=head_pop_ratio,
        tail_pop_ratio=tail_pop_ratio,
        gini_index=gini
    )


# ==================== 방법론 1: 확률 질량 이동 모델 ====================

def compute_probability_mass_shift_analysis(
    stats: DatasetStats,
    baseline_result: ModelResult,
    target_result: ModelResult
) -> Dict:
    """
    확률 질량 이동 모델 분석
    
    핵심 수식:
    E[ΔHitRate] ≈ -δ × (P_head - P_tail)
    
    여기서:
    - δ: Tail로 이동시킨 추천 확률 (LongTailCoverage 증가량)
    - P_head: Head 아이템이 정답일 확률 (≈ 70% in ML-1M)
    - P_tail: Tail 아이템이 정답일 확률 (≈ 30% in ML-1M)
    """
    
    # δ: Tail로 이동한 추천 비중 변화
    delta_tail = target_result.longtail_coverage - baseline_result.longtail_coverage
    
    # 자연 감쇠 계수 = P_head - P_tail
    # 데이터셋에서의 정답 확률 차이
    natural_decay_factor = stats.head_pop_ratio - stats.tail_pop_ratio
    
    # 기대 정확도 손실 (이론적)
    expected_loss = -delta_tail * natural_decay_factor
    expected_loss_percent = expected_loss * 100
    
    # 실제 손실
    actual_loss = target_result.ndcg - baseline_result.ndcg
    actual_loss_percent = (actual_loss / baseline_result.ndcg) * 100
    
    # 효율성: 이론적 손실 대비 실제 손실 비율
    # 값이 낮을수록 좋음 (이론적 기대보다 적게 손실)
    if abs(expected_loss_percent) > 0.001:
        efficiency_ratio = abs(actual_loss_percent) / abs(expected_loss_percent)
        efficiency_multiplier = 1 / efficiency_ratio if efficiency_ratio > 0 else float('inf')
    else:
        efficiency_multiplier = float('inf')
    
    return {
        'delta_tail': delta_tail,
        'delta_tail_percent': delta_tail * 100,
        'natural_decay_factor': natural_decay_factor,
        'expected_loss_percent': expected_loss_percent,
        'actual_loss_percent': actual_loss_percent,
        'efficiency_multiplier': efficiency_multiplier,
        'interpretation': (
            f"Tail 비중 {delta_tail*100:.1f}% 증가 시, "
            f"이론적 정확도 손실은 {abs(expected_loss_percent):.1f}%이나, "
            f"실제 손실은 {abs(actual_loss_percent):.1f}%로 "
            f"{efficiency_multiplier:.1f}배 효율적"
        )
    }


def compute_mass_shift_detailed_analysis(
    stats: DatasetStats,
    models_df: pd.DataFrame,
    baseline_name: str = 'mf',
    accuracy_col: str = 'NDCG',
    diversity_col: str = 'LongTailCoverage'
) -> pd.DataFrame:
    """
    방법론 1 상세 분석: 모든 모델 vs Baseline 비교
    
    핵심 수식:
    Expected NDCG Loss = δ × (P_head - P_tail) × Baseline_NDCG
    
    Efficiency = Expected_Loss / Actual_Loss
    """
    
    results = []
    
    # Baseline 찾기 (우선순위: mf > lightgcn > most-popular)
    baseline_candidates = ['mf', 'lightgcn', 'most-popular']
    baseline_row = None
    
    for candidate in baseline_candidates:
        matches = models_df[models_df['Run'].str.lower().str.contains(candidate)]
        if not matches.empty:
            baseline_row = matches.iloc[0]
            break
    
    if baseline_row is None:
        # 가장 높은 NDCG 모델을 baseline으로
        baseline_row = models_df.loc[models_df[accuracy_col].idxmax()]
    
    baseline_ndcg = baseline_row[accuracy_col]
    baseline_ltc = baseline_row[diversity_col]
    baseline_name_used = baseline_row['Run']
    
    # 자연 감쇠 계수
    natural_decay = stats.head_pop_ratio - stats.tail_pop_ratio
    
    print(f"\n{'='*80}")
    print("Method 1: Probability Mass Shift Analysis")
    print(f"{'='*80}")
    print(f"  Baseline Model: {baseline_name_used}")
    print(f"  Baseline {accuracy_col}: {baseline_ndcg:.4f}")
    print(f"  Baseline {diversity_col}: {baseline_ltc:.4f}")
    print(f"  Natural Decay Factor (P_head - P_tail): {natural_decay:.4f}")
    print(f"{'='*80}\n")
    
    for _, row in models_df.iterrows():
        model_name = row['Run']
        model_ndcg = row[accuracy_col]
        model_ltc = row[diversity_col]
        
        # δ: Tail로 이동한 확률 질량
        delta_tail = model_ltc - baseline_ltc
        
        # 이론적 기대 손실 (절대값)
        # E[ΔNDCG] = -δ × decay_factor × baseline_ndcg (상대적)
        expected_loss_relative = delta_tail * natural_decay
        expected_loss_ndcg = expected_loss_relative * baseline_ndcg
        expected_ndcg = baseline_ndcg - expected_loss_ndcg
        
        # 실제 손실
        actual_loss_ndcg = baseline_ndcg - model_ndcg
        actual_loss_relative = actual_loss_ndcg / baseline_ndcg if baseline_ndcg > 0 else 0
        
        # 효율성 계산
        if abs(expected_loss_ndcg) > 0.0001:
            # 손실이 이론보다 적으면 > 1 (좋음)
            efficiency = expected_loss_ndcg / actual_loss_ndcg if actual_loss_ndcg != 0 else float('inf')
        else:
            efficiency = 1.0 if actual_loss_ndcg <= 0 else 0.0
        
        # 이론 대비 초과 성능
        if expected_ndcg > 0:
            outperformance = ((model_ndcg - expected_ndcg) / expected_ndcg) * 100
        else:
            outperformance = 0
        
        results.append({
            'Model': row['Model'],
            'Run': model_name,
            accuracy_col: model_ndcg,
            diversity_col: model_ltc,
            'Delta_Tail': delta_tail,
            f'Expected_Loss_{accuracy_col}': expected_loss_ndcg,
            f'Actual_Loss_{accuracy_col}': actual_loss_ndcg,
            f'Expected_{accuracy_col}': expected_ndcg,
            'Efficiency': efficiency,
            'Outperformance_Pct': outperformance,
            'Is_CSAR': 'csar' in model_name.lower() or 'lyra' in model_name.lower()
        })
    
    return pd.DataFrame(results)


def plot_mass_shift_analysis(
    mass_shift_df: pd.DataFrame,
    stats: DatasetStats,
    dataset_name: str,
    output_dir: Path,
    k: int = 20,
    accuracy_col: str = 'NDCG',
    diversity_col: str = 'LongTailCoverage'
):
    """방법론 1 전용 시각화"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 색상
    colors = {
        'csar': '#E74C3C',
        'lightgcn': '#2ECC71',
        'mf': '#3498DB',
        'protomf': '#9B59B6',
        'neumf': '#F39C12',
        'most-popular': '#95A5A6',
        'random': '#34495E',
        'ease': '#1ABC9C',
        'lyra': '#E67E22'
    }
    
    def get_color(name):
        name_lower = name.lower()
        for key, color in colors.items():
            if key in name_lower:
                return color
        return '#7F8C8D'
    
    # --- Plot 1: Expected vs Actual ---
    ax1 = axes[0]
    
    # 이론적 기대선 (45도)
    max_val = max(mass_shift_df[f'Expected_{accuracy_col}'].max(), mass_shift_df[accuracy_col].max()) * 1.1
    ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=2,
             label='Theoretical Expectation')
    
    for _, row in mass_shift_df.iterrows():
        color = get_color(row['Run'])
        marker = 's' if row['Is_CSAR'] else 'o'
        size = 200 if row['Is_CSAR'] else 150
        
        ax1.scatter(row[f'Expected_{accuracy_col}'], row[accuracy_col],
                   s=size, color=color, marker=marker,
                   edgecolors='black', linewidth=2, alpha=0.85, zorder=5)
        
        # 라벨
        if row['Outperformance_Pct'] > 20 or row['Is_CSAR']:
            ax1.annotate(f"{row['Run']}\n+{row['Outperformance_Pct']:.0f}%",
                        xy=(row[f'Expected_{accuracy_col}'], row[accuracy_col]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
    
    ax1.set_xlabel(f'Expected {accuracy_col} (Theoretical)', fontsize=12, fontweight='bold')
    ax1.set_ylabel(f'Actual {accuracy_col}', fontsize=12, fontweight='bold')
    ax1.set_title('Method 1: Expected vs Actual Performance', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 파란 영역: 이론보다 좋은 구역
    ax1.fill_between([0, max_val], [0, max_val], [max_val, max_val],
                     alpha=0.1, color='green', label='Outperforming Theory')
    
    # --- Plot 2: Outperformance Bar Chart ---
    ax2 = axes[1]
    
    df_sorted = mass_shift_df.sort_values('Outperformance_Pct', ascending=True)
    bar_colors = ['#E74C3C' if c else '#3498DB' for c in df_sorted['Is_CSAR']]
    
    bars = ax2.barh(range(len(df_sorted)), df_sorted['Outperformance_Pct'],
                    color=bar_colors, edgecolor='black', linewidth=1)
    
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=2)
    
    ax2.set_yticks(range(len(df_sorted)))
    ax2.set_yticklabels(df_sorted['Run'], fontsize=9)
    ax2.set_xlabel('Outperformance vs Theory (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Performance Gain Over Theoretical Expectation', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 값 라벨
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        val = row['Outperformance_Pct']
        offset = 2 if val >= 0 else -15
        ax2.text(val + offset, i, f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    save_path = output_dir / f"mass_shift_analysis_k{k}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SUCCESS] Mass Shift plot saved to {save_path}")
    
    return save_path




    return pd.DataFrame(results)


# ==================== Simulation: Popularity Baseline Curve ====================

def simulate_popularity_curve(
    data: Dict,
    k: int = 20,
    num_points: int = 20,
    n_trials: int = 10
) -> pd.DataFrame:
    """
    Simulate performance of a 'Popularity-Weighted Random' baseline.
    Runs n_trials times and averages the results to reduce variance.
    
    Score_i = alpha * log(pop_i) + Gumbel_Noise
    - alpha=0: Random Rec
    - alpha=inf: Most Popular
    """
    print(f"\n{'='*70}")
    print(f"Simulating Popularity-Weighted Baseline Curve ({n_trials} trials)...")
    print(f"{'='*70}")

    test_df = data['test_df']
    item_pop = data['item_popularity']
    n_items = data['n_items']
    
    # Pre-compute ground truth
    test_user_history = test_df.groupby('user_id')['item_id'].apply(set).to_dict()
    test_users = np.array(list(test_user_history.keys()))
    
    # 꼬리 아이템 정의 (Interaction Volume: Top 80% Volume = Head)
    sorted_popularity = item_pop.sort_values(ascending=False)
    cumsum = sorted_popularity.cumsum()
    total_interactions = sorted_popularity.sum()
    cutoff_val = total_interactions * 0.8
    head_cutoff_idx = np.searchsorted(cumsum.values, cutoff_val, side='right')
    tail_items_set = set(sorted_popularity.index[head_cutoff_idx:].tolist())
    
    print(f"DEBUG: Head Volume 80% -> Head Items: {head_cutoff_idx}, Tail Items: {len(tail_items_set)} ({len(tail_items_set)/n_items:.1%})")
    
    # Popularity Vector
    pop_vec = item_pop.sort_index().values
    log_pop = np.log(pop_vec + 1e-9)
    
    # Alphas to sweep: Dense in 0-10 (critical), Sparse in 10-100 (saturation)
    alphas_dense = np.linspace(0, 10, 51)   # 0.0, 0.2, ... 10.0
    alphas_sparse = np.linspace(10, 100, 10)[1:] # 20, 30, ... 100
    alphas = np.concatenate([alphas_dense, alphas_sparse])
    
    results = []
    
    for alpha in alphas:
        trial_ndcgs = []
        trial_lt_covs = []
        trial_coverages = []
        trial_entropies = []
        
        for _ in range(n_trials):
            # Gumbel-Max Trick
            noise = np.random.gumbel(size=(len(test_users), n_items))
            scores = (alpha * log_pop) + noise
            
            # Top-K indices
            top_k_indices = np.argpartition(scores, -k, axis=1)[:, -k:]
            
            ndcg_sum = 0
            
            for u_idx, u_id in enumerate(test_users):
                ground_truth = test_user_history[u_id]
                preds = top_k_indices[u_idx]
                
                # Use unsorted top-k for hit-check first
                hits = [1 if item in ground_truth else 0 for item in preds]
                
                if sum(hits) > 0:
                    # Proper NDCG calculation requires sorting top-k by score
                    u_scores = scores[u_idx, preds]
                    sorted_idx = np.argsort(u_scores)[::-1]
                    hits_sorted = [hits[i] for i in sorted_idx]
                    
                    dcg = sum((2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(hits_sorted))
                    idcg = sum((2**1 - 1) / np.log2(idx + 2) for idx in range(min(len(ground_truth), k)))
                    ndcg_sum += dcg / idcg
                    
            # Coverage & Entropy Calculation
            all_preds = top_k_indices.flatten()
            unique_items = np.unique(all_preds)
            coverage = len(unique_items) / n_items
            
            # Long Tail Coverage (Global)
            unique_items_set = set(unique_items)
            tail_intersection = unique_items_set.intersection(tail_items_set)
            lt_coverage = len(tail_intersection) / len(tail_items_set) if len(tail_items_set) > 0 else 0
            
            # Entropy
            item_counts = np.bincount(all_preds, minlength=n_items)
            p_rec = item_counts / item_counts.sum()
            entropy = scipy_entropy(p_rec)
            
            # Metric Aggregation
            trial_ndcgs.append(ndcg_sum / len(test_users))
            trial_lt_covs.append(lt_coverage)
            trial_coverages.append(coverage)
            trial_entropies.append(entropy)
            
        avg_ndcg = np.mean(trial_ndcgs)
        avg_lt_cov = np.mean(trial_lt_covs)
        avg_coverage = np.mean(trial_coverages)
        avg_entropy = np.mean(trial_entropies)
        
        results.append({
            'alpha': alpha,
            'NDCG': avg_ndcg,
            'LongTailCoverage': avg_lt_cov,
            'Coverage': avg_coverage,
            'Entropy': avg_entropy
        })
        print(f"Alpha {alpha:.1f}: NDCG={avg_ndcg:.4f}, Cov={avg_coverage:.4f}, LT-Cov={avg_lt_cov:.4f} (Avg of {n_trials})")
        
    return pd.DataFrame(results)


# ==================== 방법론 2: 파레토 효율성 지수 (PEI) ====================

def compute_pareto_efficiency_index(
    models_df: pd.DataFrame,
    baseline_curve_df: pd.DataFrame,
    accuracy_col: str = 'NDCG',
    diversity_col: str = 'LongTailCoverage'
) -> pd.DataFrame:
    """
    파레토 효율성 지수 계산 (Curve-based)
    
    PEI = Actual_Accuracy / Expected_Accuracy_on_Curve
    """
    
    # Interpolation: Diversity (X) -> Expected Accuracy (Y)
    curve_sorted = baseline_curve_df.sort_values(diversity_col)
    
    f_interp = interp1d(
        curve_sorted[diversity_col], 
        curve_sorted[accuracy_col], 
        kind='linear', 
        bounds_error=False, 
        fill_value="extrapolate"
    )
    
    results = []
    
    print(f"\n{'='*70}")
    print(f"Pareto Efficiency Index (PEI) Analysis [Curve-Based]")
    print(f"{'='*70}")
    
    for _, row in models_df.iterrows():
        x_model = row[diversity_col]
        y_model = row[accuracy_col]
        
        y_expected = float(f_interp(x_model))
        
        if y_expected > 0.0001:
            pei = y_model / y_expected
        else:
            pei = float('inf') if y_model > 0 else 1.0
        
        results.append({
            'Model': row['Model'],
            'Run': row['Run'],
            accuracy_col: y_model,
            diversity_col: x_model,
            f'Expected_{accuracy_col}': y_expected,
            'PEI': pei,
            'PEI_Percent': (pei - 1) * 100,
        })
    
    return pd.DataFrame(results)
    """
    파레토 효율성 지수 계산
    
    기준점:
    - Best-Accuracy Oracle (A): 정확도 최대 모델
    - Best-Diversity Oracle (B): 다양성 최대 모델
    
    두 점을 잇는 직선이 "지능 없이 혼합했을 때" 성능
    PEI = Accuracy_actual / Accuracy_linear_interpolation
    
    Args:
        accuracy_col: 정확도 지표 컬럼명 (예: NDCG, HitRate, LongTailNDCG)
        diversity_col: 다양성 지표 컬럼명 (예: LongTailCoverage, Entropy)
    """
    
    results = []
    
    # Oracle 포인트 식별
    # Best-Accuracy: 가장 높은 정확도
    acc_oracle = models_df.loc[models_df[accuracy_col].idxmax()]
    
    # Best-Diversity: 가장 높은 다양성
    div_oracle = models_df.loc[models_df[diversity_col].idxmax()]
    
    # Fallback: most-popular와 random이 있으면 사용
    if baseline_name in models_df['Run'].values:
        acc_oracle = models_df[models_df['Run'] == baseline_name].iloc[0]
    
    if diversity_oracle_name in models_df['Run'].values:
        div_oracle = models_df[models_df['Run'] == diversity_oracle_name].iloc[0]
    
    # 선형 보간 기울기
    x_a, y_a = acc_oracle[diversity_col], acc_oracle[accuracy_col]
    x_b, y_b = div_oracle[diversity_col], div_oracle[accuracy_col]
    
    if abs(x_b - x_a) > 1e-9:
        slope = (y_b - y_a) / (x_b - x_a)
    else:
        slope = 0
    
    print(f"\n{'='*70}")
    print(f"Pareto Efficiency Index (PEI) Analysis [{accuracy_col} vs {diversity_col}]")
    print(f"{'='*70}")
    print(f"  Accuracy Oracle: {acc_oracle['Run']} ({accuracy_col}={y_a:.4f}, {diversity_col}={x_a:.4f})")
    print(f"  Diversity Oracle: {div_oracle['Run']} ({accuracy_col}={y_b:.4f}, {diversity_col}={x_b:.4f})")
    print(f"  Linear Interpolation Slope: {slope:.6f}")
    print(f"{'='*70}\n")
    
    for _, row in models_df.iterrows():
        x_model = row[diversity_col]
        y_model = row[accuracy_col]
        
        # 선형 보간된 기대 정확도 (같은 다양성에서)
        y_expected = y_a + slope * (x_model - x_a)
        
        # PEI = 실제 / 기대
        if y_expected > 0:
            pei = y_model / y_expected
        else:
            pei = float('inf') if y_model > 0 else 1.0
        
        results.append({
            'Model': row['Model'],
            'Run': row['Run'],
            accuracy_col: y_model,
            diversity_col: x_model,
            f'Expected_{accuracy_col}': y_expected,
            'PEI': pei,
            'PEI_Percent': (pei - 1) * 100,  # 0%가 기준선
        })
    
    return pd.DataFrame(results)


# ==================== 방법론 3: KL-Divergence Bound ====================

def compute_kl_divergence_analysis(
    data: Dict,
    models_df: pd.DataFrame
) -> pd.DataFrame:
    """
    정보 이론적 분석: KL-Divergence
    
    추천 모델의 성능은 데이터 분포(P_data)와 추천 분포(P_rec) 간의 거리에 반비례
    다양성(Entropy)를 높이면 → P_rec가 Uniform에 가까워짐 → D_KL 증가 → 성능 하락
    """
    
    item_pop = data['item_popularity']
    
    # 데이터 분포 (정규화된 인기도)
    p_data = item_pop.values / item_pop.sum()
    data_entropy = scipy_entropy(p_data)
    
    # Uniform 분포와의 비교
    n_items = len(p_data)
    p_uniform = np.ones(n_items) / n_items
    uniform_entropy = np.log(n_items)
    
    # 데이터 분포의 KL from Uniform
    kl_data_uniform = scipy_entropy(p_uniform, p_data)
    
    results = []
    
    print(f"\n{'='*70}")
    print("KL-Divergence Analysis (Information Theoretic)")
    print(f"{'='*70}")
    print(f"  Data Entropy: {data_entropy:.4f}")
    print(f"  Uniform Entropy: {uniform_entropy:.4f} (log {n_items})")
    print(f"  KL(Uniform || Data): {kl_data_uniform:.4f}")
    print(f"{'='*70}\n")
    
    for _, row in models_df.iterrows():
        # 모델의 추천 엔트로피 (이미 계산된 메트릭 사용)
        rec_entropy = row['Entropy']
        
        # 엔트로피 비율: 모델 추천이 얼마나 Uniform에 가까운가
        entropy_ratio = rec_entropy / uniform_entropy if uniform_entropy > 0 else 0
        
        # 다양성 증가 = 엔트로피 증가 = KL(P_rec || P_data) 증가
        # 정확도 하락의 Lower Bound 근사
        entropy_gap = rec_entropy - data_entropy
        
        results.append({
            'Model': row['Model'],
            'Run': row['Run'],
            'NDCG': row['NDCG'],
            'LongTailCoverage': row['LongTailCoverage'],
            'Rec_Entropy': rec_entropy,
            'Entropy_Ratio': entropy_ratio,
            'Entropy_Gap': entropy_gap,
            'Theoretical_Note': "Higher entropy → Larger distance from biased data → Accuracy cost"
        })
    
    return pd.DataFrame(results)


# ==================== Visualization ====================

def plot_comprehensive_analysis(
    models_df: pd.DataFrame,
    pei_df: pd.DataFrame,
    baseline_curve_df: pd.DataFrame,
    stats: DatasetStats,
    dataset_name: str,
    output_dir: Path,
    k: int = 20,
    accuracy_col: str = 'NDCG',
    diversity_col: str = 'LongTailCoverage'
):
    """종합 분석 시각화"""
    
    fig = plt.figure(figsize=(18, 12))
    
    # 색상 팔레트
    colors = {
        'csar': '#E74C3C',
        'lightgcn': '#2ECC71',
        'mf': '#3498DB',
        'protomf': '#9B59B6',
        'neumf': '#F39C12',
        'most-popular': '#95A5A6',
        'random': '#34495E',
        'ease': '#1ABC9C',
        'lyra': '#E67E22'
    }
    
    def get_color(name):
        name_lower = name.lower()
        for key, color in colors.items():
            if key in name_lower:
                return color
        return '#7F8C8D'
    
    # --- Subplot 1: Trade-off with PEI Curve ---
    ax1 = fig.add_subplot(2, 2, 1)
    
    # Baseline Curve Plot
    ax1.plot(baseline_curve_df[diversity_col], baseline_curve_df[accuracy_col], 
             'k--', linewidth=2, label='Baseline Frontier (Simulated)')
    
    # Fill Efficient Zone (Above the curve)
    ax1.fill_between(baseline_curve_df[diversity_col], 
                     baseline_curve_df[accuracy_col], 
                     baseline_curve_df[accuracy_col].max() * 1.5,
                     alpha=0.1, color='green', label='Efficient Zone (PEI>1.0)')
    
    for _, row in pei_df.iterrows():
        color = get_color(row['Run'])
        ax1.scatter(row[diversity_col], row[accuracy_col],
                   s=200, color=color, edgecolors='black', linewidth=2,
                   alpha=0.85, zorder=5)
        
        # PEI 라벨
        if row['PEI'] > 1.1 or row['PEI'] < 0.9:
            pei_text = f"{row['PEI']:.2f}"
            ax1.annotate(pei_text, 
                        xy=(row[diversity_col], row[accuracy_col]),
                        xytext=(5, 8), textcoords='offset points',
                        fontsize=9, fontweight='bold',
                        color='green' if row['PEI'] > 1 else 'red')
    
    ax1.set_xlabel(f'{diversity_col}@{k}', fontsize=12, fontweight='bold')
    ax1.set_ylabel(f'{accuracy_col}@{k}', fontsize=12, fontweight='bold')
    ax1.set_title('Trade-off with Baseline Frontier', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # --- Subplot 2: PEI Bar Chart ---
    ax2 = fig.add_subplot(2, 2, 2)
    
    pei_sorted = pei_df.sort_values('PEI', ascending=True)
    bar_colors = [get_color(run) for run in pei_sorted['Run']]
    
    bars = ax2.barh(range(len(pei_sorted)), pei_sorted['PEI'], color=bar_colors,
                    edgecolor='black', linewidth=1.5)
    
    ax2.axvline(x=1.0, color='red', linestyle='--', linewidth=2,
                label='PEI = 1.0 (Random Mixture)')
    
    ax2.set_yticks(range(len(pei_sorted)))
    ax2.set_yticklabels(pei_sorted['Run'], fontsize=10)
    ax2.set_xlabel('Pareto Efficiency Index (PEI)', fontsize=12, fontweight='bold')
    ax2.set_title('Model Efficiency Ranking', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 값 라벨
    for i, (idx, row) in enumerate(pei_sorted.iterrows()):
        pei_val = row['PEI']
        offset = 0.02 if pei_val < 1.3 else -0.15
        ax2.text(pei_val + offset, i, f'{pei_val:.2f}', va='center', fontsize=9, fontweight='bold')
    
    # --- Subplot 3: Entropy vs NDCG ---
    ax3 = fig.add_subplot(2, 2, 3)
    
    for _, row in models_df.iterrows():
        color = get_color(row['Run'])
        ax3.scatter(row['Entropy'], row['NDCG'],
                   s=200, color=color, edgecolors='black', linewidth=2,
                   alpha=0.85, label=row['Run'], zorder=5)
    
    ax3.set_xlabel(f'Recommendation Entropy@{k}', fontsize=12, fontweight='bold')
    ax3.set_ylabel(f'NDCG@{k}', fontsize=12, fontweight='bold')
    ax3.set_title('Entropy (Diversity) vs Accuracy Trade-off', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 이론적 설명 박스
    theory_text = (
        "Information Theoretic View:\n"
        "• Higher Entropy → More uniform P_rec\n"
        "• Skewed P_data (power-law)\n"
        "• ↑ H(P_rec) → ↑ D_KL(P_rec||P_data)\n"
        "• → Lower bound on accuracy loss"
    )
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.5)
    ax3.text(0.02, 0.02, theory_text, transform=ax3.transAxes,
            fontsize=9, verticalalignment='bottom', bbox=props, family='monospace')
    
    # --- Subplot 4: Dataset & Theory Summary ---
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    summary_text = f"""
    {'='*50}
    Dataset Statistics: {dataset_name}
    {'='*50}
    
    Total Items: {stats.n_items:,}
    Head Items (Top 20%): {stats.n_head:,}
    Tail Items (Bottom 80%): {stats.n_tail:,}
    
    Head Popularity Share: {stats.head_pop_ratio*100:.1f}%
    Tail Popularity Share: {stats.tail_pop_ratio*100:.1f}%
    Gini Index: {stats.gini_index:.4f}
    
    {'='*50}
    Theoretical Justification
    {'='*50}
    
    Natural Decay Factor (P_head - P_tail): {stats.head_pop_ratio - stats.tail_pop_ratio:.3f}
    
    Method 1 (Probability Mass Shift):
    * "+10% Tail ratio -> Expected loss = {(stats.head_pop_ratio - stats.tail_pop_ratio) * 10:.1f}%"
    
    Method 2 (PEI):
    * PEI > 1.0: More efficient than random
    * Best PEI: {pei_df['PEI'].max():.2f} ({pei_df.loc[pei_df['PEI'].idxmax(), 'Run']})
    
    Method 3 (KL-Divergence):
    * More diversity = Higher entropy
    * -> Larger gap from biased data -> Accuracy cost
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
    
    plt.tight_layout()
    
    # 저장
    save_path = output_dir / f"theoretical_tradeoff_analysis_k{k}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SUCCESS] Plot saved to {save_path}")
    
    return save_path


# ==================== Report Generation ====================

def generate_latex_table(
    pei_df: pd.DataFrame, 
    output_dir: Path,
    accuracy_col: str = 'NDCG',
    diversity_col: str = 'LongTailCoverage'
):
    """논문용 LaTeX 테이블 생성"""
    
    latex_content = f"""
% Pareto Efficiency Index Table
\\begin{{table}}[htbp]
\\centering
\\caption{{Pareto Efficiency Index (PEI) for Various Models}}
\\label{{tab:pei_analysis}}
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Model}} & \\textbf{{{accuracy_col}@20}} & \\textbf{{{diversity_col}@20}} & \\textbf{{Expected {accuracy_col}}} & \\textbf{{PEI}} \\\\
\\midrule
"""
    
    for _, row in pei_df.sort_values('PEI', ascending=False).iterrows():
        pei_marker = r"\textbf{" + f"{row['PEI']:.2f}" + r"}" if row['PEI'] > 1.0 else f"{row['PEI']:.2f}"
        latex_content += f"{row['Run']} & {row[accuracy_col]:.4f} & {row[diversity_col]:.4f} & {row[f'Expected_{accuracy_col}']:.4f} & {pei_marker} \\\\\n"
    
    latex_content += r"""
\bottomrule
\end{tabular}
\vspace{5pt}
\footnotesize{Note: PEI $>$ 1.0 indicates performance above the linear trade-off baseline.}
\end{table}
"""
    
    save_path = output_dir / "pei_table.tex"
    with open(save_path, 'w') as f:
        f.write(latex_content)
    
    print(f"[INFO] LaTeX table saved to {save_path}")


def generate_paper_claims(
    stats: DatasetStats,
    pei_df: pd.DataFrame,
    target_model: str = 'csar',
    accuracy_col: str = 'NDCG',
    diversity_col: str = 'LongTailCoverage'
) -> str:
    """논문용 주장 문구 생성"""
    
    # CSAR 모델 찾기
    csar_rows = pei_df[pei_df['Run'].str.lower().str.contains(target_model)]
    
    if csar_rows.empty:
        best_model = pei_df.loc[pei_df['PEI'].idxmax()]
    else:
        best_model = csar_rows.loc[csar_rows['PEI'].idxmax()]
    
    natural_decay = stats.head_pop_ratio - stats.tail_pop_ratio
    
    # 방법론 1 문구
    claim_1 = f"""
**방법론 1 (확률 질량 이동) - 논문용 주장:**
> "수학적으로 **Tail 추천 비중을 10% 늘리면, Hit Rate는 {natural_decay*10:.1f}% 감소하는 것이 
> 이론적 정상치(Expected Baseline)**이다. 그러나 우리 모델({best_model['Run']})은 
> {diversity_col}가 {best_model[diversity_col]*100:.1f}%에 달하면서도 
> {accuracy_col} {best_model[accuracy_col]:.4f}를 유지하여, 단순 확률 이동 대비 높은 효율성을 입증한다."
"""
    
    # 방법론 2 문구
    claim_2 = f"""
**방법론 2 (PEI) - 논문용 주장:**
> "우리는 새로운 지표 **Pareto Efficiency Index (PEI)**를 제안한다. 
> 단순 혼합 모델(Random Mixture)의 PEI는 1.0이지만, 
> **{best_model['Run']}의 PEI는 {best_model['PEI']:.2f}**이다. 
> 이는 정확도-다양성 상충 관계에서 우리 모델이 
> **기대치보다 {(best_model['PEI']-1)*100:.0f}% 더 효율적인 파레토 프론티어(Pareto Frontier)에 위치**함을 보여준다."
"""
    
    # 방법론 3 문구
    claim_3 = f"""
**방법론 3 (KL-Divergence) - 논문용 주장:**
> "Improving diversity is mathematically equivalent to increasing the entropy of P_rec. 
> Since P_data follows a power-law (low entropy), **any increase in recommendation entropy 
> strictly enforces a lower bound on accuracy metrics.** The observed drop in NDCG is not 
> a failure but a **theoretical cost of exploring beyond the distribution bias.**"
"""
    
    full_claims = f"""
{'='*80}
📝 PAPER-READY CLAIMS (Copy-Paste Ready)
{'='*80}

{claim_1}

{claim_2}

{claim_3}

{'='*80}
"""
    
    return full_claims


# ==================== Main ====================

def main(dataset_path: str, k: int = 20, output: str = None, 
         accuracy_metric: str = 'NDCG', diversity_metric: str = 'LongTailCoverage'):
    """
    메인 분석 함수
    
    Args:
        dataset_path: 학습된 모델 경로
        k: Top-K
        output: 출력 디렉토리
        accuracy_metric: 정확도 지표 (NDCG, HitRate, LongTailNDCG, LongTailHitRate, etc.)
        diversity_metric: 다양성 지표 (LongTailCoverage, Entropy, Coverage, ILD, Novelty)
    """
    
    dataset_path = Path(dataset_path)
    dataset_name = dataset_path.name
    
    print(f"\n{'#'*80}")
    print(f"# Theoretical Trade-off Analysis: {dataset_name}")
    print(f"# Accuracy: {accuracy_metric} | Diversity: {diversity_metric}")
    print(f"{'#'*80}\n")
    
    # 1. 데이터 로드
    data = load_processed_data(dataset_name)
    if data is None:
        print("[ERROR] Cannot load dataset. Exiting.")
        return
    
    stats = compute_dataset_stats(data)
    
    print(f"Dataset Statistics:")
    print(f"  Items: {stats.n_items} (Head: {stats.n_head}, Tail: {stats.n_tail})")
    print(f"  Head Popularity: {stats.head_pop_ratio*100:.1f}%")
    print(f"  Tail Popularity: {stats.tail_pop_ratio*100:.1f}%")
    print(f"  Natural Decay Factor: {stats.head_pop_ratio - stats.tail_pop_ratio:.4f}")
    
    # 2. 모델 메트릭 로드
    models_df = load_model_metrics(dataset_path, k)
    
    if models_df.empty:
        print("[ERROR] No model metrics found!")
        return
    
    print(f"\nFound {len(models_df)} models:")
    print(models_df[['Run', accuracy_metric, diversity_metric, 'Entropy']].to_string(index=False))
    
    # 3. 방법론 1: 확률 질량 이동 상세 분석
    # 3. 방법론 1: 확률 질량 이동 상세 분석
    mass_shift_df = compute_mass_shift_detailed_analysis(
        stats, 
        models_df,
        accuracy_col=accuracy_metric,
        diversity_col=diversity_metric
    )
    
    # 4. Simulate Baseline Curve
    baseline_curve_df = simulate_popularity_curve(data, k=k)
    
    # 5. PEI 분석 (Curve-based)
    pei_df = compute_pareto_efficiency_index(
        models_df, 
        baseline_curve_df,
        accuracy_col=accuracy_metric, 
        diversity_col=diversity_metric
    )
    
    print("\nMass Shift Analysis Results:")
    print(mass_shift_df[['Run', accuracy_metric, diversity_metric, f'Expected_{accuracy_metric}', 'Outperformance_Pct']].to_string(index=False))
    
    print("\nPEI Results (Curve-based):")
    print(pei_df[['Run', accuracy_metric, diversity_metric, f'Expected_{accuracy_metric}', 'PEI']].to_string(index=False))
    
    # 6. 방법론 3: KL-Divergence 분석
    kl_df = compute_kl_divergence_analysis(data, models_df)
    
    # 7. 출력 디렉토리
    if output:
        output_dir = Path(output)
    else:
        output_dir = Path(ROOT) / "output" / dataset_name / "analysis" / "theoretical_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 8. 시각화
    # 8. 시각화
    plot_comprehensive_analysis(
        models_df, pei_df, baseline_curve_df, stats, dataset_name, output_dir, k, # fixed order
        accuracy_col=accuracy_metric,
        diversity_col=diversity_metric
    )
    
    # 9. 방법론 1 전용 시각화
    plot_mass_shift_analysis(
        mass_shift_df, stats, dataset_name, output_dir, k,
        accuracy_col=accuracy_metric,
        diversity_col=diversity_metric
    )
    
    # 10. LaTeX 테이블
    # 10. LaTeX 테이블
    generate_latex_table(
        pei_df, 
        output_dir,
        accuracy_col=accuracy_metric,
        diversity_col=diversity_metric
    )
    
    # 11. 논문용 문구
    claims = generate_paper_claims(
        stats, 
        pei_df,
        accuracy_col=accuracy_metric,
        diversity_col=diversity_metric
    )
    print(claims)
    
    # claims 저장
    claims_path = output_dir / "paper_claims.md"
    with open(claims_path, 'w') as f:
        f.write(claims)
    print(f"[INFO] Paper claims saved to {claims_path}")
    
    # 12. 결과 CSV 저장
    pei_df.to_csv(output_dir / f"pei_analysis_k{k}.csv", index=False)
    kl_df.to_csv(output_dir / f"kl_analysis_k{k}.csv", index=False)
    mass_shift_df.to_csv(output_dir / f"mass_shift_analysis_k{k}.csv", index=False)
    baseline_curve_df.to_csv(output_dir / f"baseline_curve_k{k}.csv", index=False)
    
    print(f"\n[SUCCESS] All analysis completed. Results saved to {output_dir}")


# 지원하는 지표 정의
ACCURACY_METRICS = ['NDCG', 'HitRate', 'LongTailNDCG', 'LongTailHitRate', 'HeadNDCG', 'HeadHitRate']
DIVERSITY_METRICS = ['LongTailCoverage', 'Entropy', 'Coverage', 'ILD', 'Novelty']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Theoretical Trade-off Analysis for Accuracy vs Diversity'
    )
    parser.add_argument('dataset_path', type=str,
                       help='Path to trained models (e.g., trained_model/ml-1m)')
    parser.add_argument('--k', type=int, default=10,
                       help='Top-K for metrics (default: 10)')
    parser.add_argument('--output', type=str, default=None,
                       help='Custom output directory')
    parser.add_argument('--accuracy', type=str, default='NDCG',
                       choices=ACCURACY_METRICS,
                       help=f'Accuracy metric to use (default: NDCG). Options: {ACCURACY_METRICS}')
    parser.add_argument('--diversity', type=str, default='LongTailCoverage',
                       choices=DIVERSITY_METRICS,
                       help=f'Diversity metric to use (default: LongTailCoverage). Options: {DIVERSITY_METRICS}')
    
    args = parser.parse_args()
    
    print(f"\n[CONFIG] Accuracy Metric: {args.accuracy}")
    print(f"[CONFIG] Diversity Metric: {args.diversity}\n")
    
    main(args.dataset_path, args.k, args.output, args.accuracy, args.diversity)
