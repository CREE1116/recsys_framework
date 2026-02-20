"""
Oracle-based NDCG vs LongTail Coverage Trade-off 분석
실제 테스트셋으로 Popularity Oracle의 이론적 상한을 계산하고 모델 효율성 평가
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import yaml
import pickle
from scipy.interpolate import interp1d

# ==================== Data Loading ====================

def load_dataset_stats(dataset_name):
    """Dataset 통계 로드"""
    path = Path(ROOT) / "output" / dataset_name / "dataset_analysis" / "dataset_stats.json"
    if not path.exists():
        print(f"[Warning] Stats not found. Using ML-1M defaults.")
        return {
            'gini_index': 0.6622,
            'head_ratio_20': 0.6919,
            'sparsity': 0.9695
        }
    with open(path, 'r') as f:
        return json.load(f)

def find_model_metrics(dataset_path, k=20):
    """모든 모델의 메트릭 수집"""
    results = []
    dataset_path = Path(dataset_path)
    
    for model_dir in dataset_path.iterdir():
        if not model_dir.is_dir():
            continue
        
        metrics_path = model_dir / "final_metrics.json"
        config_path = model_dir / "config.yaml"
        
        if not (metrics_path.exists() and config_path.exists()):
            continue
            
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            model_name = config['model']['name']
            run_name = model_dir.name
            
            ndcg = metrics.get(f"NDCG@{k}")
            lt_cov = metrics.get(f"LongTailCoverage@{k}")
            
            if ndcg is not None and lt_cov is not None:
                results.append({
                    'Model': model_name,
                    'Run': run_name,
                    'NDCG': ndcg,
                    'LongTail_Coverage': lt_cov,
                })
        except Exception as e:
            print(f"[Error] Loading {model_dir.name}: {e}")
    
    return pd.DataFrame(results)

def load_processed_data(dataset_name):
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

# ==================== Oracle Simulation ====================

def compute_oracle_curve(data, k=20, num_points=20):
    """
    Popularity-based Oracle 시뮬레이션
    
    전략:
    1. Head/Tail을 인기도순으로 정렬
    2. Top-K 슬롯 중 N개를 tail에 강제 할당
    3. 전체 유저에 대해 동일한 추천 리스트로 NDCG 계산
    4. Tail coverage = 추천에 포함된 tail 아이템 수 / 전체 tail 수
    
    Returns:
        DataFrame with columns: tail_ratio, longtail_coverage, ndcg, std
    """
    print(f"\n{'='*70}")
    print(f"Oracle Simulation (K={k})")
    print(f"{'='*70}")
    
    test_df = data['test_df']
    item_pop = data['item_popularity']  # Series
    
    # Head/Tail split (20/80)
    sorted_items = item_pop.sort_values(ascending=False)
    n_items = len(sorted_items)
    head_cutoff = int(n_items * 0.2)
    
    head_items = sorted_items.index[:head_cutoff].tolist()
    tail_items = sorted_items.index[head_cutoff:].tolist()
    
    print(f"  Total items: {n_items}")
    print(f"  Head items (top 20%): {len(head_items)}")
    print(f"  Tail items (bottom 80%): {len(tail_items)}")
    print(f"  Head popularity share: {item_pop[head_items].sum() / item_pop.sum():.2%}")
    
    # User ground truth
    user_gt = test_df.groupby('user_id')['item_id'].apply(set).to_dict()
    all_users = list(user_gt.keys())
    
    results = []
    
    # Tail 슬롯 비율: 0% ~ 50% (10% 이상은 비현실적이지만 커브 확인용)
    tail_ratios = np.linspace(0, 0.5, num_points)
    
    for tail_ratio in tail_ratios:
        n_tail = int(k * tail_ratio)
        n_head = k - n_tail
        
        # 글로벌 추천 리스트 (popularity 기준)
        rec_list = head_items[:n_head] + tail_items[:n_tail]
        
        # Tail coverage 계산
        tail_in_recs = set(rec_list) & set(tail_items)
        longtail_coverage = len(tail_in_recs) / len(tail_items)
        
        # 각 유저의 NDCG
        ndcg_scores = []
        
        for user in all_users:
            gt = user_gt.get(user, set())
            if not gt:
                continue
            
            # DCG 계산
            dcg = 0.0
            for rank, item in enumerate(rec_list):
                if item in gt:
                    dcg += 1.0 / np.log2(rank + 2)
            
            # IDCG (perfect ranking)
            num_relevant = min(len(gt), k)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))
            
            if idcg > 0:
                ndcg_scores.append(dcg / idcg)
        
        avg_ndcg = np.mean(ndcg_scores)
        std_ndcg = np.std(ndcg_scores)
        
        results.append({
            'tail_ratio': tail_ratio,
            'longtail_coverage': longtail_coverage,
            'ndcg': avg_ndcg,
            'std': std_ndcg
        })
        
        if tail_ratio in [0.0, 0.1, 0.2, 0.3, 0.5]:
            print(f"  Tail={tail_ratio*100:4.0f}% → "
                  f"Coverage={longtail_coverage:.4f}, "
                  f"NDCG={avg_ndcg:.4f} ± {std_ndcg:.4f}")
    
    oracle_df = pd.DataFrame(results)
    
    print(f"\nOracle NDCG range: {oracle_df['ndcg'].min():.4f} ~ {oracle_df['ndcg'].max():.4f}")
    print(f"Coverage range: {oracle_df['longtail_coverage'].min():.4f} ~ {oracle_df['longtail_coverage'].max():.4f}")
    print(f"{'='*70}\n")
    
    return oracle_df

# ==================== Visualization ====================

def get_model_color(model_name):
    """모델별 색상"""
    colors = {
        'csar-rec2': '#E74C3C',
        'csar-rec': '#FF7F50',
        'lightgcn': '#2ECC71',
        'neumf': '#3498DB',
        'mf': '#95A5A6',
        'bpr': '#9B59B6',
        'ncf': '#F39C12',
    }
    for key, color in colors.items():
        if key in model_name.lower():
            return color
    return '#34495E'

def draw_tradeoff_plot(dataset_path, k=20, output_dir=None):
    """메인 플롯팅"""
    dataset_path = Path(dataset_path)
    dataset_name = dataset_path.name
    
    # 1. Load model metrics
    stats = load_dataset_stats(dataset_name)
    models_df = find_model_metrics(dataset_path, k)
    
    if models_df.empty:
        print("[ERROR] No model metrics found!")
        return
    
    print(f"\n[INFO] Found {len(models_df)} models:")
    print(models_df[['Model', 'NDCG', 'LongTail_Coverage']].to_string(index=False))
    
    # 2. Compute oracle curve
    processed_data = load_processed_data(dataset_name)
    
    if processed_data is None:
        print("[ERROR] Cannot proceed without test data!")
        return
    
    oracle_df = compute_oracle_curve(processed_data, k=k)
    
    # 3. Interpolation for efficiency calculation
    oracle_df_sorted = oracle_df.sort_values('longtail_coverage')
    
    oracle_interp = interp1d(
        oracle_df_sorted['longtail_coverage'],
        oracle_df_sorted['ndcg'],
        kind='linear',
        bounds_error=False,
        fill_value=(oracle_df_sorted['ndcg'].iloc[0], 
                   oracle_df_sorted['ndcg'].iloc[-1])
    )
    
    # 4. Plotting
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Oracle curve
    x_plot = np.linspace(
        0,
        max(oracle_df['longtail_coverage'].max() * 1.1,
            models_df['LongTail_Coverage'].max() * 1.2),
        300
    )
    y_plot = oracle_interp(x_plot)
    
    ax.plot(x_plot, y_plot, 'b--', linewidth=3, alpha=0.7,
            label='Popularity Oracle (Upper Bound)', zorder=2)
    
    # Confidence band (±1 std)
    std_avg = oracle_df['std'].mean()
    ax.fill_between(x_plot, 
                    np.maximum(y_plot - std_avg, 0),
                    y_plot + std_avg,
                    alpha=0.12, color='blue',
                    label=f'Oracle ±{std_avg:.4f} (avg std)', zorder=1)
    
    # Oracle sample points
    ax.scatter(oracle_df['longtail_coverage'], oracle_df['ndcg'],
              s=60, color='blue', alpha=0.4, marker='x',
              linewidths=2, label='Oracle Samples', zorder=3)
    
    # 5. Plot models
    for model_name in models_df['Model'].unique():
        model_subset = models_df[models_df['Model'] == model_name]
        color = get_model_color(model_name)
        
        ax.scatter(model_subset['LongTail_Coverage'], 
                  model_subset['NDCG'],
                  s=280, color=color, label=model_name,
                  alpha=0.85, edgecolors='black', linewidth=2,
                  zorder=6)
    
    # 6. Annotate efficiency
    print(f"\n{'='*80}")
    print(f"Model Efficiency vs Oracle (K={k})")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'LT_Cov':>10} {'NDCG':>8} {'Oracle':>8} {'Ratio':>8} {'Status'}")
    print(f"{'-'*80}")
    
    for _, row in models_df.iterrows():
        lt_cov = row['LongTail_Coverage']
        actual_ndcg = row['NDCG']
        
        # Oracle NDCG at this coverage
        oracle_ndcg = float(oracle_interp(lt_cov))
        oracle_ndcg = max(oracle_ndcg, 0.001)
        
        # Efficiency ratio (actual / oracle)
        efficiency_ratio = (actual_ndcg / oracle_ndcg) * 100
        
        # Color coding
        if efficiency_ratio >= 90:
            eff_color = 'green'
            status = '✓ Excellent'
        elif efficiency_ratio >= 75:
            eff_color = 'orange'
            status = '○ Good'
        else:
            eff_color = 'red'
            status = '✗ Below'
        
        # Annotation on plot
        ax.annotate(f"{efficiency_ratio:.0f}%", 
                   xy=(lt_cov, actual_ndcg),
                   xytext=(0, 14), textcoords='offset points',
                   ha='center', fontsize=11, weight='bold',
                   color=eff_color,
                   bbox=dict(boxstyle="round,pad=0.5", fc="white",
                            ec=eff_color, lw=2.5, alpha=0.95),
                   zorder=10)
        
        # Vertical line to oracle
        ax.plot([lt_cov, lt_cov], [oracle_ndcg, actual_ndcg],
               color=get_model_color(row['Model']), 
               linestyle=':', linewidth=2, alpha=0.6, zorder=4)
        
        # Print summary
        print(f"{row['Model']:<20} {lt_cov:>10.4f} {actual_ndcg:>8.4f} "
              f"{oracle_ndcg:>8.4f} {efficiency_ratio:>7.1f}% {status}")
    
    print(f"{'='*80}\n")
    
    # 7. Styling
    ax.set_xlabel(f'Long-Tail Coverage@{k}', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'NDCG@{k}', fontsize=14, fontweight='bold')
    
    title = (f"{dataset_name}: Oracle-based Trade-off Analysis\n"
            f"Gini={stats.get('gini_index', 0):.3f}, "
            f"Head-20%={stats.get('head_ratio_20', 0)*100:.1f}%, "
            f"Sparsity={stats.get('sparsity', 0)*100:.1f}%")
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95,
             edgecolor='black', fancybox=True, ncol=1)
    
    # Dataset info box
    info_text = '\n'.join([
        'Oracle Strategy:',
        '  • Popularity ranking',
        '  • Head/Tail quota',
        '  • Same recs for all users',
        '',
        'Efficiency Ratio:',
        '  • 90%+ : Excellent',
        '  • 75-90% : Good',
        '  • <75% : Below expectation'
    ])
    
    props = dict(boxstyle='round', facecolor='lightyellow', 
                alpha=0.3, edgecolor='gray', linewidth=1.5)
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', bbox=props,
           family='monospace')
    
    # 8. Save
    if output_dir is None:
        output_dir = Path("output") / dataset_name / "analysis" / "tradeoff_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = output_dir / f"oracle_tradeoff_k{k}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SUCCESS] Plot saved to {save_path}\n")
    
    # 9. Save oracle data for reference
    oracle_path = output_dir / f"oracle_curve_k{k}.csv"
    oracle_df.to_csv(oracle_path, index=False)
    print(f"[INFO] Oracle curve saved to {oracle_path}\n")

# ==================== Main ====================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Oracle-based NDCG vs LongTail trade-off analysis'
    )
    parser.add_argument('dataset_path', type=str,
                       help='Path to trained models (e.g., trained_model/ml-1m)')
    parser.add_argument('--k', type=int, default=20,
                       help='Top-K for metrics (default: 20)')
    parser.add_argument('--output', type=str, default=None,
                       help='Custom output directory')
    
    args = parser.parse_args()
    
    draw_tradeoff_plot(args.dataset_path, args.k, args.output)