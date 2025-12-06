import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

# 프로젝트 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.utils import load_model_from_run, get_analysis_output_path

def analyze_nonlinear_transition(exp_config):
    """
    [비선형 동역학 전후 비교]
    원본 임베딩과 최종 임베딩의 인기도 대비 Norm 분포 형태를 비교합니다.
    - 목표: Original(선형 편향) -> Final(이상적인 Flat 혹은 전략적인 U자형)으로의 변화 증명
    """
    run_folder_path = exp_config['run_folder_path']
    print(f"\nRunning Non-linear Transition Analysis for: {run_folder_path}")

    # 1. 모델 로드
    model, data_loader, config = load_model_from_run(run_folder_path)
    if not model: return

    # 2. 데이터 준비
    item_counts = data_loader.df['item_id'].value_counts()
    
    # (A) 원본 임베딩 (Original)
    orig_embs = model.item_embedding.weight.detach().cpu()
    orig_norms = torch.norm(orig_embs, p=2, dim=1).numpy()
    
    # (B) 최종 임베딩 (Final)
    if hasattr(model, 'get_final_item_embeddings'):
        final_embs = model.get_final_item_embeddings().detach().cpu()
    else:
        final_embs = model.attention_layer(model.item_embedding.weight).detach().cpu()
    final_norms = torch.norm(final_embs, p=2, dim=1).numpy()

    # 데이터 정렬
    pops = []
    y_orig = []
    y_final = []
    
    for item_id, count in item_counts.items():
        if item_id < len(orig_norms):
            pops.append(np.log1p(count))
            y_orig.append(orig_norms[item_id])
            y_final.append(final_norms[item_id])

    X = np.array(pops)
    Y_orig = np.array(y_orig)
    Y_final = np.array(y_final)

    # ---------------------------------------------------------
    # Visualization Setup
    # ---------------------------------------------------------
    output_path = get_analysis_output_path(config['dataset_name'], os.path.basename(run_folder_path))
    save_file = os.path.join(output_path, "nonlinear_transition_smile_curve.png")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # 공통 분석 함수 (Plotting & Fitting)
    def analyze_and_plot(ax, x_data, y_data, title_prefix, color_code):
        # Scatter
        ax.scatter(x_data, y_data, alpha=0.1, s=5, color='gray', label='Items')
        
        # 1. Linear Fit (직선)
        z1 = np.polyfit(x_data, y_data, 1)
        p1 = np.poly1d(z1)
        r2_lin = r2_score(y_data, p1(x_data))
        ax.plot(np.sort(x_data), p1(np.sort(x_data)), "r--", linewidth=2, 
                label=f'Linear (R²={r2_lin:.3f})')
        
        # 2. Poly Fit (곡선)
        z2 = np.polyfit(x_data, y_data, 2)
        p2 = np.poly1d(z2)
        r2_poly = r2_score(y_data, p2(x_data))
        curvature = z2[0] # a in ax^2 + bx + c
        curve_type = "U-Shape" if curvature > 0 else "Inverted-U"
        
        ax.plot(np.sort(x_data), p2(np.sort(x_data)), color=color_code, linewidth=3, 
                label=f'Quadratic (R²={r2_poly:.3f})')
        
        # 3. Segmented Correlation (Tail vs Head)
        df = pd.DataFrame({'x': x_data, 'y': y_data}).sort_values('x')
        cut_low = int(len(df)*0.2)
        cut_high = int(len(df)*0.8)
        
        r_tail, _ = pearsonr(df.iloc[:cut_low]['x'], df.iloc[:cut_low]['y'])
        r_head, _ = pearsonr(df.iloc[cut_high:]['x'], df.iloc[cut_high:]['y'])
        
        # 텍스트 표시
        stats_text = (f"Linear Slope: {z1[0]:.3f}\n"
                      f"Curvature (a): {curvature:.3f} ({curve_type})\n"
                      f"Tail Corr: {r_tail:.3f}\n"
                      f"Head Corr: {r_head:.3f}")
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 이상적인 Flat Line (참고용)
        ax.axhline(y=y_data.mean(), color='black', linestyle=':', alpha=0.5, label='Ideal Flat (Reference)')

        ax.set_title(f"{title_prefix}\n(Linear vs Quadratic Fit)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Log Popularity", fontsize=12)
        ax.set_ylabel("Embedding Norm", fontsize=12)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        return curvature, r_tail, r_head

    # --- 실행 ---
    
    # Plot A: Original
    curv_orig, rt_orig, rh_orig = analyze_and_plot(axes[0], X, Y_orig, "(A) Original Embeddings", "orange")
    
    # Plot B: Final
    curv_final, rt_final, rh_final = analyze_and_plot(axes[1], X, Y_final, "(B) Final Interest Embeddings", "blue")

    # 전체 타이틀
    plt.suptitle(f"Structural Transformation: From Linear Bias to Non-linear Modulation", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    print(f"Saved analysis plot to: {save_file}")
    
    # 콘솔 요약 출력
    print("\n[Dynamics Summary]")
    print(f"Original: Curvature={curv_orig:.4f}, Tail_Corr={rt_orig:.3f}, Head_Corr={rh_orig:.3f}")
    print(f"Final   : Curvature={curv_final:.4f}, Tail_Corr={rt_final:.3f}, Head_Corr={rh_final:.3f}")
    
    if abs(curv_final) > abs(curv_orig):
        print("-> Result: Non-linearity INCREASED (Smile Curve formed).")
    else:
        print("-> Result: Non-linearity DECREASED (Flattened).")

if __name__ == '__main__':
    EXPERIMENTS = [
        {'run_folder_path': '/Users/leejongmin/code/recsys_framework/trained_model/ml-1m/csar-sampled'}
    ]
    for exp in EXPERIMENTS:
        if os.path.exists(exp['run_folder_path']):
            analyze_nonlinear_transition(exp)
        else:
            print(f"[Error] Path not found: {exp['run_folder_path']}")