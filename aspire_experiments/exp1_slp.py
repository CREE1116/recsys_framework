import os
import sys
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aspire_experiments.exp_utils import get_loader_and_svd, ensure_dir

def compute_v_p_v_metrics(V: torch.Tensor, P_diag: torch.Tensor, S: torch.Tensor = None):
    """
    V: (I, k) tensor of item singular vectors
    P_diag: (I,) tensor of item popularity
    """
    # V^T P V = V^T @ diag(P) @ V
    # efficiently compute: V^T @ (P_diag.unsqueeze(1) * V)
    V_T_P = V.T @ (P_diag.unsqueeze(1) * V)  # (k, k)
    k = V_T_P.shape[0]
    
    # Extract diagonal
    diag_elements = torch.diag(V_T_P)
    diag_energy = torch.sum(diag_elements ** 2)
    
    # Extract off-diagonal
    V_T_P_off_diag = V_T_P - torch.diag(diag_elements)
    off_diag_energy = torch.sum(V_T_P_off_diag ** 2)
    
    # 1. 대각성분이 전체의 몇 %인지 (에너지 기준)
    total_energy = diag_energy + off_diag_energy
    diag_energy_percent = (diag_energy / (total_energy + 1e-12)).item() * 100.0
    
    # 2. 비대각성분의 평균(절댓값)에 비해 대각성분의 평균(절댓값)이 몇 배인지
    mean_abs_diag = torch.mean(torch.abs(diag_elements))
    
    if k > 1:
        sum_abs_off_diag = torch.sum(torch.abs(V_T_P_off_diag))
        mean_abs_off_diag = sum_abs_off_diag / (k * k - k)
        diag_to_offdiag_mean_ratio = (mean_abs_diag / (mean_abs_off_diag + 1e-12)).item()
    else:
        diag_to_offdiag_mean_ratio = 0.0
    
    # 3. Log-log linear fit for beta estimation
    beta_est, r2 = None, None
    if S is not None:
        # S is eigenvalues lambda_k. In ASPIRE context, user snippet uses S directly as 'sigma'.
        # We clamp to 0 and take sqrt if they are strictly eigenvalues, but treating them as 'sigma'
        # matching user's snippet.
        sigma = S[:k].cpu().numpy()
        p_k = diag_elements.cpu().numpy()
        
        log_sigma = np.log(sigma + 1e-8)
        log_p = np.log(p_k + 1e-8)
        
        # linear fit
        coef = np.polyfit(log_sigma, log_p, 1)
        beta_est = float(coef[0])
        
        # R^2
        pred = np.polyval(coef, log_sigma)
        ss_res = np.sum((log_p - pred)**2)
        ss_tot = np.sum((log_p - log_p.mean())**2)
        r2 = float(1 - ss_res / (ss_tot + 1e-8))
    
    metrics = {
        "diag_energy_percent": diag_energy_percent,
        "diag_to_offdiag_mean_ratio": diag_to_offdiag_mean_ratio,
        "beta_est": beta_est,
        "r2": r2
    }

    
    return metrics, V_T_P

def run_exp1(datasets):
    output_dir = ensure_dir("aspire_experiments/output/exp1")
    results = {}
    
    # k를 통제하지 않고 Full SVD/EVD 사용을 위해 (캐시가 있으면 로드)k=None으로 실행합니다.
    k_eval = None

    
    for ds in datasets:
        ds_output_dir = ensure_dir(os.path.join(output_dir, ds))
        try:
            print(f"========== Processing {ds} ==========")    
            loader, R, S, V, config = get_loader_and_svd(ds, k=k_eval)
            
            # Popularity matrix P (diagonal)
            # R is a csr_matrix (user x item). sum(axis=0) gives item popularity.
            popularity = np.array(R.sum(axis=0)).flatten()
            P_diag = torch.tensor(popularity, dtype=torch.float32, device=V.device)
            
            # V size is (I, k)
            if V.shape[0] != P_diag.shape[0]:
                print(f"Warning: V shape {V.shape} doesn't match P_diag shape {P_diag.shape}. Using matching subset.")
                min_len = min(V.shape[0], P_diag.shape[0])
                V = V[:min_len]
                P_diag = P_diag[:min_len]
            
            metrics, V_T_P = compute_v_p_v_metrics(V, P_diag, S)
            print(f"Dataset: {ds}")
            print(f"  - 대각성분 에너지 비율: {metrics['diag_energy_percent']:.2f}%")
            print(f"  - 대각성분 평균 vs 비대각성분 평균: {metrics['diag_to_offdiag_mean_ratio']:.2f}배")
            if metrics["beta_est"] is not None:
                print(f"  - Beta 추정치: {metrics['beta_est']:.4f} (R^2: {metrics['r2']:.4f})")
            
            results[ds] = {
                "diag_energy_percent": metrics["diag_energy_percent"],
                "diag_to_offdiag_mean_ratio": metrics["diag_to_offdiag_mean_ratio"],
                "beta_est": metrics["beta_est"],
                "r2": metrics["r2"],
                "n_items": int(popularity.shape[0]),
                "k_used": int(V.shape[1])
            }

            
            # Save dataset specific results
            with open(os.path.join(ds_output_dir, "results.json"), "w") as f:
                json.dump(results[ds], f, indent=4)
            
            # Plot full Heatmap
            V_T_P_mat = V_T_P.cpu().numpy()
            
            plt.figure(figsize=(10, 8))
            # V^T P V는 음수도 나올 수 있으므로, 크기를 명확히 보기 위해 절댓값의 로그 스케일을 사용
            sns.heatmap(np.log1p(np.abs(V_T_P_mat)), cmap="viridis", cbar_kws={'label': 'log(1 + |value|)'})
            title_text = (
                f"{ds}: V^T P V (Full {V_T_P.shape[0]}x{V_T_P.shape[1]})\n"
                f"Diag E.: {metrics['diag_energy_percent']:.1f}% | "
                f"Diag Mean / Off-Diag Mean: {metrics['diag_to_offdiag_mean_ratio']:.1f}x\n"
                f"Beta_est: {metrics['beta_est']:.3f} (R2={metrics['r2']:.3f})" if metrics["beta_est"] is not None else ""
            )

            plt.title(title_text)
            plt.xlabel("Item Eigenvector Component Index (k)")
            plt.ylabel("Item Eigenvector Component Index (k)")
            plt.tight_layout()
            plt.savefig(os.path.join(ds_output_dir, "heatmap.png"), dpi=150)
            plt.close()
            
        except Exception as e:
            print(f"Skipping {ds} due to error: {e}")
            
    # 전체 요약 결과를 root output 폴더에 저장
    with open(os.path.join(output_dir, "summary_results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nExperiment 1 completed. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASPIRE Exp1: V^T P V Diagonalization Measurement")
    parser.add_argument("--dataset", nargs='+', default=["ml-100k"],
                        help="평가할 데이터셋 이름 (예: ml-100k, yaml 빼고 입력). 쉼표로 구분하여 여러 개 동시 입력 가능. 'all' 입력 시 전체 실행.")
    args = parser.parse_args()
    
    # 리스트로 받아진 인자들을 하나의 문자열로 합친 뒤 쉼표로 분리 (띄어쓰기 대응)
    dataset_str = "".join(args.dataset)
    if dataset_str == 'all':
        datasets_to_run = ['ml-100k', 'ml-1m', 'yahoo_r3', 'gowalla', 'yelp2018', 'amazon-book']
    else:
        datasets_to_run = [d.strip() for d in dataset_str.split(',') if d.strip()]
        
    run_exp1(datasets_to_run)
