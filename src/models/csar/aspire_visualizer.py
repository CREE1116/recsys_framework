import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

class ASPIREVisualizer:
    @staticmethod
    def visualize_aspire_spectral(singular_values, filter_diag, alpha, gamma, effective_alpha=None, X_sparse=None, save_dir=None, file_prefix='aspire'):
        """
        ASPIRE 전용 스펙트럼 분석 시각화 (Gamma 기반).
        
        Args:
            singular_values (Tensor): SVD 고윳값
            filter_diag (Tensor): 적용된 필터 계수 h(s)
            alpha (float): 하이퍼파라미터 alpha
            gamma (float): 하이퍼파라미터 gamma
            effective_alpha (float): 스케일링이 적용된 실제 알파 (alpha / mean(s^gamma))
            X_sparse (sparse): 원본 인터랙션 행렬 (에너지 계산용)
            save_dir (str): 저장 디렉토리
            file_prefix (str): 파일명 접두사
        """
        if not save_dir:
            return
            
        os.makedirs(save_dir, exist_ok=True)
        s_vals = singular_values.detach().cpu().numpy()
        filter_w = filter_diag.detach().cpu().numpy()
        s2 = s_vals**2
        
        # 1. Metrics Calculation
        metrics = {
            "config": {
                "alpha": float(alpha), 
                "gamma": float(gamma),
                "effective_alpha": float(effective_alpha) if effective_alpha is not None else None
            },
            "spectral": {
                "s_mean": float(s_vals.mean()),
                "s_max": float(s_vals.max()),
                "s_min": float(s_vals.min())
            },
            "filter": {
                "h_mean": float(filter_w.mean()),
                "h_max": float(filter_w.max()),
                "h_min": float(filter_w.min())
            }
        }
        
        if X_sparse is not None:
            total_energy = X_sparse.power(2).sum()
            captured_energy = np.sum(s2)
            metrics['energy_ratio'] = float(captured_energy / (total_energy + 1e-9))
            
        with open(os.path.join(save_dir, f'{file_prefix}_metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)

        # 2. Plotting (3-Panel Analysis)
        plt.style.use('seaborn-v0_8-muted') if 'seaborn-v0_8-muted' in plt.style.available else plt.style.use('ggplot')
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # [Panel 1] Spectral Power Law (Log-Log)
        ax = axes[0]
        ax.loglog(s_vals, label=r'Original $\sigma_k$', alpha=0.8, color='#3498db')
        ax.set_title("Spectral Power-law Analysis", fontsize=14, fontweight='bold')
        ax.set_xlabel("Rank (k)", fontsize=12)
        ax.set_ylabel("Singular Value", fontsize=12)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend()

        # [Panel 2] Filter Transfer Function
        ax = axes[1]
        # Wiener baseline: s^2 / (s^2 + alpha)
        wiener = (s_vals**2) / (s_vals**2 + float(alpha) + 1e-9)
        
        ax.plot(s_vals, wiener, color='#95a5a6', linestyle='--', label=r'Wiener ($\gamma=2$)', alpha=0.6)
        ax.plot(s_vals, filter_w, color='#e67e22', linewidth=2, label=rf'ASPIRE ($\gamma={gamma:.2f}$)')
        
        ax.set_xscale('log')
        ax.invert_xaxis()
        
        title_alpha = rf"$\alpha={alpha:.1f}$"
        if effective_alpha is not None:
            title_alpha += rf", $\alpha_{{eff}}={effective_alpha:.4f}$"
            
        ax.set_title(f"Filter Shape ({title_alpha})", fontsize=14, fontweight='bold')
        ax.set_xlabel(r"Singular Value $\sigma$", fontsize=12)
        ax.set_ylabel(r"$h(\sigma)$", fontsize=12)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend()

        # [Panel 3] Spectral Restoration (Effective Sigma)
        ax = axes[2]
        eff_s = filter_w * s_vals
        ax.plot(eff_s, color='#2ecc71', label=r'Filtered $\sigma_{eff}$')
        ax.set_yscale('log')
        ax.set_title("Spectral Restoration (Effective)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Rank (k)", fontsize=12)
        ax.set_ylabel(r"$h(\sigma_k) \cdot \sigma_k$", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{file_prefix}_analysis.png'), dpi=150)
        plt.close()
        
        print(f"[ASPIREVisualizer] Visualizations saved to {save_dir} (Eff Alpha: {effective_alpha})")
