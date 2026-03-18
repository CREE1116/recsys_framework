import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

class ASPIREVisualizer:
    @staticmethod
    def visualize_aspire_spectral(singular_values, filter_diag, alpha, beta, X_sparse=None, save_dir=None, file_prefix='aspire'):
        """
        ASPIRE 전용 스펙트럼 시각화.
        LIRA에서 분리되어 ASPIRE의 필터 특성(Spectral Scaling)을 집중적으로 보여줌.
        """
        if not save_dir:
            return
            
        os.makedirs(save_dir, exist_ok=True)
        metrics = {}
        s_vals = singular_values.detach().cpu().numpy()
        s2 = s_vals**2
        
        # 기준 위너 필터 (alpha 기반)
        wiener_filter = s2 / (s2 + float(alpha) + 1e-9)
        
        cum_energy = np.cumsum(s2)
        filter_w = filter_diag.detach().cpu().numpy()

        # Handle vectorized beta
        if isinstance(beta, (np.ndarray, list)):
            beta_val = float(np.mean(beta))
        elif hasattr(beta, "item"):
            beta_val = float(beta.item() if beta.numel() == 1 else beta.mean().item())
        else:
            beta_val = float(beta)

        metrics['SingularValues'] = {"mean": float(s_vals.mean()), "std": float(s_vals.std())}
        metrics['Filter_ASPIRE'] = {
            "mean": float(filter_w.mean()),
            "std": float(filter_w.std()),
            "max": float(filter_w.max()),
            "min": float(filter_w.min()),
            "beta_est": beta_val
        }
        
        if X_sparse is not None:
            total_energy = X_sparse.power(2).sum()
            metrics['SVD_Energy'] = {
                "captured_energy": float(cum_energy[-1]),
                "total_energy": float(total_energy),
                "ratio": float(cum_energy[-1] / (total_energy + 1e-9))
            }
            cum_energy_ratio = cum_energy / (total_energy + 1e-9)
        else:
            cum_energy_ratio = cum_energy / (cum_energy[-1] + 1e-9)
            
        with open(os.path.join(save_dir, f'{file_prefix}_metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)

        plt.figure(figsize=(18, 5))
        
        # 1. Spectrum (Log Scale)
        plt.subplot(1, 3, 1)
        plt.plot(s_vals, label=r'$\sigma_k$')
        plt.yscale('log')
        plt.title("Spectral Distribution (Log)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 2. Filter Magnitude (ASPIRE vs Wiener)
        plt.subplot(1, 3, 2)
        plt.plot(s_vals, wiener_filter + 1e-12, color='blue', linestyle='--', label='Wiener (Baseline)')
        plt.plot(s_vals, filter_w + 1e-12, color='orange', label='ASPIRE (Proposed)')
        plt.yscale('log')
        plt.gca().invert_xaxis() # Head(Large Sigma) on Left
        plt.title(fr"Filter Shape ($\beta={beta_val:.3f}, \alpha={alpha:.1f}$)")
        plt.xlabel(r"Singular Value $\sigma$ (Head $\rightarrow$ Tail)")
        plt.ylabel(r"$h(\sigma)$")
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        
        # 3. Filter Impact (Relative Energy Correction)
        plt.subplot(1, 3, 3)
        plt.plot(s_vals, filter_w - wiener_filter, color='red', label='Correction (ASPIRE - Wiener)')
        plt.title("Spectral Restoration Impact")
        plt.xlabel(r"Singular Value $\sigma$")
        plt.ylabel(r"$\Delta h(\sigma)$")
        plt.axhline(0, color='black', linewidth=1, linestyle='--')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{file_prefix}_analysis.png'))
        plt.close()
        
        print(f"[ASPIREVisualizer] Visualizations saved to {save_dir}")
