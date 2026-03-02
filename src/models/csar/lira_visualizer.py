import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

class LIRAVisualizer:
    @staticmethod
    def visualize_dense_lira(S, K_Gram=None, S_raw=None, save_dir=None):
        if not save_dir:
            return
            
        os.makedirs(save_dir, exist_ok=True)
        
        S_tensor = S.detach().cpu()
        S_np = S_tensor.numpy()
        
        metrics = {}
        
        def get_stats(name, tensor):
            return {
                "min": float(tensor.min()),
                "max": float(tensor.max()),
                "mean": float(tensor.mean()),
                "std": float(tensor.std()),
                "sparsity": float((tensor.abs() < 1e-6).sum() / tensor.numel())
            }

        metrics['S_norm'] = get_stats('S_norm', S_tensor)
        
        if S_raw is not None:
            S_raw_tensor = S_raw.detach().cpu()
            S_raw_np = S_raw_tensor.numpy()
            metrics['S_raw'] = get_stats('S_raw', S_raw_tensor)
            
        if K_Gram is not None:
            gram_tensor = K_Gram.detach().cpu()
            K_np = gram_tensor.numpy()
            metrics['K_Gram'] = get_stats('K_Gram', gram_tensor)

        metrics_path = os.path.join(save_dir, 'lira_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        def plot_heatmap(matrix, name, filename, cmap='viridis', center=None, vmin=None, vmax=None):
            plt.figure(figsize=(10, 8))
            N, M = matrix.shape
            if N > 1000 or M > 1000:
                step_n = max(1, N // 1000)
                step_m = max(1, M // 1000)
                matrix_viz = matrix[::step_n, ::step_m]
            else:
                matrix_viz = matrix
            sns.heatmap(matrix_viz, cmap=cmap, center=center, square=True, vmax=vmax, vmin=vmin)
            plt.title(name)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, filename), dpi=150)
            plt.close()

        if K_Gram is not None:
            plot_heatmap(K_np, 'Gram Matrix K', 'viz_K_Gram.png')
            
        if S_raw is not None:
            vmax_raw = np.percentile(np.abs(S_raw_np), 99)
            plot_heatmap(S_raw_np, 'S Matrix (Raw)', 'viz_S_Raw.png', vmax=vmax_raw, vmin=-vmax_raw/10)
            
        vmax_norm = np.percentile(np.abs(S_np), 99)
        plot_heatmap(S_np, 'S Matrix (Norm)', 'viz_S_Norm.png', vmax=vmax_norm, vmin=-vmax_norm/10)
        
        print(f"[LIRAVisualizer] Visualizations saved to {save_dir}")

    @staticmethod
    def visualize_svd_lira(singular_values, filter_diag, reg_lambda, X_sparse=None, save_dir=None, file_prefix='lightlira'):
        if not save_dir:
            return
            
        os.makedirs(save_dir, exist_ok=True)
        metrics = {}
        
        s_vals = singular_values.detach().cpu().numpy()
        s2 = s_vals**2
        cum_energy = np.cumsum(s2)
        
        metrics['SingularValues'] = {"mean": float(s_vals.mean()), "std": float(s_vals.std())}
        
        if X_sparse is not None:
            total_energy = X_sparse.power(2).sum()
            metrics['SVD_Energy'] = {
                "captured_energy": float(cum_energy[-1]),
                "total_energy": float(total_energy),
                "ratio": float(cum_energy[-1] / total_energy)
            }
            cum_energy_ratio = cum_energy / total_energy
        else:
            cum_energy_ratio = cum_energy / (cum_energy[-1] + 1e-9)
            
        filter_w = filter_diag.detach().cpu().numpy()

        with open(os.path.join(save_dir, f'{file_prefix}_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(s_vals, label=r'$\sigma_k$')
        plt.yscale('log')
        plt.title("Spectrum")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(cum_energy_ratio, color='green')
        plt.title(f"Energy: {cum_energy_ratio[-1]*100:.1f}%")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.plot(filter_w, color='orange')
        plt.title(fr"Filter ($\lambda={reg_lambda}$)")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{file_prefix}_spectrum.png'))
        plt.close()
        
        print(f"[LIRAVisualizer] SVD Visualizations saved to {save_dir}")

    @staticmethod
    def visualize_spectral_tikhonov(singular_values, filter_diag, alpha, beta, X_sparse=None, save_dir=None, file_prefix='spectral_tikhonov'):
        if not save_dir:
            return
            
        os.makedirs(save_dir, exist_ok=True)
        metrics = {}
        
        s_vals = singular_values.detach().cpu().numpy()
        s2 = s_vals**2
        
        # Calculate standard Wiener Filter to compare against
        wiener_filter = s2 / (s2 + alpha)
        
        cum_energy = np.cumsum(s2)
        
        metrics['SingularValues'] = {"mean": float(s_vals.mean()), "std": float(s_vals.std())}
        
        # Calculate filter statistics
        filter_w = filter_diag.detach().cpu().numpy()
        metrics['Filter_Tikhonov'] = {
            "mean": float(filter_w.mean()),
            "std": float(filter_w.std()),
            "max": float(filter_w.max()),
            "min": float(filter_w.min())
        }
        
        metrics['Filter_Wiener_Reference'] = {
            "mean": float(wiener_filter.mean()),
            "std": float(wiener_filter.std()),
            "max": float(wiener_filter.max()),
            "min": float(wiener_filter.min())
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
            
        with open(os.path.join(save_dir, f'{file_prefix}_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

        plt.figure(figsize=(20, 5))
        
        plt.subplot(1, 4, 1)
        plt.plot(s_vals, label=r'$\sigma_k$')
        plt.yscale('log')
        plt.title("Spectrum (Log Scale)")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 4, 2)
        plt.plot(cum_energy_ratio, color='green')
        plt.title(f"Cumulative Energy: {cum_energy_ratio[-1]*100:.1f}%")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 4, 3)
        plt.plot(s_vals, wiener_filter, color='blue', label=r'Wiener ($\beta=0$)')
        plt.plot(s_vals, filter_w, color='orange', label=fr'Tikhonov ($\beta={beta}$)')
        plt.title(fr"Filters $h(\sigma)$ vs $\sigma$ ($\alpha={alpha}$)")
        plt.xlabel(r"Singular Value $\sigma$")
        plt.ylabel(r"Filter Value $h(\sigma)$")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 4, 4)
        plt.plot(s_vals, filter_w - wiener_filter, color='red', label='Difference (Tikhonov - Wiener)')
        plt.title(r"Filter Impact vs $\sigma$")
        plt.xlabel(r"Singular Value $\sigma$")
        plt.ylabel(r"$\Delta h(\sigma)$")
        plt.axhline(0, color='black', linewidth=1, linestyle='--')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{file_prefix}_spectrum.png'))
        plt.close()
        
        print(f"[LIRAVisualizer] Spectral Tikhonov Visualizations saved to {save_dir}")

    @staticmethod
    def visualize_sparse_lira(S_sparse, save_dir=None, title_suffix=''):
        if not save_dir:
            return
            
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 10))
        S_dense = S_sparse.to_dense().detach().cpu().numpy()
        plt.imshow(S_dense, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title(f"Sparse Kernel S {title_suffix}")
        plt.savefig(os.path.join(save_dir, "S_matrix.png"))
        plt.close()
        
        S_values = S_dense[S_dense != 0]
        plt.figure(figsize=(8, 5))
        plt.hist(S_values, bins=100)
        plt.title("Distribution of Non-Zero S values")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(save_dir, "S_hist.png"))
        plt.close()
        
        print(f"[LIRAVisualizer] Sparse Visualizations saved to {save_dir}")
