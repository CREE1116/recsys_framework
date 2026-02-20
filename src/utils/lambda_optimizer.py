import numpy as np
import matplotlib.pyplot as plt
import os
import json

class LambdaPeakOptimizer:
    """
    Finds the optimal Lambda peak for a metric (NDCG, Coverage, etc.) 
    using Quadratic Interpolation (Parabolic Fitting) in Log-Lambda space.
    Supports local 3-point interpolation or N-point least-squares.
    """
    
    def __init__(self, lambdas, metrics, metric_name="NDCG", use_log_y=False):
        """
        lambdas: List of lambda values (at least 3 required)
        metrics: List of metric values corresponding to lambdas
        use_log_y: If True, fits parabola to log(metrics). Better for skewed peaks.
        """
        if len(lambdas) < 3:
            raise ValueError("At least 3 points are required for quadratic interpolation.")
            
        self.lambdas = np.array(lambdas)
        self.metrics = np.array(metrics)
        self.metric_name = metric_name
        self.use_log_y = use_log_y
        
        # Sort by lambda for consistency
        sort_idx = np.argsort(self.lambdas)
        self.lambdas = self.lambdas[sort_idx]
        self.metrics = self.metrics[sort_idx]
        
        # Log-space transformation (Standard for Lambda)
        self.x = np.log10(self.lambdas)
        
        if self.use_log_y:
            # Shift a bit to avoid log(0)
            self.y = np.log(self.metrics + 1e-9)
        else:
            self.y = self.metrics
        
        self.coeffs = None # [a, b, c] for y = ax^2 + bx + c
        self.peak_x = None
        self.peak_lambda = None
        self.peak_val = None

    def fit(self):
        """
        Fit y = ax^2 + bx + c using Least Squares.
        """
        A = np.column_stack([self.x**2, self.x, np.ones_like(self.x)])
        
        try:
            self.coeffs, _, _, _ = np.linalg.lstsq(A, self.y, rcond=None)
        except Exception as e:
            print(f"[Optimizer] Error during fitting for {self.metric_name}: {e}")
            return None
            
        a, b, c = self.coeffs
        
        if a >= 0:
            # Curve is convex or flat. Vertex is a minimum.
            # Pick the observed best idx
            best_idx = np.argmax(self.metrics)
            self.peak_x = self.x[best_idx]
        else:
            # Peak calculation: x_peak = -b / (2 * a)
            self.peak_x = -b / (2 * a)
            
        # Clip peak_x to a reasonable range (limit extrapolation)
        min_x, max_x = self.x.min() - 0.5, self.x.max() + 0.5
        self.peak_x = np.clip(self.peak_x, min_x, max_x)
        
        self.peak_lambda = 10 ** self.peak_x
        self.peak_val = self.predict(self.peak_x)
        
        return self.peak_lambda

    def predict(self, x_in):
        """Predict original-scale metric for a given log-lambda x."""
        if self.coeffs is None: return 0.0
        a, b, c = self.coeffs
        y_pred = a * x_in**2 + b * x_in + c
        if self.use_log_y:
            return np.exp(y_pred)
        return y_pred

    def visualize(self, save_path=None):
        if self.coeffs is None: return
        
        x_range = np.linspace(min(self.x) - 0.5, max(self.x) + 0.5, 100)
        y_range = [self.predict(xi) for xi in x_range]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.x, self.metrics, color='red', label='Measured', s=100)
        plt.plot(x_range, y_range, '--', label='Quadratic Fit', alpha=0.7)
        plt.scatter([self.peak_x], [self.peak_val], color='blue', marker='*', s=200, label=f'Peak: λ={self.peak_lambda:.1f}')
        
        plt.title(f"{self.metric_name} Fit (concave: {self.coeffs[0]<0})")
        plt.xlabel("log10(Lambda)")
        plt.ylabel(self.metric_name)
        plt.grid(True, alpha=0.3)
        plt.legend()
        if save_path: plt.savefig(save_path, dpi=150)
        plt.show(); plt.close()

def find_intersection(opt1, opt2):
    """
    Find the intersection of two parabolas in log-lambda space.
    """
    if opt1.coeffs is None or opt2.coeffs is None:
        return None
        
    a1, b1, c1 = opt1.coeffs
    a2, b2, c2 = opt2.coeffs
    
    A = a1 - a2
    B = b1 - b2
    C = c1 - c2
    
    delta = B**2 - 4*A*C
    if delta < 0: return None
        
    x1 = (-B + np.sqrt(delta)) / (2 * A) if abs(A) > 1e-9 else -C/B
    x2 = (-B - np.sqrt(delta)) / (2 * A) if abs(A) > 1e-9 else -C/B
    
    x_min, x_max = min(opt1.x.min(), opt2.x.min()), max(opt1.x.max(), opt2.x.max())
    candidates = [x for x in [x1, x2] if x_min <= x <= x_max]
    
    if not candidates:
        center = (x_min + x_max) / 2
        return x1 if abs(x1-center) < abs(x2-center) else x2
        
    return candidates[0]

def visualize_joint(opt_ndcg, opt_cov, save_path=None):
    """
    Combined plot with normalized curves to find the Balanced Sweet Spot.
    """
    x_min = min(opt_ndcg.x.min(), opt_cov.x.min())
    x_max = max(opt_ndcg.x.max(), opt_cov.x.max())
    x_range = np.linspace(x_min - 0.2, x_max + 0.2, 100)
    
    y_ndcg = np.array([opt_ndcg.predict(xi) for xi in x_range])
    y_cov = np.array([opt_cov.predict(xi) for xi in x_range])
    
    # Normalization for "Intersection" visibility
    # Note: Intersection is scale-dependent, we find the point where normalized importance is balanced.
    y1_n = (y_ndcg - y_ndcg.min()) / (y_ndcg.max() - y_ndcg.min() + 1e-9)
    y2_n = (y_cov - y_cov.min()) / (y_cov.max() - y_cov.min() + 1e-9)
    
    # Intersection of normalized curves
    idx = np.argmin(np.abs(y1_n - y2_n))
    sweet_x = x_range[idx]
    sweet_lambda = 10**sweet_x
    
    # Max product peak
    y_prod = y_ndcg * y_cov
    p_idx = np.argmax(y_prod)
    prod_lambda = 10**x_range[p_idx]

    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    ax1.set_xlabel('log10(Lambda)')
    ax1.set_ylabel('NDCG@10', color='tab:blue')
    ax1.plot(x_range, y_ndcg, color='tab:blue', linestyle='--', label='NDCG Fit')
    ax1.scatter(opt_ndcg.x, opt_ndcg.metrics, color='tab:blue', alpha=0.5)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Coverage@10', color='tab:red')
    ax2.plot(x_range, y_cov, color='tab:red', linestyle='--', label='Coverage Fit')
    ax2.scatter(opt_cov.x, opt_cov.metrics, color='tab:red', alpha=0.5)
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    plt.axvline(x=sweet_x, color='green', linestyle=':', label=f'Balanced Sweet Spot λ={sweet_lambda:.0f}')
    plt.axvline(x=x_range[p_idx], color='purple', linestyle='-.', label=f'Peak Product λ={prod_lambda:.0f}')
    
    plt.title(f"Joint Analysis: Sweet Spot vs Max Product\n"
              f"Balanced λ ≈ {sweet_lambda:.0f}, Max Product λ ≈ {prod_lambda:.0f}")
    
    fig.tight_layout()
    ax1.grid(True, alpha=0.2)
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[Optimizer] Joint plot saved to {save_path}")
    plt.show(); plt.close()
    
    return sweet_lambda, prod_lambda

def optimize_from_dict(raw_data, metric_name="NDCG"):
    """
    Smart selection: If N > 3, use 3 local points around the maximum for higher precision.
    """
    all_lambdas = np.array(sorted(list(raw_data.keys())))
    all_metrics = np.array([raw_data[l] for l in all_lambdas])
    
    # Find Local 3 points around max
    best_idx = np.argmax(all_metrics)
    start = max(0, best_idx - 1)
    if start + 3 > len(all_lambdas):
        start = max(0, len(all_lambdas) - 3)
    
    sub_lambdas = all_lambdas[start:start+3]
    sub_metrics = all_metrics[start:start+3]
    
    use_log = "NDCG" in metric_name
    
    # We use all points for a 'Global Fit' intended for visualization/joint,
    # but the individual 'Peak' should ideally be local.
    # To satisfy both, we'll fit on ALL points but maybe we should provide a choice.
    # User's "Something isn't right" probably came from Global LS fit being bad.
    # Let's switch back to LOCAL 3 for the individual result.
    
    opt = LambdaPeakOptimizer(sub_lambdas, sub_metrics, metric_name=metric_name, use_log_y=use_log)
    opt.fit()
    return opt

if __name__ == "__main__":
    dummy_ndcg = {100: 0.02, 1000: 0.04, 10000: 0.042, 100000: 0.038}
    dummy_cov = {100: 0.9, 1000: 0.97, 10000: 0.5, 100000: 0.2} # Sharp peak at 1000
    
    opt1 = optimize_from_dict(dummy_ndcg, "NDCG@10")
    opt2 = optimize_from_dict(dummy_cov, "Coverage@10")
    
    print(f"NDCG Peak: {opt1.peak_lambda:.2f}")
    print(f"Coverage Peak: {opt2.peak_lambda:.2f}")
    
    visualize_joint(opt1, opt2)
