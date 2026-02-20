
import torch
import numpy as np
import scipy.sparse as sp
from src.data_loader import DataLoader
from src.evaluation import evaluate_metrics
import time
import yaml
import copy

# Mock Config
config_str = """
dataset_name: ml100k
data_path: "/Users/leejongmin/code/recsys_framework/data/ml100k/u.data"
separator: "\t"
columns: ["user_id", "item_id", "rating", "timestamp"]
rating_threshold: 4
min_user_interactions: 5
min_item_interactions: 5

model:
    name: infinity_ae
    reg_lambda: 100.0

evaluation:
    validation_method: "sampled"
    final_method: "full"
    metrics: ["NDCG", "HitRate"]
    top_k: [10]
    main_metric: "NDCG"
    main_metric_k: 10
    long_tail_percent: 0.8
    batch_size: 1024

train:
    epochs: 1
    batch_size: 1024
    
device: "mps"
"""

class InfinityAE_Variant(torch.nn.Module):
    def __init__(self, config, data_loader, kernel_type="matrix_square"):
        super().__init__()
        self.config = config
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.reg_lambda = config['model']['reg_lambda']
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        self.kernel_type = kernel_type
        
        train_df = data_loader.train_df
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        values = np.ones(len(train_df), dtype=np.float32)
        
        self.train_matrix_csr = sp.csr_matrix(
            (values, (rows, cols)), 
            shape=(self.n_users, self.n_items),
            dtype=np.float32
        )
        self.weight_matrix = None
        
    def fit(self):
        print(f"[{self.kernel_type}] Fitting...")
        X = torch.from_numpy(self.train_matrix_csr.toarray()).float().to(self.device)
        
        # [User Request] Normalize by degree
        deg = torch.sum(X, dim=1, keepdim=True)
        X = X / torch.sqrt(deg + 1e-8)
        
        # 1. Gram Matrix
        G = X @ X.t()
        
        # 2. Kernel
        if self.kernel_type == "matrix_square": # Current Impl
            K = G @ G
        elif self.kernel_type == "element_square": # Polynomial
            K = G * G
        elif self.kernel_type == "linear" or self.kernel_type == "corrected_linear": # EASE-like
            K = G
        elif self.kernel_type == "relu": # ArcCosine 0 (simplified)
            # Simple approximation or just skipping for now
             K = G # Placeholder
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_type}")
            
        # 3. Solve
        diag_indices = torch.arange(self.n_users, device=self.device)
        K_reg = K.clone()
        K_reg[diag_indices, diag_indices] += self.reg_lambda
        
        try:
             if self.kernel_type == "corrected_linear":
                 # Solves (K + lambda I) * Alpha = I  => Alpha = (K + lambda I)^-1
                 lhs = torch.eye(self.n_users, device=self.device)
                 Alpha = torch.linalg.solve(K_reg, lhs)
             else:
                 # Original Code: Solves (K + lambda I) * Alpha = K
                 Alpha = torch.linalg.solve(K_reg, K)
        except:
             if self.kernel_type == "corrected_linear":
                 lhs = torch.eye(self.n_users)
                 Alpha = torch.linalg.solve(K_reg.cpu(), lhs).to(self.device)
             else:
                 Alpha = torch.linalg.solve(K_reg.cpu(), K.cpu()).to(self.device)
             
        # 4. Weights
        # B = X.T @ Alpha @ X
        temp = Alpha @ X
        B = X.t() @ temp
        B.fill_diagonal_(0)
        self.weight_matrix = B
        
    def forward(self, user_ids, item_ids=None):
        return self.predict_for_pairs(user_ids, item_ids)

    def predict_for_pairs(self, user_ids, item_ids=None):
        if isinstance(user_ids, torch.Tensor):
            u_ids_np = user_ids.cpu().numpy()
        else:
            u_ids_np = np.array(user_ids)
            
        user_input_sparse = self.train_matrix_csr[u_ids_np]
        user_input = torch.from_numpy(user_input_sparse.toarray()).float().to(self.device)
        
        scores = user_input @ self.weight_matrix
        
        if item_ids is not None:
            if not isinstance(item_ids, torch.Tensor):
                item_ids = torch.tensor(item_ids, device=self.device)
            
            batch_indices = torch.arange(len(user_ids), device=self.device)
            return scores[batch_indices, item_ids]
        
        return scores
    
    def get_final_item_embeddings(self):
        return self.weight_matrix

import traceback

def main():
    config = yaml.safe_load(config_str)
    
    # Load Data
    data_loader = DataLoader(config)
    
    # Test Variants
    variants = ["matrix_square"]
    lambdas = [100, 1000, 10000, 100000, 1000000]
    
    for v in variants:
        print(f"\n================ Testing {v} ================")
        for lam in lambdas:
            config['model']['reg_lambda'] = lam
            model = InfinityAE_Variant(config, data_loader, kernel_type=v)
            try:
                model.fit()
                
                # Quick Evaluation (Sampled Valid)
                valid_loader = data_loader.get_sampled_valid_loader(batch_size=1024, ratio=0.1)
                metrics = evaluate_metrics(model, data_loader, config['evaluation'], model.device, test_loader=valid_loader, is_final=False)
                print(f"[{v}] Lambda={lam} -> NDCG@10: {metrics['NDCG@10']:.4f}")
            except Exception as e:
                traceback.print_exc()
                print(f"[{v}] Lambda={lam} Failed: {e}")

if __name__ == "__main__":
    main()
