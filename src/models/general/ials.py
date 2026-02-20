import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from ..base_model import BaseModel
import time

class iALS(BaseModel):
    """
    Strict Implicit ALS (CG Solver)
    - Mathematically exact
    - CG solver (no approximation)
    - Reference: Rendle et al., "Revisiting the Performance of iALS"
    """

    def __init__(self, config, data_loader):
        super(iALS, self).__init__(config, data_loader)
        
        self.embedding_dim = config['model'].get('embedding_dim', 512)
        if isinstance(self.embedding_dim, list):
            self.embedding_dim = self.embedding_dim[0]
            
        self.reg_lambda = config['model'].get('reg_lambda', 0.01)
        if isinstance(self.reg_lambda, list):
            self.reg_lambda = self.reg_lambda[0]
            
        self.alpha = config['model'].get('alpha', 40)
        if isinstance(self.alpha, list):
            self.alpha = self.alpha[0]
            
        self.max_iter = config['model'].get('max_iter', 15)
        if isinstance(self.max_iter, list):
            self.max_iter = self.max_iter[0]
            
        self.cg_steps = config['model'].get('cg_steps', 3)
        if isinstance(self.cg_steps, list):
            self.cg_steps = self.cg_steps[0]

        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

        self.device = self.user_embedding.weight.device

    # --------------------------------------------------------
    # Conjugate Gradient
    # --------------------------------------------------------
    def _cg(self, matvec, b, x0=None):

        d = b.shape[0]
        x = torch.zeros_like(b) if x0 is None else x0

        r = b - matvec(x)
        p = r.clone()
        rs_old = torch.dot(r, r)

        for _ in range(self.cg_steps):
            Ap = matvec(p)
            alpha = rs_old / (torch.dot(p, Ap) + 1e-8)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.dot(r, r)

            if torch.sqrt(rs_new) < 1e-6:
                break

            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

        return x

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    def fit(self, data_loader):

        print(f"\n[iALS] Fitting with d={self.embedding_dim}, "
              f"lambda={self.reg_lambda}, alpha={self.alpha}, "
              f"iter={self.max_iter}, cg_steps={self.cg_steps}")
        
        start_time = time.time()

        train_df = data_loader.train_df
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        values = np.ones(len(train_df), dtype=np.float32)

        X = sp.csr_matrix(
            (values, (rows, cols)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32
        )

        U = self.user_embedding.weight.data
        V = self.item_embedding.weight.data

        # Precompute lambda * I
        lambda_I = self.reg_lambda * torch.eye(self.embedding_dim, device=self.device)
        
        loss_history = []

        for it in range(self.max_iter):

            t0 = time.time()

            # --- Update Users ---
            VtV = V.t() @ V
            self._update(U, V, VtV, X)

            # --- Update Items ---
            Xt = X.transpose().tocsr()
            UtU = U.t() @ U
            self._update(V, U, UtU, Xt)
            
            # --- Calculate Loss (Approx/Exact) ---
            # Loss = sum_{u,i} c_{ui} (p_{ui} - u_u^T v_i)^2 + lambda (|U|^2 + |V|^2)
            # This is expensive to compute exactly for all pairs.
            # We use the efficient decomposition:
            # Loss = sum_{all} (u^T v)^2  <-- from trace(UtU * VtV)
            #      + sum_{observed} [ (1+alpha)(1 - 2*y + y^2) - y^2 ] 
            #      where y = u^T v for observed pairs
            #      + Regularization
            
            with torch.no_grad():
                # 1. Trace term (sum of squares of all predictions)
                # trace(U^T U * V^T V) = sum((U^T U) * (V^T V))
                # Note: UtU and VtV are already computed above (but VtV might be stale if we used it for U update? No, VtV is item cov, updated after item step? No, VtV used for U update. UtU used for V update.)
                # Actually, at end of iter, U and V are updated. We need fresh UtU and VtV.
                
                UtU = U.t() @ U
                VtV = V.t() @ V
                trace_term = (UtU * VtV).sum()
                
                # 2. Observed term
                # We need predictions for all observed pairs.
                # X is CSR.
                # We can iterate or use sparse ops?
                # For efficiency, we might skip this log every iter, or do it efficiently.
                # Let's use indices from self.train_matrix_csr
                
                # Coalesce to batch for GPU calc
                coo = X.tocoo()
                row_indices = torch.from_numpy(coo.row).to(self.device).long()
                col_indices = torch.from_numpy(coo.col).to(self.device).long()
                
                # Predict batch
                # To avoid OOM on large datasets, process in chunks
                pos_loss_sum = 0
                chunk_size = 100000
                num_interactions = len(row_indices)
                
                for i in range(0, num_interactions, chunk_size):
                    u_batch = row_indices[i:i+chunk_size]
                    v_batch = col_indices[i:i+chunk_size]
                    
                    user_emb = U[u_batch]
                    item_emb = V[v_batch]
                    
                    y_pred = (user_emb * item_emb).sum(dim=1)
                    
                    # Term: (1+alpha)*(1 - 2y + y^2) - y^2
                    #     = (1+alpha) - 2(1+alpha)y + (1+alpha)y^2 - y^2
                    #     = (1+alpha) - 2(1+alpha)y + alpha*y^2
                    
                    term = (1 + self.alpha) - 2 * (1 + self.alpha) * y_pred + self.alpha * (y_pred ** 2)
                    pos_loss_sum += term.sum().item()
                    
                # 3. Regularization
                reg_loss = self.reg_lambda * ( (U**2).sum() + (V**2).sum() )
                
                total_loss = trace_term.item() + pos_loss_sum + reg_loss.item()
                loss_history.append(total_loss)

            print(f"[iALS] Iter {it+1}/{self.max_iter} Loss={total_loss:.4e} "
                  f"({time.time()-t0:.2f}s)")
                  
        elapsed = time.time() - start_time
        print(f"[iALS] Training complete. Time: {elapsed:.2f}s")
        
        # Save loss history for plotting
        self.train_losses = {'total_loss': loss_history}
        
        # We need to manually trigger saving/plotting because Trainer.train() is skipped.
        # But Trainer calls fit(), then evaluate(). 
        # Trainer._visualize_results() is called at end of train(), but here train() is not called.
        # So we should probably save it to a file here, or attach it so Trainer can find it?
        # Trainer isn't looking for self.model.train_losses.
        
        # Let's save it directly here as 'losses_history.json' in the current model directory?
        # But we don't know the output directory here easily (it's in Trainer).
        # Actually, if we just store it in self.train_losses, we can update Trainer to look for it?
        # Or more simply, since we are inside 'run_all_smart_searches' -> 'main' -> 'trainer.train' (skipped) -> 'evaluate'
        # The 'BEST' run executes 'run_single_experiment' which calls 'main'.
        # 'main' checks: if 'train' in config: trainer.train() else: model.fit(), trainer.evaluate().
        # So for iALS (no 'train' config), trainer.train() is NEVER called.
        # Thus trainer._visualize_results() is NEVER called.
        
        # So we MUST generate the plot here if we want it.
        # But we need the output path.
        # Config usually has 'run_name' or we can deduce it?
        # Alternatively, we can add 'train' section to iALS config just to trigger Trainer?
        # No, that would trigger gradient descent training.
        
        # Best bet: Save to self.train_losses and rely on the fact that we can call
        # plot_results from here if we knew the path. 
        # Note: Trainer passes 'config' to model. We can maybe find path from config?
        # Trainer constructs path: 'trained_model/{dataset}/{model_name}__{run_name}'
        
        try:
             import os
             import json
             from ...utils.plotting import plot_results
             
             dataset_name = self.config.get('dataset_name', 'default')
             model_name = self.config['model']['name']
             run_name = self.config.get('run_name')
             
             base_path = os.path.join('trained_model', dataset_name)
             if run_name and run_name != 'default':
                 output_path = os.path.join(base_path, f"{model_name}__{run_name}")
             else:
                 output_path = os.path.join(base_path, model_name)
                 
             if os.path.exists(output_path):
                 loss_file = os.path.join(output_path, 'losses_history.json')
                 with open(loss_file, 'w') as f:
                     json.dump({'total_loss': loss_history}, f, indent=4)
                     
                 plot_path = os.path.join(output_path, 'total_loss_plot.png')
                 plot_results(
                    data_dict={'total_loss': loss_history},
                    title="iALS Training Loss",
                    xlabel="Iteration",
                    ylabel="Loss",
                    file_path=plot_path
                 )
                 print(f"[iALS] Loss plot saved to {plot_path}")
        except Exception as e:
            print(f"[iALS] Failed to save loss plot: {e}")

    # --------------------------------------------------------
    # Exact update using CG
    # --------------------------------------------------------
    def _update(self, Target, Fixed, FixedTFixed, InteractionMat):

        for u in range(Target.shape[0]):

            start = InteractionMat.indptr[u]
            end = InteractionMat.indptr[u + 1]

            if start == end:
                Target[u] = torch.zeros(self.embedding_dim,
                                        device=self.device)
                continue

            idx = InteractionMat.indices[start:end]
            idx = torch.tensor(idx, device=self.device)

            V_u = Fixed[idx]  # [L, d]

            # RHS
            b = (1 + self.alpha) * V_u.sum(dim=0)

            # Define matvec(A_u x)
            def matvec(x):

                term1 = FixedTFixed @ x
                term2 = self.alpha * (V_u.t() @ (V_u @ x))
                term3 = self.reg_lambda * x

                return term1 + term2 + term3

            Target[u] = self._cg(matvec, b, Target[u])

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------
    def forward(self, user_ids, item_ids=None):

        if not isinstance(user_ids, torch.Tensor):
            user_ids = torch.tensor(user_ids,
                                    device=self.device)

        users = self.user_embedding(user_ids)

        if item_ids is not None:
            if not isinstance(item_ids, torch.Tensor):
                item_ids = torch.tensor(item_ids,
                                        device=self.device)
            items = self.item_embedding(item_ids)
            return (users * items).sum(dim=-1)

        return users @ self.item_embedding.weight.t()

    def predict_for_pairs(self, user_ids, item_ids):
        return self.forward(user_ids, item_ids)

    def calc_loss(self, batch_data):
        return torch.tensor(0.0, device=self.device), None

    def get_final_item_embeddings(self):
        return self.item_embedding.weight.data
