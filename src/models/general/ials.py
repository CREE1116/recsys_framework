import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from ..base_model import BaseModel
import time
from tqdm import tqdm


class iALS(BaseModel):
    """
    Implicit ALS (iALS) with Batched GPU Solve.
    
    Standard iALS per-user update:
        A_u = V^T V + alpha * V[I_u]^T @ V[I_u] + lambda * I
        b_u = (1 + alpha) * V[I_u].sum(0)
        u_u = solve(A_u, b_u)

    [Optimization] Row-by-row loop replaced with batched approach:
    1. Build A_u and b_u for a batch of users using pre-extracted CSR indices (CPU numpy, no per-user GPU transfer).
    2. Solve all systems in one call: torch.linalg.solve((B, d, d), (B, d, 1)) on GPU.

    Reference: Rendle et al., "Revisiting the Performance of iALS"
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

        # batch_users: how many users/items to process per GPU solve call
        self.batch_users = config['model'].get('batch_users', 1024)
        if isinstance(self.batch_users, list):
            self.batch_users = self.batch_users[0]

        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

        self.device = self.user_embedding.weight.device

    # --------------------------------------------------------
    # Batched ALS Update (GPU)
    # --------------------------------------------------------
    def _update_batched(self, Target, Fixed, FixedTFixed, InteractionMat):
        """
        Batch ALS update: solve all linear systems in chunks via torch.linalg.solve.
        No per-row CG — exact direct solve, fully GPU-batched.
        
        Args:
            Target: (N, d) embedding matrix to update (in-place)
            Fixed: (M, d) the other factor (item or user)
            FixedTFixed: (d, d) precomputed F^T F
            InteractionMat: scipy CSR matrix (N, M)
        """
        N, d = Target.shape
        device = Target.device
        batch_size = self.batch_users
        alpha = float(self.alpha)
        reg = float(self.reg_lambda)

        # Precompute base: P = F^T F + lambda * I  (d x d) — shared across all rows
        P = FixedTFixed + reg * torch.eye(d, device=device)  # (d, d)
        P_np = P.cpu().numpy().astype(np.float32)

        # Fixed as numpy for fast per-user indexing (avoids repeated .to(device))
        Fixed_np = Fixed.detach().cpu().numpy().astype(np.float32)

        # Precomputed CSR data for fast row slicing
        indptr = InteractionMat.indptr
        indices = InteractionMat.indices

        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            B = batch_end - batch_start

            # Pre-allocate A and b on CPU numpy (fast for small d x d)
            A_batch = np.broadcast_to(P_np, (B, d, d)).copy()   # (B, d, d)
            b_batch = np.zeros((B, d), dtype=np.float32)         # (B, d)
            zero_mask = np.zeros(B, dtype=bool)

            for i, u in enumerate(range(batch_start, batch_end)):
                start_ptr = indptr[u]
                end_ptr = indptr[u + 1]
                if start_ptr == end_ptr:
                    zero_mask[i] = True
                    continue
                idx = indices[start_ptr:end_ptr]          # item indices for user u
                V_u = Fixed_np[idx]                       # (L, d) numpy
                # A_u += alpha * V_u^T @ V_u
                A_batch[i] += alpha * (V_u.T @ V_u)
                # b_u = (1 + alpha) * V_u.sum(0)
                b_batch[i] = (1.0 + alpha) * V_u.sum(axis=0)

            # Move to GPU once per batch
            A_t = torch.from_numpy(A_batch).to(device)    # (B, d, d)
            b_t = torch.from_numpy(b_batch).to(device)    # (B, d)

            # Batched direct solve: A_t @ x = b_t  =>  x = A_t^{-1} b_t
            # torch.linalg.solve supports batched (B, d, d) @ (B, d, 1) -> (B, d)
            x_t = torch.linalg.solve(A_t, b_t.unsqueeze(-1)).squeeze(-1)   # (B, d)

            # Zero out users with no interactions
            if zero_mask.any():
                x_t[torch.from_numpy(zero_mask).to(device)] = 0.0

            Target[batch_start:batch_end] = x_t

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    def fit(self, data_loader):
        print(f"\n[iALS] Fitting with d={self.embedding_dim}, "
              f"lambda={self.reg_lambda}, alpha={self.alpha}, "
              f"iter={self.max_iter}, batch_users={self.batch_users}")
        print(f"[iALS] Device: {self.device}")

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
        Xt = X.T.tocsr()

        U = self.user_embedding.weight.data   # (N, d) on device
        V = self.item_embedding.weight.data   # (M, d) on device

        loss_history = []

        pbar = tqdm(range(self.max_iter), desc="[iALS] Training", unit="iter")
        for it in pbar:
            t0 = time.time()

            # --- Update Users ---
            VtV = V.t() @ V                      # (d, d)
            self._update_batched(U, V, VtV, X)

            # --- Update Items ---
            UtU = U.t() @ U                      # (d, d)
            self._update_batched(V, U, UtU, Xt)

            # --- Fast Approximate Loss (trace trick) ---
            with torch.no_grad():
                UtU = U.t() @ U
                VtV = V.t() @ V
                trace_term = (UtU * VtV).sum()

                coo = X.tocoo()
                row_idx = torch.from_numpy(coo.row.copy()).long().to(self.device)
                col_idx = torch.from_numpy(coo.col.copy()).long().to(self.device)

                pos_loss_sum = 0.0
                chunk = 200000
                num_nnz = len(row_idx)
                for ci in range(0, num_nnz, chunk):
                    u_b = row_idx[ci:ci+chunk]
                    v_b = col_idx[ci:ci+chunk]
                    y = (U[u_b] * V[v_b]).sum(dim=1)
                    pos_loss_sum += ((1 + self.alpha) - 2*(1+self.alpha)*y + self.alpha*(y**2)).sum().item()

                reg_loss = self.reg_lambda * ((U**2).sum() + (V**2).sum())
                total_loss = trace_term.item() + pos_loss_sum + reg_loss.item()
                loss_history.append(total_loss)

            iter_time = time.time() - t0
            pbar.set_postfix(loss=f"{total_loss:.4e}", t=f"{iter_time:.1f}s")


        elapsed = time.time() - start_time
        print(f"[iALS] Done. Total: {elapsed:.1f}s")
        self.train_losses = {'total_loss': loss_history}

        try:
            import os, json
            from ...utils.plotting import plot_results

            dataset_name = self.config.get('dataset_name', 'default')
            model_name = self.config['model']['name']
            run_name = self.config.get('run_name')

            output_path = self.output_path

            if os.path.exists(output_path):
                with open(os.path.join(output_path, 'losses_history.json'), 'w') as f:
                    json.dump({'total_loss': loss_history}, f, indent=4)
                plot_results(
                    data_dict={'total_loss': loss_history},
                    title="iALS Training Loss",
                    xlabel="Iteration",
                    ylabel="Loss",
                    file_path=os.path.join(output_path, 'total_loss_plot.png')
                )
        except Exception as e:
            print(f"[iALS] Could not save loss plot: {e}")

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------
    def forward(self, user_ids, item_ids=None):
        if not isinstance(user_ids, torch.Tensor):
            user_ids = torch.tensor(user_ids, device=self.device)

        users = self.user_embedding(user_ids)

        if item_ids is not None:
            if not isinstance(item_ids, torch.Tensor):
                item_ids = torch.tensor(item_ids, device=self.device)
            items = self.item_embedding(item_ids)
            return (users * items).sum(dim=-1)

        return users @ self.item_embedding.weight.t()

    def predict_for_pairs(self, user_ids, item_ids):
        return self.forward(user_ids, item_ids)

    def calc_loss(self, batch_data):
        return torch.tensor(0.0, device=self.device), None

    def get_final_item_embeddings(self):
        return self.item_embedding.weight.data
