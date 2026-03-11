import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from ..base_model import BaseModel
from ...utils.gpu_accel import get_device
from tqdm import tqdm
import time


class iALS(BaseModel):

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)

        self.embedding_dim = config['model'].get('embedding_dim', 128)
        self.reg_lambda = config['model'].get('reg_lambda', 0.01)
        self.alpha = config['model'].get('alpha', 40)
        self.max_iter = config['model'].get('max_iter', 15)

        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

        # Automatic optimal device selection (MPS/CUDA/CPU)
        self.device = get_device('auto')
        self.batch_users = config['model'].get('batch_users', 1024)

    # --------------------------------------------------------
    # Batched GPU ALS Update
    # --------------------------------------------------------

    def _update_batched(self, Target, Fixed, FixedTFixed, indptr, indices):
        """
        Fully vectorized, loopless GPU batched update.
        Replaces slow CPU loops with batched Cholesky solves.
        """
        N, d = Target.shape
        device = self.device

        alpha = float(self.alpha)
        reg = float(self.reg_lambda)

        # Precompute base: P = F^T F + lambda * I  (d x d) — shared across all rows
        P = FixedTFixed + reg * torch.eye(d, device=device)  # (d, d)

        for batch_start in range(0, N, self.batch_users):
            batch_end = min(batch_start + self.batch_users, N)
            B = batch_end - batch_start

            start_idx = indptr[batch_start].item()
            end_idx = indptr[batch_end].item()
            
            zero_mask = (indptr[batch_start:batch_end] == indptr[batch_start+1:batch_end+1])
            lens = indptr[batch_start+1:batch_end+1] - indptr[batch_start:batch_end]
            max_len = lens.max().item()

            if max_len > 0:
                # Build padding mask (B, max_len)
                mask = torch.arange(max_len, device=device).unsqueeze(0) < lens.unsqueeze(1)
                
                # Fetch indices - default to 0 for padding to avoid out-of-bounds
                padded_idx = torch.zeros((B, max_len), dtype=torch.long, device=device)
                batch_indices = indices[start_idx:end_idx]
                padded_idx[mask] = batch_indices
                
                # Gather and mask embeddings
                V_u_padded = Fixed[padded_idx] * mask.unsqueeze(-1).float() # (B, max_len, d)
                
                # Batched matrix multiplication (highly efficient on GPU)
                A_u_batch = torch.bmm(V_u_padded.transpose(1, 2), V_u_padded) # (B, d, d)
                b_batch = (1.0 + alpha) * V_u_padded.sum(dim=1) # (B, d)
                
                A_batch = P.unsqueeze(0) + alpha * A_u_batch
            else:
                A_batch = P.unsqueeze(0).repeat(B, 1, 1)
                b_batch = torch.zeros((B, d), dtype=torch.float32, device=device)

            # Batched GPU Solve
            # CUDA fully supports batched Cholesky and is extremely fast.
            # MPS (Apple Silicon) struggles with batched Cholesky, so we use a CPU fallback for it.
            if device.type == 'cuda':
                try:
                    L = torch.linalg.cholesky(A_batch)
                    x_t = torch.cholesky_solve(b_batch.unsqueeze(-1), L).squeeze(-1)   # (B, d)
                except RuntimeError:
                    # Fallback to general solver if Cholesky fails on CUDA (e.g. numerical instability)
                    x_t = torch.linalg.solve(A_batch, b_batch.unsqueeze(-1)).squeeze(-1)
            else:
                # Move minimal data to CPU for batched Cholesky
                # PyTorch's native CPU batched cholesky uses optimized ATen/OpenMP backend
                A_cpu = A_batch.cpu()
                b_cpu = b_batch.cpu().unsqueeze(-1)
                
                try:
                    L_cpu = torch.linalg.cholesky(A_cpu)
                    x_cpu = torch.cholesky_solve(b_cpu, L_cpu).squeeze(-1)
                except RuntimeError:
                    x_cpu = torch.linalg.solve(A_cpu, b_cpu).squeeze(-1)

                x_t = x_cpu.to(device)

            # Zero out users with no interactions
            x_t.masked_fill_(zero_mask.unsqueeze(-1), 0.0)

            Target[batch_start:batch_end] = x_t

    # --------------------------------------------------------
    # training
    # --------------------------------------------------------

    def fit(self, data_loader):

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

        self.user_embedding = self.user_embedding.to(self.device)
        self.item_embedding = self.item_embedding.to(self.device)

        U = self.user_embedding.weight.data
        V = self.item_embedding.weight.data
        
        # Move pointers and indices to device for GPU scatter logic
        X_indptr = torch.from_numpy(X.indptr).long().to(self.device)
        X_indices = torch.from_numpy(X.indices).long().to(self.device)
        
        Xt_indptr = torch.from_numpy(Xt.indptr).long().to(self.device)
        Xt_indices = torch.from_numpy(Xt.indices).long().to(self.device)

        start_time = time.time()

        pbar = tqdm(range(self.max_iter), desc="[iALS]")

        for it in pbar:
            t0 = time.time()

            VtV = V.t() @ V
            self._update_batched(U, V, VtV, X_indptr, X_indices)

            UtU = U.t() @ U
            self._update_batched(V, U, UtU, Xt_indptr, Xt_indices)

            pbar.set_postfix(
                t=f"{time.time()-t0:.2f}s"
            )

        print("training time:", time.time() - start_time)


    # --------------------------------------------------------
    # inference
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