import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel
from ...loss import BPRLoss, MSELoss, SampledSoftmaxLoss

class CSAR_Minimal(BaseModel):
    """
    CSAR_Minimal (v4.1) - The True Minimal Edition
    Uses raw embeddings directly with a data-driven Ridge G matrix.
    No membership layers, no learned propagation matrix, no alignment loss.
    
    Architecture:
    Score = (E_u @ G_ema) @ E_i^T
    
    - G: Interaction-Projected Ridge updated via EMA (Purely data-driven).
    - Inference: Symmetric + L2 Normalized G_ema.
    - Learning: Only embeddings (E_u, E_i) are trained via BPR.
    """
    def __init__(self, config, data_loader):
        super(CSAR_Minimal, self).__init__(config, data_loader)
        
        self.embedding_dim = self.config['model']['embedding_dim']
        self.ema_momentum = self.config['model'].get('ema_momentum', 0.9)
        self.reg_lambda = self.config['model'].get('reg_lambda', 10.0)
        
        self.n_users = self.data_loader.n_users
        self.n_items = self.data_loader.n_items

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        
        self._init_weights()
        
        # Identity-initialized Kernel
        self.register_buffer('_cached_G', torch.eye(self.embedding_dim))
        
        # Spectral stats for logging
        self.last_g_stats = {}
        
        self.loss_fn = BPRLoss()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def _prepare_interaction_matrix(self):
        """Build symmetrically normalized interaction matrix as a Sparse Tensor (stays Sparse)"""
        train_df = self.data_loader.train_df
        rows = torch.tensor(train_df['user_id'].values, dtype=torch.long)
        cols = torch.tensor(train_df['item_id'].values, dtype=torch.long)
        vals = torch.ones(len(train_df), dtype=torch.float32)
        X = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (self.n_users, self.n_items))
        
        X = X.coalesce()
        
        # Symmetrically normalize X: D_u^-0.5 * X * D_i^-0.5
        # Perform all normalization on CPU to avoid device mismatch with sparse indices
        user_deg = torch.sparse.sum(X, dim=1).to_dense()
        item_deg = torch.sparse.sum(X, dim=0).to_dense()

        def inv_sqrt(d):
            res = torch.zeros_like(d)
            mask = d > 0
            res[mask] = torch.pow(d[mask], -0.5)
            return res

        d_u_inv = inv_sqrt(user_deg)
        d_i_inv = inv_sqrt(item_deg)

        indices = X.indices()
        v = X.values()
        v_norm = d_u_inv[indices[0]] * v * d_i_inv[indices[1]]
        
        X_norm = torch.sparse_coo_tensor(indices, v_norm, (self.n_users, self.n_items))
        return X_norm.cpu().coalesce() # Keep on CPU for sparse operations support

    def on_epoch_start(self, epoch):
        """G를 Ridge-regularized inverse Gramian으로 업데이트 (centering + 스케일 안정화 강화)"""
        with torch.no_grad():
            # GPU에서 그대로 계산 (대부분의 경우 메모리/속도 이득)
            device = self.user_embedding.weight.device
            u_emb = self.user_embedding.weight      # (n_users, dim)
            i_emb = self.item_embedding.weight      # (n_items, dim)

            total_count = self.n_users + self.n_items
            mu = (u_emb.sum(0) + i_emb.sum(0)) / total_count

            # 올바른 centering
            Eu_c = u_emb - mu
            Ei_c = i_emb - mu
            EtE_centered = Eu_c.t() @ Eu_c + Ei_c.t() @ Ei_c

            # Ridge regularization
            I = torch.eye(self.embedding_dim, device=device)
            A = EtE_centered + self.reg_lambda * I

            # 안정적인 inverse 계산 (solve 실패 시 pinv fallback)
            try:
                new_G = torch.linalg.solve(A, I)
            except RuntimeError as e:
                print(f"[CSAR_Minimal] solve 실패 ({e}), pinv로 fallback")
                new_G = torch.linalg.pinv(A)

            # EMA 업데이트
            m = self.ema_momentum
            updated_G = m * self._cached_G + (1 - m) * new_G

            # 1. Symmetric normalization (row/col degree-like)
            row_sums = updated_G.abs().sum(dim=1) + 1e-8
            d_inv_sqrt = torch.pow(row_sums, -0.5)
            G_sym = d_inv_sqrt.unsqueeze(1) * updated_G * d_inv_sqrt.unsqueeze(0)

            # 최종 저장: 대칭 정규화가 적용된 G_sym을 저장
            self._cached_G.copy_(G_sym)
            
            # [Spectral Analysis for Logging]
            try:
                # MPS does not support eigvalsh yet, move to CPU
                G_cpu = G_sym.detach().cpu()
                
                # Eigenvalues (G is symmetric since it's an inverse of a symmetric matrix)
                evs = torch.linalg.eigvalsh(G_cpu)
                evs_abs = evs.abs()
                
                max_ev = evs_abs.max().item()
                min_ev = evs_abs.min().item()
                
                # Orthogonality: ||G G^T - I||_F / dim
                # Identity-like property check
                G_GT = G_cpu @ G_cpu.t()
                orth_err = torch.norm(G_GT - torch.eye(self.embedding_dim), p='fro').item()
                
                # Identity Distance: ||G - I||_F / dim
                # To see if G converges to Identity
                id_dist = torch.norm(G_cpu - torch.eye(self.embedding_dim), p='fro').item()
                
                self.last_g_stats = {
                    "g_ev_max": max_ev,
                    "g_ev_min": min_ev,
                    "g_ev_cond": max_ev / (min_ev + 1e-10),
                    "g_ev_var": evs.var().item(),
                    "g_orth_err": orth_err / self.embedding_dim,
                    "g_id_dist": id_dist / self.embedding_dim
                }
            except Exception as e:
                print(f"[CSAR_Minimal] Spectral analysis failed: {e}")
                self.last_g_stats = {}
    def get_propagation_matrix(self):
        """
        Returns the EMA-updated G (Pre-normalized with L2).
        """
        # Detach to ensure no gradients flow through the data-driven filter
        return self._cached_G.detach()

    def forward(self, users):
        e_u = self.user_embedding(users)
        e_i = self.item_embedding.weight
        
        # Use Normalized G
        G_norm = self.get_propagation_matrix()
        u_p = torch.matmul(e_u, G_norm)
        scores = torch.matmul(u_p, e_i.t())
        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        e_u = self.user_embedding(user_ids)
        e_i = self.item_embedding(item_ids)
        
        # Use Normalized G
        G_norm = self.get_propagation_matrix()
        u_p = torch.matmul(e_u, G_norm)
        scores = (u_p * e_i).sum(dim=-1)
        return scores

    def calc_loss(self, batch_data):
        users = batch_data['user_id']
        pos_items = batch_data['pos_item_id']
        neg_items = batch_data['neg_item_id']
        
        pos_scores = self.predict_for_pairs(users, pos_items)
        neg_scores = self.predict_for_pairs(users, neg_items)
    
        main_loss = self.loss_fn(pos_scores, neg_scores)
        
        return (main_loss,), {
            "g_diag_mean": self._cached_G.diag().mean().item(),
            "g_abs_mean": self._cached_G.abs().mean().item(),
            **self.last_g_stats
        }

    def get_final_item_embeddings(self):
        return self.item_embedding.weight.detach()
