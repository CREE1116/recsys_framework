import torch
import torch.nn as nn
from ..base_model import BaseModel
from src.loss import BPRLoss

class LightGCN(BaseModel):
    """LightGCN: supports both sparse and dense adjacency matrices."""

    def __init__(self, config, data_loader):
        super(LightGCN, self).__init__(config, data_loader)

        self.embedding_dim = self.config['model']['embedding_dim']
        self.n_layers = self.config['model']['n_layers']

        self.n_users = self.data_loader.n_users
        self.n_items = self.data_loader.n_items

        # adj_matrix is used to build norm_adj_matrix; we don't need to keep it as a buffer
        adj_matrix = self.data_loader.get_interaction_graph().to(self.device).float()
        self.norm_adj_matrix = self._get_normalized_adj_matrix(adj_matrix)

        # Small graphs: convert sparse to dense for faster GPU matmul
        total_nodes = self.n_users + self.n_items
        if total_nodes < 15000:
            self._log(f"Node count {total_nodes} < 15000 — converting to dense for GPU matmul.")
            self.norm_adj_matrix = self.norm_adj_matrix.to_dense()

        # Pre-resolve propagation strategy to avoid per-iteration checks in forward.
        # MPS does not reliably support sparse mm; keep sparse matrix on CPU and move
        # results back to device after each propagation step.
        # Use regular attribute instead of register_buffer to prevent model.to(device) from moving it.
        if self.norm_adj_matrix.is_sparse and self.device.type == 'mps':
            self.norm_adj_matrix = self.norm_adj_matrix.cpu()
            self._adj_device = torch.device('cpu')
            self._sparse_prop = True
        elif self.norm_adj_matrix.is_sparse:
            self.norm_adj_matrix = self.norm_adj_matrix.to(self.device)
            self._adj_device = self.device
            self._sparse_prop = True
        else:
            self.norm_adj_matrix = self.norm_adj_matrix.to(self.device)
            self._adj_device = self.device
            self._sparse_prop = False

        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        
        self.loss_fn = BPRLoss()
        self._init_weights()
        
        self._final_user_emb = None
        self._final_item_emb = None

    def _get_normalized_adj_matrix(self, A):
        """Compute symmetric normalized adjacency D^{-0.5} A D^{-0.5}."""
        if A.is_sparse:
            A = A.coalesce()
            row_sum = torch.sparse.sum(A, dim=1).to_dense()
            D_inv_sqrt = torch.pow(row_sum, -0.5)
            D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.

            indices = A.indices()
            values = A.values() * D_inv_sqrt[indices[0]] * D_inv_sqrt[indices[1]]
            return torch.sparse_coo_tensor(indices, values, A.shape).float()
        else:
            D = A.sum(1)
            D_inv_sqrt = torch.pow(D, -0.5)
            D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.
            # D^{-0.5} A D^{-0.5} via broadcasting: (N,1) * (N,N) * (1,N)
            return (A * D_inv_sqrt.unsqueeze(1) * D_inv_sqrt.unsqueeze(0)).float()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self._final_user_emb, self._final_item_emb = None, None

    def eval(self):
        super().eval()
        self._final_user_emb, self._final_item_emb = None, None

    def _propagate_embeddings(self):
        if not self.training and self._final_user_emb is not None and self._final_item_emb is not None:
            return self._final_user_emb, self._final_item_emb

        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embeddings_list = [all_embeddings]

        for _ in range(self.n_layers):
            if self._sparse_prop:
                # Force float32 for sparse mm as CUDA doesn't support float16 for this operation
                # Disable autocast to prevent inputs from being cast back to Half
                with torch.amp.autocast(device_type=self.device.type, enabled=False):
                    all_embeddings = torch.sparse.mm(
                        self.norm_adj_matrix.float(), 
                        all_embeddings.to(self._adj_device).float()
                    ).to(self.device)
            else:
                all_embeddings = torch.matmul(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        
        final_embeddings = torch.mean(torch.stack(embeddings_list, dim=0), dim=0)
        final_user_emb, final_item_emb = torch.split(final_embeddings, [self.n_users, self.n_items])

        if not self.training:
            self._final_user_emb = final_user_emb
            self._final_item_emb = final_item_emb

        return final_user_emb, final_item_emb

    def get_embeddings(self):
        return self._propagate_embeddings()

    def get_final_item_embeddings(self):
        _, item_embeddings = self.get_embeddings()
        return item_embeddings.detach()

    def calc_loss(self, batch_data):
        users = batch_data['user_id']
        pos_items = batch_data['pos_item_id']
        neg_items = batch_data['neg_item_id']

        user_embeds, item_embeds = self.get_embeddings()
        
        user_vec = user_embeds[users]
        pos_item_vec = item_embeds[pos_items]
        neg_item_vec = item_embeds[neg_items]

        pos_scores = (user_vec * pos_item_vec).sum(dim=-1)
        neg_scores = (user_vec * neg_item_vec).sum(dim=-1)

        loss = self.loss_fn(pos_scores, neg_scores)
        
        # L2 regularization on base embeddings (not propagated)
        u_base_emb = self.user_embedding(users)
        p_base_emb = self.item_embedding(pos_items)
        n_base_emb = self.item_embedding(neg_items)
        l2_loss = self.get_l2_reg_loss(u_base_emb, p_base_emb, n_base_emb)

        return (loss, l2_loss), {'loss_main': loss.item(), 'loss_l2': l2_loss.item()}

    def forward(self, users):
        user_embeds, item_embeds = self.get_embeddings()
        user_vecs = user_embeds[users]
        scores = torch.matmul(user_vecs, item_embeds.t())
        return scores
    
    def predict_for_pairs(self, users, items):
        user_embeds, item_embeds = self.get_embeddings()
        
        user_vec = user_embeds[users] # [B, 1, D] or [B, D]
        item_vec = item_embeds[items] # [B, 1, D] or [B, N, D]

        return (user_vec * item_vec).sum(dim=-1)

    def __str__(self):
        return f"LightGCN(embedding_dim={self.embedding_dim}, n_layers={self.n_layers})"