import torch
import torch.nn as nn
import torch.nn.functional as F
from .lightgcn import LightGCN
from src.loss import BPRLoss

class SimGCL(LightGCN):
    """
    SimGCL: Simple Graph Contrastive Learning for Recommendation (SIGIR'22)
    Inherits from LightGCN and adds detailed contrastive learning with noise perturbation.
    """
    def __init__(self, config, data_loader):
        super(SimGCL, self).__init__(config, data_loader)
        
        self.cl_rate = self.config['model'].get('cl_rate', 0.2)
        self.eps = self.config['model'].get('eps', 0.1)
        self.tau = self.config['model'].get('tau', 0.2)
        
    def _propagate_embeddings(self, perturbed=False):
        """
        Propagate embeddings with optional perturbation.
        NOTE: SimGCL (SELFRec implementation) excludes the 0-th layer (initial) embedding 
        from the final sum to ensure contrastive views are sufficiently distinct.
        """
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embeddings_list = [] # Start empty, unlike LightGCN which includes E0

        for _ in range(self.n_layers):
            # 1. Graph Aggregation
            if self.norm_adj_matrix.is_sparse:
                if self.device.type == 'mps':
                    # MPS does not support torch.sparse.mm; fall back to dense matmul
                    all_embeddings = torch.mm(self.norm_adj_matrix.to_dense(), all_embeddings)
                else:
                    all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            else:
                all_embeddings = torch.matmul(self.norm_adj_matrix, all_embeddings)
            
            # 2. Perturbation (SimGCL specific)
            if perturbed:
                # noise = sign(emb) * normalize(random_noise) * eps
                # Use randn (Gaussian) for spherical uniformity as suggested
                random_noise = torch.randn_like(all_embeddings).to(self.device)
                
                # F.normalize default dim=1 (across features), which is correct
                all_embeddings += torch.sign(all_embeddings) * F.normalize(random_noise, dim=1) * self.eps
                
            embeddings_list.append(all_embeddings)
        
        # Stack and Mean
        final_embeddings = torch.mean(torch.stack(embeddings_list, dim=0), dim=0)
        final_user_emb, final_item_emb = torch.split(final_embeddings, [self.n_users, self.n_items])

        return final_user_emb, final_item_emb
        
    def calc_loss(self, batch_data):
        users = batch_data['user_id']
        pos_items = batch_data['pos_item_id']
        neg_items = batch_data['neg_item_id']

        # 1. Main View (BPR Loss)
        user_emb, item_emb = self._propagate_embeddings(perturbed=False)
        
        u_e = user_emb[users]
        pos_e = item_emb[pos_items]
        neg_e = item_emb[neg_items]
        
        pos_scores = torch.sum(u_e * pos_e, dim=1)
        neg_scores = torch.sum(u_e * neg_e, dim=1)
        
        rec_loss = self.loss_fn(pos_scores, neg_scores)
        
        # 2. InfoNCE Loss (CL Loss)
        # We need unique users and items in the batch to compute CL loss
        unique_users = torch.unique(users)
        unique_pos_items = torch.unique(pos_items)
        
        # View 1
        user_view_1, item_view_1 = self._propagate_embeddings(perturbed=True)
        # View 2
        user_view_2, item_view_2 = self._propagate_embeddings(perturbed=True)
        
        # Calculate loss on unique batch elements
        user_cl_loss = self.info_nce_loss(user_view_1[unique_users], user_view_2[unique_users])
        item_cl_loss = self.info_nce_loss(item_view_1[unique_pos_items], item_view_2[unique_pos_items])
        
        cl_loss = self.cl_rate * (user_cl_loss + item_cl_loss)
        
        # 3. L2 Regularization (On base embeddings)
        l2_loss = self.get_l2_reg_loss(
            self.user_embedding(users), 
            self.item_embedding(pos_items), 
            self.item_embedding(neg_items)
        )
        
        total_loss = rec_loss + cl_loss + l2_loss
        
        # Return tensors specifically for Trainer logging
        return (total_loss, rec_loss, cl_loss, l2_loss), {
            'loss_main': rec_loss.item(),
            'loss_cl': cl_loss.item(),
            'loss_l2': l2_loss.item()
        }

    def info_nce_loss(self, view1, view2):
        """
        Calculates InfoNCE loss: -log( exp(sim(v1, v2)/tau) / sum(exp(sim(v1, vk)/tau)) )
        """
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)
        
        # Similarity matrix: (Batch, Batch)
        numerator = torch.matmul(view1, view2.t()) / self.tau
        
        # Positive pairs are on the diagonal
        targets = torch.arange(numerator.size(0)).to(self.device)
        
        return F.cross_entropy(numerator, targets)
        
    def __str__(self):
        return f"SimGCL(dim={self.embedding_dim}, layers={self.n_layers}, eps={self.eps}, cl={self.cl_rate})"
