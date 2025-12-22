import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel
from src.loss import BPRLoss, MSELoss, InfoNCELoss
class MMR(BaseModel):
    """
    MMR: Maximal Marginal Relevance
    - Greedy re-ranking: relevance - λ × similarity_to_selected
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        
        self.embedding_dim = int(config['model']['embedding_dim'])
        self.lambda_mmr = float(config['model'].get('lambda_mmr', 0.5))
        
        self.n_users = self.data_loader.n_users
        self.n_items = self.data_loader.n_items
        
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
    
    def forward(self, users, top_k=10):
        u_emb = self.user_embedding(users)
        i_emb_all = self.item_embedding.weight
        
        # Initial relevance scores
        relevance = torch.matmul(u_emb, i_emb_all.T)  # [B, M]
        
        # MMR re-ranking
        diverse_items = self.mmr_rerank(relevance, i_emb_all, top_k)
        return diverse_items
    
    def mmr_rerank(self, relevance, item_embeddings, k=10, lambda_mmr=None, candidate_k=100):
        """
        Fully vectorized + optimized MMR for speed
        """
        if lambda_mmr is None:
            lambda_mmr = self.lambda_mmr
            
        batch_size, n_items = relevance.shape
        device = relevance.device
        
        # 1. Top candidate_k (e.g. 100)
        candidate_k = min(n_items, candidate_k)
        
        # [B, C]
        topk_scores, topk_indices = torch.topk(relevance, k=candidate_k, dim=1)
        
        # 2. Gather candidate embeddings
        # [B, C, D]
        cand_embs = item_embeddings[topk_indices]
        # Normalize
        cand_embs = F.normalize(cand_embs, p=2, dim=2)
        
        # 3. Selected items storage
        # indices relative to candidates (0..C-1)
        selected_rel_indices = torch.zeros(batch_size, k, dtype=torch.long, device=device)
        
        # Mask for candidates (0: available, 1: selected)
        # [B, C]
        is_selected = torch.zeros(batch_size, candidate_k, dtype=torch.bool, device=device)
        
        # 4. Compute similarity matrix among candidates
        # [B, C, C]
        # sim[b, i, j] = sim(cand[i], cand[j])
        sim_matrix = torch.bmm(cand_embs, cand_embs.transpose(1, 2))
        
        # 5. First item: Highest relevance (greedy standard)
        # topk_scores is sorted? torch.topk returns sorted. So index 0 is best.
        # But MMR should pick based on relevance first anyway.
        first_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        selected_rel_indices[:, 0] = first_idx
        is_selected[:, 0] = True
        
        # Track Max Similarity to ANY selected item
        # Initial: sim to first item
        # current_max_sim [B, C]
        # sim_matrix[:, 0, :] -> sim row for 0th item (which is first_idx)
        # Actually first_idx is user-dependent? No, topk sorts it, so 0 is always best score.
        current_max_sim = sim_matrix[:, 0, :] 
        
        # 6. Iterative Selection (Vectorized over Batch)
        batch_indices = torch.arange(batch_size, device=device)
        
        for step in range(1, k):
            # MMR Score = lambda * relevance - (1 - lambda) * max_sim
            # relevance: topk_scores [B, C]
            # max_sim: current_max_sim [B, C]
            
            # Weighted score
            mmr_score = lambda_mmr * topk_scores - (1.0 - lambda_mmr) * current_max_sim
            
            # Mask selected items
            mmr_score[is_selected] = -float('inf')
            
            # Select best
            best_idx = torch.argmax(mmr_score, dim=1) # [B]
            selected_rel_indices[:, step] = best_idx
            
            # Update mask
            is_selected[batch_indices, best_idx] = True
            
            # Update max_sim
            # New selected item 'best_idx'
            # Sim of all candidates to this new item:
            # sim_matrix[b, best_idx[b], :]
            # We need to gather this row.
            
            # sim_matrix: [B, C, C]
            # best_idx: [B] -> expand to [B, 1, C]? No.
            # We want [B, C] output.
            # sim_matrix[b, best_idx[b]] works?
            # PyTorch advanced indexing:
            new_sims = sim_matrix[batch_indices, best_idx, :] # [B, C]
            
            # Update max
            current_max_sim = torch.max(current_max_sim, new_sims)
            
        # 7. Map back to global indices and Assign Ranks
        final_scores = torch.full_like(relevance, -float('inf'))
        
        # [B, K]
        global_selected_indices = torch.gather(topk_indices, 1, selected_rel_indices)
        
        # Assign rank scores
        ranks = torch.arange(k, 0, -1, device=device).float() + 1000.0
        # Expand ranks to [B, K]
        ranks = ranks.unsqueeze(0).expand(batch_size, -1)
        
        final_scores.scatter_(1, global_selected_indices, ranks)
        
        return final_scores
    
    def calc_loss(self, batch_data):
        # Standard MF loss (MMR은 inference에만 적용)
        users = batch_data['user_id'].squeeze(-1)
        pos_items = batch_data['pos_item_id'].squeeze(-1)
        neg_items = batch_data['neg_item_id'].squeeze(-1)
        
        u_emb = self.user_embedding(users)
        pos_emb = self.item_embedding(pos_items)
        neg_emb = self.item_embedding(neg_items)
        
        pos_score = (u_emb * pos_emb).sum(dim=-1)
        neg_score = (u_emb * neg_emb).sum(dim=-1)
        
        bpr_loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10).mean()
        
        return (bpr_loss,), {'bpr': bpr_loss.item()}

    def predict_for_pairs(self, user_ids, item_ids):
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)
        return (u_emb * i_emb).sum(dim=-1)

    def get_final_item_embeddings(self):
        return self.item_embedding.weight.detach()