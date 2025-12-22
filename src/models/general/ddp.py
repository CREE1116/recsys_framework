import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel
from src.loss import BPRLoss, MSELoss, InfoNCELoss
class DPP(BaseModel):
    """
    DPP: Determinantal Point Process for Diverse Recommendations
    - Quality와 Diversity를 동시에 최적화
    - Kernel matrix로 아이템 간 유사도 측정
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        
        self.embedding_dim = int(config['model']['embedding_dim'])
        self.diversity_weight = float(config['model'].get('diversity_weight', 0.5))
        
        self.n_users = self.data_loader.n_users
        self.n_items = self.data_loader.n_items
        
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
    
    def forward(self, users, top_k=10):
        u_emb = self.user_embedding(users)
        i_emb_all = self.item_embedding.weight
        
        # Relevance scores
        scores = torch.matmul(u_emb, i_emb_all.T)  # [B, M]
        
        # DPP sampling for diverse top-k
        diverse_items = self.dpp_sample(scores, i_emb_all, top_k)
        return diverse_items
    
    def dpp_sample(self, scores, item_embeddings, k):
        """
        DPP utilizing vectorized operations for batch construction.
        Note: The greedy selection loop over K is sequential, but we vectorize over Batch.
        In each step K, we evaluate ALL C candidates for ALL B users in parallel (or batched loops).
        """
        batch_size = scores.size(0)
        n_items = scores.size(1)
        device = scores.device
        
        # 1. Select top-C candidates (e.g., 50 or 100 is usually enough for diversity re-ranking)
        # Using 100 to keep memory low while ensuring quality
        candidate_k = min(n_items, 100) 
        
        # [B, C]
        top_vals, top_inds = torch.topk(scores, k=candidate_k, dim=1)
        
        # 2. Construct Kernel Matrices for Candidates
        # L [B, C, C]
        # First, computing global sim [M, M] is too big if M is large, but acceptable for 1349.
        # But we only need [B, C, C] submatrices.
        # We can gather: global_sim[top_inds] -> [B, C, M] -> ... [B, C, C].
        
        # Global sim [M, M]
        global_sim = F.cosine_similarity(
            item_embeddings.unsqueeze(0), 
            item_embeddings.unsqueeze(1), 
            dim=2
        )
        
        # Gather sim submatrices for each user
        # indices: top_inds [B, C]
        # index1 [B, C, 1]: top_inds.unsqueeze(2)
        # index2 [B, 1, C]: top_inds.unsqueeze(1)
        # We need to extract sim[top_inds[b][i], top_inds[b][j]] from global_sim
        
        # Using intelligent indexing:
        # global_sim[top_inds.unsqueeze(2), top_inds.unsqueeze(1)] -> [B, C, C] -- No, this broadcasts.
        # We need a loop or advanced gather
        sim_batch = torch.zeros(batch_size, candidate_k, candidate_k, device=device)
        # Vectorized gather is safer via loop over B if M is small, but we want to avoid B loop.
        # However, advanced indexing with expanded dimensions works:
        # matrix[ind_row[:, :, None], ind_col[:, None, :]]
        # global_sim is [M, M].
        # We want result[b, i, j] = global_sim[inds[b, i], inds[b, j]]
        # This is not directly vectorized without 'vmap'.
        # But B=1024, C=100. 1024 calls of index_select is fast?
        # A simple trick:
        # L = global_sim[top_inds[:, :, None], top_inds[:, None, :]] works if global_sim is used as lookup
        # But global_sim is 2D. 
        # Correct way: global_sim[index_tensor] where index_tensor is [B, C, C] flatten? No.
        
        # Let's rely on a loop for kernel construction (GPU is fast at indexing) or optimized gather.
        # Actually, reconstructing sim on the fly for [B, C] items:
        # emb_cand [B, C, D] = item_embeddings[top_inds]
        # sim_batch = bmm(emb_cand, emb_cand.T) -> [B, C, C].
        # THIS IS MUCH FASTER! (No global 1349x1349 construction, just local 100x100)
        
        # [B, C, D]
        cand_embs = item_embeddings[top_inds]
        # Normalize
        cand_embs = F.normalize(cand_embs, p=2, dim=2)
        # [B, C, C]
        sim_batch = torch.bmm(cand_embs, cand_embs.transpose(1, 2))
        
        # Quality [B, C]
        quality = torch.sigmoid(top_vals)
        sqrt_quality = quality.sqrt().unsqueeze(2) # [B, C, 1]
        
        # L = sim * outer(sqrt_q, sqrt_q)
        # [B, C, C]
        L_batch = sim_batch * torch.matmul(sqrt_quality, sqrt_quality.transpose(1, 2))
        
        # 3. Batched Greedy MAP
        # Returns indices relative to 0..C-1
        # selected_local_indices: [B, K]
        selected_local_indices = self.batched_greedy_map(L_batch, k)
        
        # 4. Map back to global scores
        final_scores = torch.full_like(scores, -float('inf'))
        
        # [B, K]
        selected_global_indices = torch.gather(top_inds, 1, selected_local_indices)
        
        # Assign scores
        # We can't vectorize assignment easily to sparse locations in [B, M] without scatter.
        # [B, K] -> values [1000, 999...]
        ranks = torch.arange(k, 0, -1, device=device).float() + 1000.0 # [K]
        ranks = ranks.unsqueeze(0).expand(batch_size, -1) # [B, K]
        
        final_scores.scatter_(1, selected_global_indices, ranks)
        
        return final_scores

    def batched_greedy_map(self, L_batch, k):
        """
        Vectorized Greedy MAP for DPP.
        L_batch: [B, C, C]
        Returns: selected_indices [B, k] (indices in 0..C-1)
        """
        batch_size, n_candidates, _ = L_batch.size()
        device = L_batch.device
        
        selected_indices = torch.zeros(batch_size, k, dtype=torch.long, device=device)
        
        # Mask for remaining items (1=available, 0=selected)
        # [B, C]
        remaining_mask = torch.ones(batch_size, n_candidates, dtype=torch.bool, device=device)
        
        # Current selected per user (for constructing submatrices)
        # We iteratively build the selected list
        # List of [B, 1] tensors
        selected_history = [] 
        
        # Epsilon eye for numerical stability
        eye_eps = 1e-6
        
        for step in range(k):
            # We want to find candidate 'j' maximizing logdet(L_{S+j})
            # S is current selected set (size 'step')
            # If step == 0: Just pick max diagonal
            
            if step == 0:
                # [B, C]
                diags = torch.diagonal(L_batch, dim1=1, dim2=2)
                # Pick max
                # [B]
                best_vals, best_cols = torch.max(diags, dim=1)
                
                selected_indices[:, step] = best_cols
                selected_history.append(best_cols.unsqueeze(1))
                
                # Update mask
                # scatter 0
                remaining_mask.scatter_(1, best_cols.unsqueeze(1), 0)
                
            else:
                # Optimized marginal gain check?
                # Marginal gain of adding j to S:
                # gain(j) = L_jj - L_{j,S} * inv(L_{S,S}) * L_{S,j} (Schur complement)
                # This avoids computing full logdet for every candidate.
                # We need L_{S,S} inverse.
                
                # S indices: selected_history (concatenated) -> [B, step]
                # Gather L_{S,S} [B, step, step]
                current_sel = torch.cat(selected_history, dim=1) # [B, step]
                
                # Gather L_SS
                # We need to gather submatrices for each batch.
                # L_batch: [B, C, C]
                # indices: current_sel [B, step]
                # expanded indices for gather
                # This is tricky in pure torch without loop over B.
                # But 'step' is small (1..10).
                
                # Alternative: Use loop over candidates 'c' and compute full logdet?
                # Loop C (100) vs Loop B (1024).
                # 100 * small_op vs 1024 * small_op.
                # Schur complement is better if we can vectorize.
                
                # Let's try iterating over candidates C for simplicity and vectorization over B.
                # For each candidate c (0..C-1), if available (mask), compute score.
                
                # To vectorize checking all candidates:
                # We need to construct matrices [B, step+1, step+1] for all C? No, too big memory.
                # Iterate c in 0..C-1.
                
                best_gains = torch.full((batch_size,), -float('inf'), device=device)
                best_cands = torch.zeros(batch_size, dtype=torch.long, device=device)
                
                # Pre-compute inverse of L_SS for Schur?
                # L_SS: [B, step, step]
                # We actually need to gather L_SS properly.
                # Let's gather L_SS once per step.
                # Gather strategy:
                # dim1: current_sel.unsqueeze(2).expand(-1, -1, step)
                # dim2: current_sel.unsqueeze(1).expand(-1, step, -1)
                # This works!
                
                idx1 = current_sel.unsqueeze(2).expand(-1, -1, step)
                idx2 = current_sel.unsqueeze(1).expand(-1, step, -1)
                L_SS = torch.gather(L_batch, 1, idx1) # [B, step, C] ? No gather is 1D index usually?
                # Torch gather is along ONE dim.
                # We need advanced indexing.
                # Since we can't do L_batch[range(B), current_sel, current_sel],
                # We will use a loop over Batch? No, that's what we want to avoid.
                
                # Solution: batch_gather helper
                # Flatten batch: [B*C, C]
                # This is getting complicated.
                
                # FALLBACK STRATEGY: 
                # Since we know B is large and C is relatively small (100), 
                # looping over candidates `c` is acceptable if `c` is handled in parallel for B.
                # BUT, we need `logdet([L_SS, L_Sc; L_cS, L_cc])`.
                # We still need L_SS for each user.
                
                # Let's take the hit and gather L_SS using gather+reshape
                # L_batch_flat = L_batch.view(batch_size, -1) # [B, C*C]
                # indices = ... calculations ...
                # It's cleaner to loop over C candidates.
                
                # For each candidate c:
                # Form [B, step+1, step+1] matrix.
                #   TopLeft: L_SS (Previous selected)
                #   RightCol: L_{S,c}
                #   BottomRow: L_{c,S}
                #   BottomRight: L_{cc}
                # Compute logdet.
                
                # Optimization: We only compute L_SS inverse ONCE per step for all users.
                # Because L_SS is constant for a user across all candidate tests.
                
                # 1. Extract L_SS for all users [B, step, step]
                # How?
                # We can iterate `step` times to gather columns?
                l_ss_rows = []
                for r in range(step):
                    row_idx = current_sel[:, r] # [B]
                    # col_indices = current_sel [B, step]
                    # We want L[b, row_idx[b], col_indices[b]]
                    # L_batch[b, row_idx[b]] -> [C]
                    # Then select col_indices.
                    
                    # L_batch[range(B), row_idx] -> [B, C]
                    r_vec = L_batch[torch.arange(batch_size, device=device), row_idx]
                    # Select cols
                    # r_vec.gather(1, current_sel) -> [B, step]
                    row_vals = r_vec.gather(1, current_sel)
                    l_ss_rows.append(row_vals)
                
                L_SS = torch.stack(l_ss_rows, dim=1) # [B, step, step]
                
                # Compute Inverse + Epsilon
                L_SS_inv = torch.linalg.inv(L_SS + torch.eye(step, device=device).unsqueeze(0) * eye_eps)
                
                # Now loop over candidates to find max gain (Schur)
                # gain = L_cc - L_cS @ L_SS_inv @ L_Sc
                # L_cS: [B, 1, step] -> L_batch[b, c, S]
                # L_Sc: [B, step, 1] -> L_cS.T
                
                # We can do this block-wise? 
                # Check 100 candidates in chunks? Or just loop.
                # 100 loops of matrix vector mult is fast.
                
                for c in range(n_candidates):
                    # Check if masked (vectorized check?)
                    # mask is [B, C].
                    # If c is masked for user b, we skip or set output -inf.
                    
                    # We process all users. Some have c masked.
                    # L_cc: [B] -> L_batch[:, c, c]
                    L_cc = L_batch[:, c, c]
                    
                    # L_cS: [B, 1, step]
                    # row c, cols S
                    # L_batch[:, c] -> [B, C]
                    # gather S
                    L_cS_vec = L_batch[:, c, :].gather(1, current_sel).unsqueeze(1) # [B, 1, step]
                    L_Sc_vec = L_cS_vec.transpose(1, 2) # [B, step, 1]
                    
                    # Schur term: L_cS @ L_SS_inv @ L_Sc
                    # [B, 1, step] @ [B, step, step] @ [B, step, 1] -> [B, 1, 1]
                    term = torch.matmul(torch.matmul(L_cS_vec, L_SS_inv), L_Sc_vec).squeeze(2).squeeze(1)
                    
                    gain = L_cc - term
                    
                    # Apply mask (if c is already selected or invalid)
                    # mask[:, c] is True if available
                    is_available = remaining_mask[:, c]
                    gain = torch.where(is_available, gain, torch.tensor(-float('inf'), device=device))
                    
                    # Update best
                    # For each user, if gain > best, update
                    update_mask = gain > best_gains
                    best_gains = torch.where(update_mask, gain, best_gains)
                    best_cands = torch.where(update_mask, torch.tensor(c, device=device), best_cands)
                
                # End candidate loop
                selected_indices[:, step] = best_cands
                selected_history.append(best_cands.unsqueeze(1))
                # Update mask
                remaining_mask.scatter_(1, best_cands.unsqueeze(1), 0)

        return selected_indices
    
    def calc_loss(self, batch_data):
        # Standard BPR + DPP regularization
        users = batch_data['user_id'].squeeze(-1)
        pos_items = batch_data['pos_item_id'].squeeze(-1)
        neg_items = batch_data['neg_item_id'].squeeze(-1)
        
        u_emb = self.user_embedding(users)
        pos_emb = self.item_embedding(pos_items)
        neg_emb = self.item_embedding(neg_items)
        
        pos_score = (u_emb * pos_emb).sum(dim=-1)
        neg_score = (u_emb * neg_emb).sum(dim=-1)
        
        bpr_loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10).mean()
        
        # DPP regularization: Encourage diverse item embeddings (Sampled)
        # 전체 M x M을 계산하면 메모리 폭발, 배치 내 아이템만 사용
        batch_items = torch.cat([pos_items, neg_items])
        batch_emb = F.normalize(self.item_embedding(batch_items), dim=-1)
        item_sim = torch.matmul(batch_emb, batch_emb.T)
        # Minimize off-diagonal similarity
        mask = ~torch.eye(item_sim.size(0), dtype=torch.bool, device=item_sim.device)
        diversity_loss = item_sim[mask].mean()
        
        return (bpr_loss, self.diversity_weight * diversity_loss), {
            'bpr': bpr_loss.item(),
            'diversity': diversity_loss.item()
        }

    def predict_for_pairs(self, user_ids, item_ids):
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)
        return (u_emb * i_emb).sum(dim=-1)

    def get_final_item_embeddings(self):
        return self.item_embedding.weight.detach()