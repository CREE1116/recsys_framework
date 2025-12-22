import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel

class DICE(BaseModel):
    """
    DICE: Disentangling User Interest and Conformity (WWW '21)
    - Optimized: Vectorized Sampling & Epoch-based Curriculum
    """
    def __init__(self, config, data_loader):
        super(DICE, self).__init__(config, data_loader)
        
        self.n_users = self.data_loader.n_users
        self.n_items = self.data_loader.n_items
        self.embedding_dim = self.config['model'].get('embedding_dim', 64)
        
        # Hyperparameters
        self.dis_pen = float(self.config['model'].get('dis_pen', 0.02))  # Orthogonality penalty
        self.int_weight = float(self.config['model'].get('int_weight', 0.1))  # Interest weight (final)
        self.pop_weight = float(self.config['model'].get('pop_weight', 0.1))  # Conformity weight (final)
        
        # Curriculum Learning Settings
        self.current_epoch = 0
        self.warmup_epochs = 20  # Stage 1 Duration (Conformity Only)
        
        # Embeddings (4 Types: User/Item x Interest/Conformity)
        self.user_int_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.user_con_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_int_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        self.item_con_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        
        self._init_weights()
        
        # Pre-compute Popularity Ranks for O(1) Sampling
        self._init_popularity()

    def _init_weights(self):
        for emb in [self.user_int_embedding, self.user_con_embedding,
                    self.item_int_embedding, self.item_con_embedding]:
            nn.init.xavier_uniform_(emb.weight)

    def _init_popularity(self):
        """
        Pre-compute item popularity ranks to enable O(1) sampling.
        We sort items by popularity so we can sample by index range.
        """
        # 1. Count interactions
        item_counts = torch.zeros(self.n_items)
        for user_id in range(self.n_users):
            items = self.data_loader.user_history.get(user_id, [])
            for item in items:
                item_counts[item] += 1
        
        self.item_popularity = item_counts  # Raw counts (for debugging if needed)
        
        # 2. Sort items by popularity (Ascending)
        # sorted_items[0] = Least Popular, sorted_items[-1] = Most Popular
        sorted_indices = torch.argsort(item_counts)
        
        self.register_buffer('sorted_items', sorted_indices)
        
        # 3. Create Rank Map: item_id -> rank (0 to N-1)
        # item_rank[item_id] gives its rank in sorted list
        rank_map = torch.zeros(self.n_items, dtype=torch.long)
        rank_map[sorted_indices] = torch.arange(self.n_items)
        
        self.register_buffer('item_rank', rank_map)

    def on_epoch_start(self, epoch):
        """Update current epoch for curriculum scheduling"""
        self.current_epoch = epoch
        
        if epoch == self.warmup_epochs:
            print(f"[DICE] Curriculum Update: Entering Stage 2 (Joint Training) at Epoch {epoch}")

    def get_stage_weights(self):
        """
        Stage 1 (Warm-up): Only learn Conformity to capture bias first.
        Stage 2 (Disentangle): Learn Interest by subtracting Conformity.
        """
        if self.current_epoch < self.warmup_epochs:
            # Stage 1: Conformity Only
            return {'conformity': 1.0, 'interest': 0.0}
        else:
            # Stage 2: Joint Learning (Standard DICE weights)
            return {'conformity': self.pop_weight, 'interest': self.int_weight}

    def sample_cause_specific_negatives(self, pos_items):
        """
        Vectorized Sampling using Pre-computed Ranks (O(1) Complexity)
        """
        batch_size = pos_items.size(0)
        device = pos_items.device
        
        # Get ranks of positive items
        pos_ranks = self.item_rank[pos_items]  # (B,)
        
        # --- O1 (Conformity): Sample Less Popular (Rank < pos_rank) ---
        # Generate random indices in [0, pos_rank)
        # If pos_rank is 0 (least popular), we clamp to avoid error, then handle later
        valid_pop_range = pos_ranks.float()
        rand_pop = (torch.rand(batch_size, device=device) * valid_pop_range).long()
        neg_pop = self.sorted_items[rand_pop]
        
        # Fallback for rank 0: Sample random item
        mask_zero = (pos_ranks == 0)
        if mask_zero.any():
            random_negs = torch.randint(self.n_items, (mask_zero.sum(),), device=device)
            neg_pop[mask_zero] = random_negs

        # --- O2 (Interest): Sample More Popular (Rank > pos_rank) ---
        # Generate random indices in (pos_rank, N) -> [pos_rank + 1, N)
        # Range length = N - 1 - pos_rank
        valid_unpop_range = (self.n_items - 1 - pos_ranks).float()
        rand_unpop = (torch.rand(batch_size, device=device) * valid_unpop_range).long()
        neg_unpop_indices = pos_ranks + 1 + rand_unpop
        
        # Clamp to avoid out of bounds (for rank N-1)
        neg_unpop_indices = neg_unpop_indices.clamp(max=self.n_items - 1)
        neg_unpop = self.sorted_items[neg_unpop_indices]

        # Fallback for rank N-1 (Most popular): Sample random item
        mask_last = (pos_ranks == self.n_items - 1)
        if mask_last.any():
            random_negs = torch.randint(self.n_items, (mask_last.sum(),), device=device)
            neg_unpop[mask_last] = random_negs
        
        return neg_pop, neg_unpop

    def forward(self, users, items=None, component='both'):
        """Calculate scores based on component"""
        u_int = self.user_int_embedding(users)
        u_con = self.user_con_embedding(users)
        
        if items is None:
            # Full evaluation
            i_int = self.item_int_embedding.weight
            i_con = self.item_con_embedding.weight
            score_int = torch.matmul(u_int, i_int.T)
            score_con = torch.matmul(u_con, i_con.T)
        else:
            # Pairwise training
            i_int = self.item_int_embedding(items)
            i_con = self.item_con_embedding(items)
            score_int = (u_int * i_int).sum(dim=-1)
            score_con = (u_con * i_con).sum(dim=-1)
        
        if component == 'interest':
            return score_int
        elif component == 'conformity':
            return score_con
        else:
            return score_int + score_con

    def calc_loss(self, batch_data):
        users = batch_data['user_id'].squeeze(-1)
        pos_items = batch_data['pos_item_id'].squeeze(-1)
        
        # 1. Efficient Sampling
        neg_pop, neg_unpop = self.sample_cause_specific_negatives(pos_items)
        
        # 2. Get Curriculum Weights
        weights = self.get_stage_weights()
        
        total_loss = 0.0
        loss_con_val = 0.0
        loss_int_val = 0.0
        
        # 3. Conformity Loss (O1)
        # Learn that "Pos" is more popular than "Neg_Pop"
        if weights['conformity'] > 0:
            pos_con_score = self.forward(users, pos_items, component='conformity')
            neg_con_score = self.forward(users, neg_pop, component='conformity')
            loss_con = -torch.log(torch.sigmoid(pos_con_score - neg_con_score) + 1e-10).mean()
            total_loss += weights['conformity'] * loss_con
            loss_con_val = loss_con.item()

        # 4. Interest Loss (O2)
        # Learn that "Pos" is more interesting than "Neg_Unpop" (even if Neg is popular)
        if weights['interest'] > 0:
            pos_int_score = self.forward(users, pos_items, component='interest')
            neg_int_score = self.forward(users, neg_unpop, component='interest')
            loss_int = -torch.log(torch.sigmoid(pos_int_score - neg_int_score) + 1e-10).mean()
            total_loss += weights['interest'] * loss_int
            loss_int_val = loss_int.item()

        # 5. Discrepancy Loss (Force Independence)
        # Minimize correlation between Interest and Conformity vectors
        u_int_norm = F.normalize(self.user_int_embedding(users), dim=-1)
        u_con_norm = F.normalize(self.user_con_embedding(users), dim=-1)
        i_int_norm = F.normalize(self.item_int_embedding(pos_items), dim=-1)
        i_con_norm = F.normalize(self.item_con_embedding(pos_items), dim=-1)
        
        loss_dis = (u_int_norm * u_con_norm).sum(dim=-1).abs().mean() + \
                   (i_int_norm * i_con_norm).sum(dim=-1).abs().mean()
        
        total_loss += self.dis_pen * loss_dis
        
        params_to_log = {
            'loss_int': loss_int_val,
            'loss_con': loss_con_val,
            'loss_dis': loss_dis.item(),
            'stage': 2 if self.current_epoch >= self.warmup_epochs else 1
        }
        
        return (total_loss,), params_to_log

    def predict_for_pairs(self, user_ids, item_ids):
        """
        Inference Strategy:
        Ideally, use Interest only. If performance is bad, try adding Conformity with small weight.
        Here we follow the paper: Interest Only.
        """
        return self.forward(user_ids, item_ids, component='interest')

    def get_final_item_embeddings(self):
        return self.item_int_embedding.weight