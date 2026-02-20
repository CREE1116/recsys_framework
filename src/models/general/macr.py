import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel


class MACR(BaseModel):

    def __init__(self, config, data_loader):
        super(MACR, self).__init__(config, data_loader)

        self.n_users = self.data_loader.n_users
        self.n_items = self.data_loader.n_items
        self.embedding_dim = self.config['model'].get('embedding_dim', 64)

        # hyperparameters (논문 권장: c 30~40)
        self.c = self.config['model'].get('c', 30.0)
        self.alpha = self.config['model'].get('alpha', 1e-3)
        self.beta = self.config['model'].get('beta', 1e-3)

        # Main matching branch
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)

        # Item popularity branch
        self.item_pop_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        self.item_pop_mlp = nn.Linear(self.embedding_dim, 1)

        # User activity branch
        self.user_act_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.user_act_mlp = nn.Linear(self.embedding_dim, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.item_pop_embedding.weight)
        nn.init.xavier_uniform_(self.user_act_embedding.weight)

    # -------------------------------------------------
    # Forward
    # -------------------------------------------------

    def forward(self, users, items=None):

        # -------- Inference (Full Ranking) --------
        if items is None:
            u_emb = self.user_embedding(users)                # [B, D]
            all_i_emb = self.item_embedding.weight            # [N, D]

            y_ui = torch.matmul(u_emb, all_i_emb.t())         # [B, N]

            all_i_pop_emb = self.item_pop_embedding.weight    # [N, D]
            y_i = self.item_pop_mlp(all_i_pop_emb).squeeze(-1)  # [N]

            scores = y_ui - self.c * torch.sigmoid(y_i)
            return scores

        # -------- Training --------
        u_emb = self.user_embedding(users)
        i_emb = self.item_embedding(items)

        y_ui = (u_emb * i_emb).sum(dim=-1)                    # logit

        i_pop_emb = self.item_pop_embedding(items)
        y_i = self.item_pop_mlp(i_pop_emb).squeeze(-1)        # logit

        u_act_emb = self.user_act_embedding(users)
        y_u = self.user_act_mlp(u_act_emb).squeeze(-1)        # logit

        return y_ui, y_u, y_i

    # -------------------------------------------------
    # Loss (논문 정식 multiplicative 구조)
    # -------------------------------------------------

    # -------------------------------------------------
    # Loss (BPR Implementation for MACR)
    # -------------------------------------------------

    def calc_loss(self, batch_data):
        user_ids = batch_data['user_id']
        pos_ids = batch_data['pos_item_id']
        neg_ids = batch_data['neg_item_id'] # BPR loader provides this

        # 1. Forward Positives
        u_emb = self.user_embedding(user_ids)
        
        # Pos Items
        i_pos_emb = self.item_embedding(pos_ids)
        y_ui_pos = (u_emb * i_pos_emb).sum(dim=-1)
        
        i_pop_pos_emb = self.item_pop_embedding(pos_ids)
        y_i_pos = self.item_pop_mlp(i_pop_pos_emb).squeeze(-1)

        # Neg Items
        i_neg_emb = self.item_embedding(neg_ids)
        y_ui_neg = (u_emb * i_neg_emb).sum(dim=-1) # Broadcasting might be needed if negs > 1
        
        i_pop_neg_emb = self.item_pop_embedding(neg_ids)
        y_i_neg = self.item_pop_mlp(i_pop_neg_emb).squeeze(-1)

        # User Activity (Shared for pos/neg)
        u_act_emb = self.user_act_embedding(user_ids)
        y_u = self.user_act_mlp(u_act_emb).squeeze(-1)
        
        # 2. Multiplicative Predictions (Sigmoid wrapped)
        # \hat{y}_{ui} = \sigma(y_{ui}) * \sigma(y_i) * \sigma(y_u)
        
        # Sigmoid activations
        s_ui_pos = torch.sigmoid(y_ui_pos)
        s_i_pos = torch.sigmoid(y_i_pos)
        s_u = torch.sigmoid(y_u)
        
        s_ui_neg = torch.sigmoid(y_ui_neg)
        s_i_neg = torch.sigmoid(y_i_neg)
        
        # Combined Scores for Main Loss
        y_hat_pos = s_ui_pos * s_i_pos * s_u
        y_hat_neg = s_ui_neg * s_i_neg * s_u
        
        # 3. Losses
        
        # Main BPR Loss
        loss_main = -torch.mean(torch.log(torch.sigmoid(y_hat_pos - y_hat_neg) + 1e-8))
        
        # Item Bias BPR Loss (Optimize propersity for positive items)
        loss_i = -torch.mean(torch.log(torch.sigmoid(y_i_pos - y_i_neg) + 1e-8))
        
        # User Bias Loss
        # In BPR with (u, i, j), user bias is constant. 
        # Standard MACR BPR implementations often omit L_u or use BCE if non-pairwise.
        # Here we only apply L2 reg to y_u implicitly via optimizer weight decay or just 0.
        loss_u = torch.tensor(0.0, device=user_ids.device)

        log_info = {
            "main_loss": loss_main.item(),
            "item_loss": loss_i.item(),
            "user_loss": loss_u.item()
        }

        # User criticized sharing labels, but BPR inherently handles relative ranking.
        return (loss_main, self.alpha * loss_i, self.beta * loss_u), log_info

    # -------------------------------------------------
    # Pair prediction (Counterfactual inference)
    # -------------------------------------------------

    def predict_for_pairs(self, user_ids, item_ids):

        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)

        y_ui = (u_emb * i_emb).sum(dim=-1)

        i_pop_emb = self.item_pop_embedding(item_ids)
        y_i = self.item_pop_mlp(i_pop_emb).squeeze(-1)

        score = y_ui - self.c * torch.sigmoid(y_i)
        return score

    def get_final_item_embeddings(self):
        return self.item_embedding.weight
