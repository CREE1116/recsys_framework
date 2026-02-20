import torch
import torch.nn as nn
from src.loss import BPRLoss, SampledSoftmaxLoss
from ..base_model import BaseModel
from .csar_layers import CSAR_basic

class CSAR_Basic(BaseModel):
    def __init__(self, config, data_loader):
        super(CSAR_Basic, self).__init__(config, data_loader)
        
        self.embedding_dim = self.config['model']['embedding_dim']
        if isinstance(self.embedding_dim, list):
            self.embedding_dim = self.embedding_dim[0]
            
        self.n_users = self.data_loader.n_users
        self.n_items = self.data_loader.n_items
            
        self.num_interests = self.config['model']['num_interests']
        if isinstance(self.num_interests, list):
            self.num_interests = self.num_interests[0]
        
        self.reg_lambda = self.config['model'].get('reg_lambda', 500.0)
        self.normalize = self.config['model'].get('normalize', True)
        self.use_ridge_g = self.config['model'].get('use_ridge_g', False)
        self.ema_momentum = self.config['model'].get('ema_momentum', 1.0) # Default 1.0 = no EMA
        
        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        
        # CSAR Basic Layer
        self.model_layer = CSAR_basic(self.num_interests, self.embedding_dim, reg_lambda=self.reg_lambda, normalize=self.normalize)
        
        self._init_weights()
        
        # Ridge mode related
        if self.use_ridge_g:
            self.X = self._prepare_interaction_matrix()
            self._cached_G = None
            self._cached_d_inv_sqrt = None

        # Loss
        self.loss_type = self.config['train'].get('loss_type', 'bpr')
        if self.loss_type == 'sampled_softmax':
            self.loss_fn = SampledSoftmaxLoss(temperature=0.1)
        else:
            self.loss_fn = BPRLoss()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def _prepare_interaction_matrix(self):
        """Build sparse interaction matrix (N x M)"""
        train_df = self.data_loader.train_df
        rows = torch.tensor(train_df['user_id'].values, dtype=torch.long)
        cols = torch.tensor(train_df['item_id'].values, dtype=torch.long)
        vals = torch.ones(len(train_df), dtype=torch.float32)
        X = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (self.n_users, self.n_items))
        return X.to(self.device).to_dense() # Following user's Mode 2 logic (dense for solve)

    def on_epoch_start(self, epoch):
        """Update G at the beginning of each epoch if use_ridge_g is True"""
        if self.use_ridge_g:
            with torch.no_grad():
                # Use .weight directly for efficiency
                user_emb = self.user_embedding.weight
                item_emb = self.item_embedding.weight
                
                # Initially compute memberships WITHOUT G (raw projection)
                # In Ridge Mode, d_inv_sqrt is handled internally or returned as None
                M_u = self.model_layer.get_membership(user_emb, d_inv_sqrt=None)
                M_i = self.model_layer.get_membership(item_emb, d_inv_sqrt=None)
                
                # Compute new G via Ridge Regression
                new_G, new_d_inv_sqrt = self.model_layer.get_gram_matrix(
                    use_ridge=True, M_u=M_u, M_i=M_i, X=self.X
                )
                
                # Apply EMA
                if self._cached_G is None:
                    self._cached_G = new_G
                else:
                    m = self.ema_momentum
                    self._cached_G = (1 - m) * self._cached_G + m * new_G
                
                self._cached_d_inv_sqrt = new_d_inv_sqrt
                
            print(f"[CSAR_Basic] G matrix updated via Ridge Regression (EMA m={self.ema_momentum}) at epoch {epoch}")

    def _get_g_and_norm(self):
        if self.use_ridge_g:
            if self._cached_G is None:
                self.on_epoch_start(0) # Initial update
            return self._cached_G, self._cached_d_inv_sqrt
        else:
            return self.model_layer.get_gram_matrix()

    def forward(self, users):
        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding.weight
        
        # G matrix (Prop) and D matrix (Norm)
        G, d_inv_sqrt = self._get_g_and_norm()
        
        # Get membership with normalization
        user_mem = self.model_layer.get_membership(user_emb, d_inv_sqrt) # [B, K]
        item_mem = self.model_layer.get_membership(item_emb, d_inv_sqrt) # [N, K]
        
        # Score = (User_Mem @ G) @ Item_Mem.T
        user_prop = torch.matmul(user_mem, G)
        scores = torch.matmul(user_prop, item_mem.t())
        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        G, d_inv_sqrt = self._get_g_and_norm()
        
        user_mem = self.model_layer.get_membership(user_emb, d_inv_sqrt)
        item_mem = self.model_layer.get_membership(item_emb, d_inv_sqrt)
        
        # Score = (User_Mem @ G) @ Item_Mem.T
        user_prop = torch.matmul(user_mem, G)
        scores = (user_prop * item_mem).sum(dim=-1)
        return scores

    def calc_loss(self, batch_data):
        users = batch_data['user_id']
        pos_items = batch_data['pos_item_id']
        neg_items = batch_data['neg_item_id']
        
        pos_scores = self.predict_for_pairs(users, pos_items)
        neg_scores = self.predict_for_pairs(users, neg_items)
        
        main_loss = self.loss_fn(pos_scores, neg_scores)
        
        # Return tuple of losses for the trainer
        return (main_loss, torch.tensor(0.0, device=self.device)), {"scale": self.model_layer.scale.item()}

    def get_final_item_embeddings(self):
        return self.item_embedding.weight.detach()

    def get_interest_keys(self):
        return self.model_layer.interest_keys.detach()
