import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel
from src.loss import BPRLoss, MSELoss, InfoNCELoss

class DGCF(BaseModel):
    """
    DGCF: Disentangled Graph Collaborative Filtering (SIGIR '20)
    - Graph-based implementation
    - Intent-aware message passing with dynamic routing
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        
        self.embedding_dim = int(config['model']['embedding_dim'])
        self.num_intents = int(config['model'].get('num_intents', 4))
        self.n_layers = int(config['model'].get('n_layers', 2)) # Usually fewer layers needed for DGCF
        self.lambda_cor = float(config['model'].get('lambda_cor', 0.01))
        
        self.n_users = self.data_loader.n_users
        self.n_items = self.data_loader.n_items
        
        # Initial Embeddings (Xavier Init)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        
        # Load Graph
        # We need indices to perform dynamic routing (attention on edges)
        self.raw_adj = self.data_loader.get_interaction_graph().to(self.device)
        
        if self.raw_adj.is_sparse:
            self.raw_adj = self.raw_adj.coalesce()
            self.edge_indices = self.raw_adj.indices() # [2, E]
            self.use_sparse_graph = True
            self.num_nodes = self.raw_adj.size(0)
        else:
            # Dense Graph (MPS support)
            # self.raw_adj is [N, N] tensor with 1s
            self.use_sparse_graph = False
            self.num_nodes = self.raw_adj.size(0)
            
            # Extract indices for dynamic routing attention
            # nonzero() returns [E, 2], we want [2, E]
            self.edge_indices = self.raw_adj.nonzero().t()
            
        
        # Disentangled Initialization: Separate chunks or separate embeddings?
        # DGCF paper uses chunks of a large embedding. 
        # Here we use separate transformation for each intent or just separate initial embeddings?
        # Usually DGCF initializes K independent embeddings.
        # We will split self.embedding_dim into K chunks? 
        # OR maintain K separate full-dim embeddings if embedding_dim is small (64).
        # Let's assume embedding_dim is TOTAL dimension.
        # For simplicity and capacity, let's keep embedding_dim per intent if K is small.
        # But config['embedding_dim'] usually means total.
        # Let's use K separate embeddings of size (dim // K) to match total dim constraint accurately.
        # OR just use full dim per intent if capacity allows. 
        # Let's use full dim per intent to be safe on capacity (64 dim is small).
        
        self.user_intent_embs = nn.ModuleList([
            nn.Embedding(self.n_users, self.embedding_dim) for _ in range(self.num_intents)
        ])
        self.item_intent_embs = nn.ModuleList([
            nn.Embedding(self.n_items, self.embedding_dim) for _ in range(self.num_intents)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        for emb in self.user_intent_embs:
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.item_intent_embs:
            nn.init.xavier_uniform_(emb.weight)
            
    def _get_ego_embeddings(self):
        # Concatenate u/i for easier graph ops
        # List of [N+M, D] tensors
        ego_embeddings = []
        for k in range(self.num_intents):
            u_k = self.user_intent_embs[k].weight
            i_k = self.item_intent_embs[k].weight
            ego_embeddings.append(torch.cat([u_k, i_k], dim=0))
        return ego_embeddings

    def forward(self, users=None):
        # Propagate
        all_embeddings = self._propagate()
        
        # Aggregate intents: Sum or Concat? DGCF usually sums.
        final_embs = sum(all_embeddings) / self.num_intents
        
        users_emb, items_emb = torch.split(final_embs, [self.n_users, self.n_items])
        
        if users is not None:
            batch_users = users_emb[users]
            scores = torch.matmul(batch_users, items_emb.T)
            return scores
        
        return users_emb, items_emb
        
    def _propagate(self):
        # Iterative Routing & Propagation
        ego_embeddings = self._get_ego_embeddings() # List of K tensors [Nodes, D]
        all_layer_embeddings = [ego_embeddings] # Store for skip connection? DGCF usually concatenates layers.
        
        # Dynamic Routing Loop
        # In each layer, we update adjacency values based on current embeddings
        
        current_embeddings = ego_embeddings
        
        indices = self.edge_indices
        rows, cols = indices[0], indices[1]
        
        for layer in range(self.n_layers):
            next_embeddings = []
            
            # 1. Provide Intent-Aware Attention (Routing)
            # Compute score(u_k, i_k) for all edges for all k
            # We want A_k(u, i) = softmax(u_k * i_k) over k
            
            # Gather node embeddings for edges: [K, E, D]
            # This uses a lot of memory if E is large. 
            # Optimization: Compute dot product in a loop or batched?
            # 100k edges * 64 dim * 4 intents is small.
            
            intent_logits = []
            for k in range(self.num_intents):
                emb = current_embeddings[k]
                # node_u: emb[rows], node_i: emb[cols]
                # score: (node_u * node_i).sum(-1)
                # tanh activation is common in routing
                score = (emb[rows] * emb[cols]).sum(dim=1) 
                intent_logits.append(score) # [E]
            
            # Stack: [E, K]
            intent_logits = torch.stack(intent_logits, dim=1)
            intent_logits = torch.tanh(intent_logits) 
            
            # Softmax over intents -> [E, K]
            # Weights for each intent graph
            att_weights = F.softmax(intent_logits, dim=1)
            
            # 2. Propagate for each intent
            for k in range(self.num_intents):
                vals = att_weights[:, k]
                
                if self.use_sparse_graph:
                     adj_k = torch.sparse_coo_tensor(indices, vals, size=(self.num_nodes, self.num_nodes)).coalesce()
                     # Message Passing
                     # E_k = A_k * E_k
                     emb_k = torch.sparse.mm(adj_k, current_embeddings[k])
                else:
                    # Dense Graph (MPS)
                    # Construct Dense Adj: [N, N]
                    # This is slow O(N^2) but needed for MPS if sparse is not supported
                    # Or use masking?
                    # Create empty matrix
                    adj_k = torch.zeros((self.num_nodes, self.num_nodes), device=self.device)
                    # Fill values at indices
                    adj_k[indices[0], indices[1]] = vals
                    
                    emb_k = torch.matmul(adj_k, current_embeddings[k])
                    
                next_embeddings.append(emb_k)
            
            current_embeddings = next_embeddings
            all_layer_embeddings.append(current_embeddings)
            
        # Final aggregation: Sum over layers (LightGCN style) or just Last?
        # DGCF sums layers.
        
        final_embeddings_per_intent = []
        for k in range(self.num_intents):
            # Sum embeddings from all layers for intent k
            layer_sum = sum([layers[k] for layers in all_layer_embeddings])
            # Average?
            layer_mean = layer_sum / (self.n_layers + 1)
            final_embeddings_per_intent.append(layer_mean)
            
        return final_embeddings_per_intent

    def calc_loss(self, batch_data):
        users = batch_data['user_id'].squeeze(-1)
        pos_items = batch_data['pos_item_id'].squeeze(-1)
        neg_items = batch_data['neg_item_id'].squeeze(-1)
        
        # Forward pass to get updated embeddings
        # Since graph prop is full-batch efficiently, we do it once per batch logic?
        # No, we must do it every step. DGCF is full-batch model usually.
        # But we are in mini-batch training loop.
        # Ideally we compute full graph prop once per batch.
        
        final_intent_embs = self._propagate() # List of K tensors [Nodes, D]
        
        u_final_embs = [emb[:self.n_users] for emb in final_intent_embs]
        i_final_embs = [emb[self.n_users:] for emb in final_intent_embs]
        
        # BPR Loss per intent
        bpr_loss = 0
        for k in range(self.num_intents):
            u_k = u_final_embs[k][users]
            pos_k = i_final_embs[k][pos_items]
            neg_k = i_final_embs[k][neg_items]
            
            pos_score = (u_k * pos_k).sum(dim=-1)
            neg_score = (u_k * neg_k).sum(dim=-1)
            
            bpr_loss += -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10).mean()
            
        bpr_loss /= self.num_intents
        
        # Independence Loss (Correlation)
        cor_loss = 0
        # Check simple method: Centered Covariance or just dot product?
        # Using simple dot product of weights
        
        # Aggregate user embs for correlation check?
        # DGCF checks correlation of the *learned representations*.
        # Sampled users correlation
        
        # To save memory, check correlation on batch users
        for k1 in range(self.num_intents):
            for k2 in range(k1 + 1, self.num_intents):
                 u_k1 = u_final_embs[k1][users]
                 u_k2 = u_final_embs[k2][users]
                 
                 # Centered correlation
                 u_k1 = u_k1 - u_k1.mean(dim=0)
                 u_k2 = u_k2 - u_k2.mean(dim=0)
                 cov = (u_k1 * u_k2).sum(dim=-1).abs().mean()
                 cor_loss += cov
                 
        return (bpr_loss, self.lambda_cor * cor_loss), {
            'bpr': bpr_loss.item(),
            'cor': cor_loss.item()
        }

    def predict_for_pairs(self, user_ids, item_ids):
        # Full propagation needed for inference too
        final_intent_embs = self._propagate()
        
        # Average over intents
        combined_u = sum([emb[:self.n_users] for emb in final_intent_embs]) / self.num_intents
        combined_i = sum([emb[self.n_users:] for emb in final_intent_embs]) / self.num_intents
        
        u_emb = combined_u[user_ids]
        i_emb = combined_i[item_ids]
        
        return (u_emb * i_emb).sum(dim=-1)

    def get_final_item_embeddings(self):
        final_intent_embs = self._propagate()
        combined_i = sum([emb[self.n_users:] for emb in final_intent_embs]) / self.num_intents
        return combined_i.detach()