import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..base_model import BaseModel
class MacridVAE(BaseModel):
    def __init__(self, config, data_loader):
        super(MacridVAE, self).__init__(config, data_loader)
        
        self.n_items = self.data_loader.n_items
        self.latent_dim = self.config['model'].get('latent_dim', 64)
        self.num_concepts = self.config['model'].get('num_concepts', 4)
        self.hidden_dim = self.config['model'].get('hidden_dim', 600)
        self.dropout_rate = self.config['model'].get('dropout_rate', 0.5)
        self.beta = self.config['model'].get('beta', 0.2)
        self.tau = self.config['model'].get('tau', 0.1)  # Gumbel-Softmax temperature
        
        # Encoder
        self.encoder_shared = nn.Sequential(
            nn.Linear(self.n_items, self.hidden_dim),
            nn.Tanh()
        )
        
        # Macro: Concept selection (routing)
        self.concept_logits = nn.Linear(self.hidden_dim, self.num_concepts)
        
        # Micro: Per-concept latent distribution
        self.encoder_mu = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.latent_dim) 
            for _ in range(self.num_concepts)
        ])
        self.encoder_logvar = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.latent_dim) 
            for _ in range(self.num_concepts)
        ])
        
        # Decoder: Per-concept item embeddings
        self.item_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(self.n_items, self.latent_dim))
            for _ in range(self.num_concepts)
        ])
        
        for item_emb in self.item_embeddings:
            nn.init.xavier_uniform_(item_emb)
        
        self.update_count = 0

    def gumbel_softmax(self, logits, tau=1.0, hard=False):
        """Gumbel-Softmax 샘플링"""
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / tau
        y_soft = F.softmax(gumbels, dim=-1)
        
        if hard:
            # Straight-through estimator
            index = y_soft.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            return y_hard - y_soft.detach() + y_soft
        else:
            return y_soft

    def forward(self, users):
        # 1. Input
        x = self._get_user_history_matrix(users)
        h_norm = F.normalize(x, p=2, dim=1)
        
        if self.training:
            h_norm = F.dropout(h_norm, p=self.dropout_rate)
        
        # 2. Encoder
        h = self.encoder_shared(h_norm)
        
        # 3. Macro: Concept routing
        concept_logits = self.concept_logits(h)  # [B, K]
        concept_probs = self.gumbel_softmax(concept_logits, self.tau, hard=self.training)
        
        # 4. Micro: Per-concept latent
        mu_list = []
        logvar_list = []
        z_list = []
        
        for k in range(self.num_concepts):
            mu_k = self.encoder_mu[k](h)  # [B, d]
            logvar_k = self.encoder_logvar[k](h)  # [B, d]
            z_k = self.reparameterize(mu_k, logvar_k)  # [B, d]
            
            mu_list.append(mu_k)
            logvar_list.append(logvar_k)
            z_list.append(z_k)
        
        mu = torch.stack(mu_list, dim=1)  # [B, K, d]
        logvar = torch.stack(logvar_list, dim=1)  # [B, K, d]
        z = torch.stack(z_list, dim=1)  # [B, K, d]
        
        # 5. Decode: Weighted sum over concepts
        logits = torch.zeros(users.size(0), self.n_items, device=users.device)
        
        for k in range(self.num_concepts):
            # z_k: [B, d], item_emb_k: [M, d]
            logits_k = torch.matmul(z[:, k, :], self.item_embeddings[k].T)  # [B, M]
            logits += concept_probs[:, k:k+1] * logits_k  # [B, M]
        
        if self.training:
            return logits, mu, logvar, concept_probs
        else:
            return logits

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def _get_user_history_matrix(self, users):
        batch_size = users.size(0)
        x = torch.zeros(batch_size, self.n_items, device=users.device)
        users_list = users.cpu().numpy()
        for i, u_id in enumerate(users_list):
            hist_items = list(self.data_loader.train_user_history.get(u_id, []))
            x[i, hist_items] = 1.0
        return x

    def calc_loss(self, batch_data):
        users = batch_data['user_id'].squeeze(-1)
        targets = self._get_user_history_matrix(users)
        
        logits, mu, logvar, concept_probs = self.forward(users)
        
        # 1. Reconstruction Loss
        log_softmax = F.log_softmax(logits, dim=1)
        nll_loss = -(log_softmax * targets).sum(dim=1).mean()
        
        # 2. KL Divergence (per concept, weighted by concept prob)
        kl_loss = 0
        for k in range(self.num_concepts):
            mu_k = mu[:, k, :]  # [B, d]
            logvar_k = logvar[:, k, :]  # [B, d]
            kl_k = -0.5 * torch.sum(1 + logvar_k - mu_k.pow(2) - logvar_k.exp(), dim=1)
            kl_loss += (concept_probs[:, k] * kl_k).mean()
        
        params_to_log = {
            'nll': nll_loss.item(),
            'kl': kl_loss.item(),
            'concept_entropy': -(concept_probs * torch.log(concept_probs + 1e-10)).sum(1).mean().item()
        }
        
        return (nll_loss , self.beta * kl_loss), params_to_log

    def predict_for_pairs(self, user_ids, item_ids):
        logits = self.forward(user_ids)
        scores = logits[torch.arange(len(user_ids)), item_ids]
        return scores

    def get_final_item_embeddings(self):
        return None