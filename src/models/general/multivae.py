import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..base_model import BaseModel

class MultiVAE(BaseModel):
    """
    Multi-VAE: Variational Autoencoder for Collaborative Filtering
    - Generative model with Multinomial Likelihood.
    - Uses Encoder (p_phi(z|x)) and Decoder (p_theta(x|z)).
    """
    def __init__(self, config, data_loader):
        super(MultiVAE, self).__init__(config, data_loader)
        
        self.n_items = self.data_loader.n_items
        self.latent_dim = self.config['model'].get('latent_dim', 200)
        self.hidden_dim = self.config['model'].get('hidden_dim', 600)
        self.dropout_rate = self.config['model'].get('dropout_rate', 0.5)
        self.anneal_cap = self.config['model'].get('anneal_cap', 0.2)
        self.total_anneal_steps = self.config['model'].get('total_anneal_steps', 200000)
        
        self.update_count = 0
        
        # Encoder
        self.encoder_dims = [self.n_items, self.hidden_dim, self.latent_dim * 2] # *2 for mu and logvar
        self.encoder = nn.Sequential(
            nn.Linear(self.n_items, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.latent_dim * 2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.n_items)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.001)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, users):
        x = self._get_user_history_matrix(users)
        h = F.normalize(x, p=2, dim=1)
        
        if self.training:
            h = F.dropout(h, p=self.dropout_rate, training=True)
            
        # [DEBUG] Leakage Check (Only during eval)
        if not self.training:
            # Check if input 'x' contains any items from test set for these users
            # This is expensive, so only do it once or for first batch
            if not hasattr(self, '_leakage_checked'):
                self._leakage_checked = True
                test_df = self.data_loader.test_df
                # Create a lookup for test items: user_id -> test_item_id
                test_items_map = test_df.set_index('user_id')['item_id'].to_dict()
                
                users_np = users.cpu().numpy()
                leak_count = 0
                for i, u_id in enumerate(users_np):
                    if u_id in test_items_map:
                        test_item = test_items_map[u_id]
                        if x[i, test_item] > 0:
                            leak_count += 1
                            print(f"[CRITICAL WARNING] Data Leakage detected for user {u_id}: Test item {test_item} is in input!")
                
                if leak_count > 0:
                     print(f"[CRITICAL] Total leaked users in batch: {leak_count}")
                else:
                     print("[INFO] No data leakage detected in MultiVAE input (checked first batch).")

        mu_logvar = self.encoder(h)
        mu, logvar = torch.chunk(mu_logvar, 2, dim=1)
        
        if self.training:
            z = self.reparameterize(mu, logvar)
            logits = self.decoder(z)
            return logits, mu, logvar
        else:
            z = mu 
            logits = self.decoder(z)
            return logits

    def _get_user_history_matrix(self, users):
        batch_size = users.size(0)
        x = torch.zeros(batch_size, self.n_items, device=users.device)
        for i, u_id in enumerate(users_list):
            # Ensure u_id is a native python int for dictionary lookup
            uid_key = int(u_id)
            hist_items = list(self.data_loader.train_user_history.get(uid_key, []))
            x[i, hist_items] = 1.0
        return x

    def predict_for_pairs(self, user_ids, item_ids):
        # Evaluation helper
        out = self.forward(user_ids)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
        
        scores = logits[torch.arange(len(user_ids)), item_ids]
        return scores
        
    def get_final_item_embeddings(self):
        # VAE doesn't have item embeddings in the traditional MF sense.
        # It models p(x|z). We explicitly return None or raise warning.
        return None

    def calc_loss(self, batch_data):
        users = batch_data['user_id'].squeeze(-1)
        
        # Multi-VAE takes the user, looks up their full history, and tries to reconstruct it.
        # It does NOT use the pairwise (pos, neg) format typically.
        # But our loader provides pairwise. We ignore neg_items and just reconstruct the user's FULL history.
        # Note: This is computationally heavier if batch size is large.
        
        targets = self._get_user_history_matrix(users) # Ground Truth (Multi-hot)
        logits, mu, logvar = self.forward(users)
        
        # Multinomial NLL (Log-Softmax)
        log_softmax_var = F.log_softmax(logits, dim=1)
        nll_loss = -(log_softmax_var * targets).sum(dim=1).mean()
        
        # KL Divergence
        # KL(q(z|x) || p(z)) approx -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        # Annealing
        self.update_count += 1
        anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)
        
        params_to_log = {
            'nll': nll_loss.item(),
            'kl': kl_loss.item(),
            'anneal': anneal
        }
        
        return (nll_loss , anneal * kl_loss,), params_to_log

    def forward_eval_all(self, users):
        # for evaluation override if base_model assumes something else
        out = self.forward(users)
        if isinstance(out, tuple):
            return out[0]
        return out
