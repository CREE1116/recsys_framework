import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp

from ..base_model import BaseModel

class NaiveBayes(BaseModel):
    """
    Naive Bayes for Collaborative Filtering.
    This model implements a probabilistic approach where we estimate P(item|user).
    
    Using Bayes' theorem and independence assumptions (or simpler co-occurrence models):
    Score(u, i) propto P(i) * Prod_{j in H_u} (P(j|i) / P(j)) ? 
    
    We implement a Log-Linear model based on Item-Item Co-occurrence probabilities.
    Score(u, target_item) = Sum_{interacted_item} log( P(interacted_item | target_item) ) + log( P(target_item) )
    
    where P(j | i) = Count(i, j) / Count(i)
    Smoothing is applied to avoid log(0).
    """
    def __init__(self, config, data_loader):
        super(NaiveBayes, self).__init__(config, data_loader)
        
        self.n_users = self.data_loader.n_users
        self.n_items = self.data_loader.n_items
        self.smoothing = self.config['model'].get('smoothing', 1.0) # Laplace smoothing
        
        # Logs of probabilities
        self.log_prior = None # log P(i)
        self.log_cond_prob = None # log P(j | i) matrix (n_items, n_items)
        
        # User history matrix (sparse)
        self.user_item_matrix = None
        
        print(f"NaiveBayes model initialized with smoothing={self.smoothing}.")

    def fit(self, data_loader):
        """
        Compute required probabilities from the training data.
        """
        print("Building matrices for Naive Bayes...")
        
        # 1. Build User-Item Interaction Matrix
        rows = data_loader.train_df['user_id'].values
        cols = data_loader.train_df['item_id'].values
        
        self.user_item_matrix = sp.csr_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(self.n_users, self.n_items),
            dtype=float
        )
        
        # Check if matrix is binary (implicit feedback)
        self.user_item_matrix.data = np.ones_like(self.user_item_matrix.data)
        
        # 2. Compute Item Counts (P(i))
        # sum over users -> (n_items,)
        item_counts = np.array(self.user_item_matrix.sum(axis=0)).flatten()
        total_interactions = item_counts.sum()
        
        # P(i) = (Count(i) + alpha) / (Total + alpha * N)
        # Using simple frequency for now, or just log counts + smoothing
        # log P(i)
        self.log_prior = np.log(item_counts + self.smoothing) - np.log(total_interactions + self.smoothing * self.n_items)
        self.log_prior = torch.FloatTensor(self.log_prior).to(self.device)
        
        # 3. Compute Co-occurrence Counts (Count(i, j))
        # X.T @ X gives (n_items, n_items) matrix where (i, j) is count of users who interacted with both i and j
        print("Computing co-occurrence matrix...")
        # This might be memory intensive for large item sets.
        # But ml-1m (3k items) is fine. 
        co_occurrence = (self.user_item_matrix.T @ self.user_item_matrix)
        
        # We need log P(j | i) where j is feature (history), i is class (target)
        # Or commonly in NB for text: P(word | class) -> P(history_item | target_item)
        # P(j | i) = (Count(i, j) + alpha) / (Count(i) + alpha * N_items)
        
        # Convert to dense for probability calculation if possible, or keep sparse?
        # For 3000 items, 3000*3000 = 9M floats = 36MB. Safe to be dense.
        co_occurrence_dense = co_occurrence.toarray()
        
        # Denominator: Count(i) + alpha * V
        # item_counts is (n_items,) corresponding to 'i' (target)
        denominator = item_counts[:, np.newaxis] + (self.smoothing * self.n_items)
        
        # Numerator: Count(i, j) + alpha
        numerator = co_occurrence_dense + self.smoothing
        
        # Calculate probabilities
        # We want P(j | i) where j (column) is conditioned on i (row).
        # Actually co_occurrence[i, j] is symmetric.
        # But P(j|i) != P(i|j) because denominators differ.
        # If we predict likelihood of 'i' being the target:
        # Score(i) = log P(i) + Sum_{j in history} log P(j | i)
        
        # So we need matrix M[i, j] = log P(j | i)
        # i is row (candidate), j is col (history item)
        
        # Note: denominator depends on 'i' (row). Broadcasts correctly?
        # item_counts[:, newaxis] is (N, 1). co_occurrence is (N, N).
        # numerator / denominator -> (N, N)
        
        cond_prob = numerator / denominator
        self.log_cond_prob = np.log(cond_prob)
        self.log_cond_prob = torch.FloatTensor(self.log_cond_prob).to(self.device)
        
        print("NaiveBayes fitted successfully.")

    def forward(self, users):
        """
        Score(u, i) = log P(i) + Sum_{j in H_u} log P(j | i)
        """
        # users: (batch,)
        batch_size = len(users)
        scores = []
        
        user_ids_np = users.cpu().numpy()
        
        for i, user_id in enumerate(user_ids_np):
            # 1. Get user history indices
            history_indices = self.user_item_matrix[user_id].indices # List of j
            
            if len(history_indices) == 0:
                # Just priors
                score_vector = self.log_prior
            else:
                # 2. Sum log P(j | i) for all j in history
                # log_cond_prob is [n_items(targets), n_items(features)]
                # we select columns = history_indices
                # sum over columns -> (n_items,)
                
                log_likelihoods = self.log_cond_prob[:, history_indices].sum(dim=1)
                score_vector = self.log_prior + log_likelihoods
                
            scores.append(score_vector)
            
        return torch.stack(scores)

    def predict_for_pairs(self, user_ids, item_ids):
        # Optimized for batch evaluation pairs?
        # Usually evaluation calls forward(users) and masks/ranks.
        # But if needed:
        # Score(u, i) = log P(i) + sum_{j in H_u} log P(j|i)
        
        result = []
        # Not efficient to loop, but if predict_for_pairs is used sparsely:
        # ...
        # Actually evaluate usually uses forward (full ranking).
        # If framework calls this for large batches, better implementation needed.
        # But standard eval uses forward.
        
        # Let's implement a loop for safety.
        user_ids_np = user_ids.cpu().numpy()
        item_ids_np = item_ids.cpu().numpy()
        
        for u, i in zip(user_ids_np, item_ids_np):
            history_indices = self.user_item_matrix[u].indices
            prior = self.log_prior[i]
            if len(history_indices) == 0:
                likelihood = 0
            else:
                likelihood = self.log_cond_prob[i, history_indices].sum()
            result.append(prior + likelihood)
            
        return torch.Tensor(result).to(self.device)

    def get_final_item_embeddings(self):
        return torch.eye(self.n_items, device=self.device)

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0),), None
