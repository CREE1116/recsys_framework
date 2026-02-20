import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from ..base_model import BaseModel
import time
from tqdm import tqdm

class SLIM(BaseModel):
    """
    SLIM (Sparse Linear Methods for Top-N Recommender Systems)
    
    Optimized for:
    - Memory efficiency (column-wise optimization)
    - MacBook MPS compatibility
    - Large-scale datasets
    
    Reference: Ning & Karypis, ICDM 2011
    """
    def __init__(self, config, data_loader):
        super(SLIM, self).__init__(config, data_loader)
        
        # Hyperparameters
        self.alpha = config['model'].get('alpha', 0.1)
        self.l1_ratio = config['model'].get('l1_ratio', 0.5)
        self.positive_constraint = config['model'].get('positive_constraint', True)
        self.max_iter = config['model'].get('max_iter', 100)
        self.tol = float(config['model'].get('tol', 1e-4))
        
        # Data dimensions
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        
        # Weight matrix (Item × Item)
        # Store on CPU initially, move to device only for inference
        self.W = None
        self.train_matrix_csr = None
        
        # Device handling for MPS
        self._setup_device()
        
        print(f"[SLIM] Initialized:")
        print(f"  - alpha: {self.alpha}")
        print(f"  - l1_ratio: {self.l1_ratio}")
        print(f"  - positive: {self.positive_constraint}")
        print(f"  - device: {self.inference_device}")

    def _setup_device(self):
        """
        MPS-aware device setup
        Training on CPU (sklearn), inference on device
        """
        if self.device == 'mps':
            self.train_device = 'cpu'  # sklearn on CPU
            self.inference_device = 'mps'  # torch inference on MPS
        elif self.device == 'cuda':
            self.train_device = 'cpu'
            self.inference_device = 'cuda'
        else:
            self.train_device = 'cpu'
            self.inference_device = 'cpu'
        
        print(f"[SLIM] Device config: train={self.train_device}, inference={self.inference_device}")

    def _build_sparse_matrix(self, data_loader):
        """Build user-item interaction matrix"""
        train_df = data_loader.train_df
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        values = np.ones(len(train_df), dtype=np.float32)
        
        X = sp.csr_matrix(
            (values, (rows, cols)), 
            shape=(self.n_users, self.n_items),
            dtype=np.float32
        )
        
        print(f"[SLIM] Interaction matrix: {X.shape}, nnz={X.nnz:,}, density={X.nnz/(X.shape[0]*X.shape[1]):.6f}")
        return X

    def fit(self, data_loader):
        """
        Column-wise optimization using sklearn ElasticNet
        Memory-efficient, works on any device
        """
        print(f"\n{'='*60}")
        print(f"[SLIM] Starting training...")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # 1. Build sparse matrix
        self.train_matrix_csr = self._build_sparse_matrix(data_loader)
        X_csc = self.train_matrix_csr.tocsc()  # Column-sparse for efficiency
        
        # 2. Initialize weight matrix (on CPU during training)
        W = np.zeros((self.n_items, self.n_items), dtype=np.float32)
        
        # 3. Import sklearn
        try:
            from sklearn.linear_model import ElasticNet
        except ImportError:
            raise ImportError("sklearn required for SLIM. Install: pip install scikit-learn")
        
        # 4. Column-wise optimization
        print(f"[SLIM] Optimizing {self.n_items} columns...")
        
        # Progress tracking
        n_nonzero_columns = 0
        total_nnz = 0
        
        for j in tqdm(range(self.n_items), desc="SLIM Training", ncols=80):
            # Target: j-th item's ratings
            y = X_csc[:, j].toarray().ravel()
            
            # Skip if no interactions
            if np.sum(y) == 0:
                continue
            
            # Efficiently zero out the j-th column without copying the matrix
            # We modify data array directly to preserve sparsity structure and avoid warnings
            start_ptr = X_csc.indptr[j]
            end_ptr = X_csc.indptr[j+1]
            
            # Backup original column data
            # Note: We only need to backup .data because .indices aren't changed
            original_data = X_csc.data[start_ptr:end_ptr].copy()
            
            # Zero out the column in-place
            X_csc.data[start_ptr:end_ptr] = 0.0
            
            # Elastic Net solver
            model = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                positive=self.positive_constraint,
                max_iter=self.max_iter,
                tol=self.tol,
                fit_intercept=False,
                selection='random',  # Faster than cyclic
                copy_X=False # Prevent internal copy by sklearn if possible
            )
            
            # Fit column using the modified X_csc
            try:
                model.fit(X_csc, y)
                W[:, j] = model.coef_
                
                # Track sparsity
                nnz = np.sum(model.coef_ > 0)
                if nnz > 0:
                    n_nonzero_columns += 1
                    total_nnz += nnz
                    
            except Exception as e:
                print(f"\n[SLIM] Warning: Failed to fit column {j}: {e}")
            
            finally:
                # Restore the column data for the next iteration
                X_csc.data[start_ptr:end_ptr] = original_data
        
        # 5. Store weight matrix
        self.W = W
        
        # 6. Statistics
        elapsed = time.time() - start_time
        sparsity = 1.0 - (total_nnz / (self.n_items * self.n_items))
        avg_nnz_per_col = total_nnz / max(n_nonzero_columns, 1)
        
        print(f"\n{'='*60}")
        print(f"[SLIM] Training complete!")
        print(f"  - Time: {elapsed:.2f}s ({elapsed/self.n_items:.3f}s per item)")
        print(f"  - Non-zero columns: {n_nonzero_columns}/{self.n_items}")
        print(f"  - Total nnz: {total_nnz:,}")
        print(f"  - Sparsity: {sparsity:.4f}")
        print(f"  - Avg nnz per column: {avg_nnz_per_col:.1f}")
        print(f"{'='*60}\n")

    def forward(self, user_ids, item_ids=None):
        """
        Forward pass: compute scores for users
        
        Args:
            user_ids: Tensor or array of user IDs
            item_ids: Optional tensor of item IDs for specific predictions
            
        Returns:
            scores: (batch_size, n_items) or (batch_size,)
        """
        if self.W is None:
            raise RuntimeError("[SLIM] Model not fitted. Call fit() first.")
        
        # Convert user_ids to numpy
        if isinstance(user_ids, torch.Tensor):
            u_ids_np = user_ids.cpu().numpy()
        else:
            u_ids_np = np.array(user_ids)
        
        # Get user histories (sparse)
        user_input_sparse = self.train_matrix_csr[u_ids_np]
        
        # Convert to dense and move to inference device
        user_input_dense = user_input_sparse.toarray().astype(np.float32)
        user_input = torch.from_numpy(user_input_dense).to(self.inference_device)
        
        # Weight matrix to device (cached)
        if not hasattr(self, 'W_tensor') or self.W_tensor is None:
            self.W_tensor = torch.from_numpy(self.W).to(self.inference_device)
        
        # Compute scores: U @ W
        with torch.no_grad():
            scores = user_input @ self.W_tensor
        
        # Return specific items if requested
        if item_ids is not None:
            if not isinstance(item_ids, torch.Tensor):
                item_ids = torch.tensor(item_ids, device=self.inference_device)
            else:
                item_ids = item_ids.to(self.inference_device)
            
            batch_indices = torch.arange(len(user_ids), device=self.inference_device)
            return scores[batch_indices, item_ids]
        
        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        """
        Predict scores for (user, item) pairs
        
        Args:
            user_ids: (N,) tensor
            item_ids: (N,) tensor
            
        Returns:
            scores: (N,) tensor
        """
        return self.forward(user_ids, item_ids)

    def recommend_topk(self, user_ids, k=20, exclude_seen=True):
        """
        Generate top-K recommendations
        
        Args:
            user_ids: Tensor of user IDs
            k: Number of recommendations
            exclude_seen: Whether to exclude training items
            
        Returns:
            topk_items: (batch_size, k) array
            topk_scores: (batch_size, k) array
        """
        if not isinstance(user_ids, torch.Tensor):
            user_ids = torch.tensor(user_ids, device=self.inference_device)
        else:
            user_ids = user_ids.to(self.inference_device)
        
        # Get all scores
        scores = self.forward(user_ids)  # (batch_size, n_items)
        
        if exclude_seen:
            # Mask observed items
            u_ids_np = user_ids.cpu().numpy()
            for i, uid in enumerate(u_ids_np):
                seen_items = self.train_matrix_csr[uid].indices
                scores[i, seen_items] = -1e9
        
        # Top-K
        topk_scores, topk_items = torch.topk(scores, k, dim=1)
        
        return topk_items.cpu().numpy(), topk_scores.cpu().numpy()

    def calc_loss(self, batch_data):
        """
        Loss calculation (not used in SLIM - closed-form solution)
        """
        return (torch.tensor(0.0, device=self.inference_device, requires_grad=True),), None

    def get_final_item_embeddings(self):
        """
        Return item similarity matrix W
        """
        if self.W is None:
            raise RuntimeError("[SLIM] Model not fitted.")
        
        if not hasattr(self, 'W_tensor') or self.W_tensor is None:
            self.W_tensor = torch.from_numpy(self.W).to(self.inference_device)
        
        return self.W_tensor

    def get_train_matrix(self, data_loader):
        """Return training matrix"""
        if self.train_matrix_csr is None:
            self.train_matrix_csr = self._build_sparse_matrix(data_loader)
        return self.train_matrix_csr

    def save_weights(self, filepath):
        """Save weight matrix to disk"""
        if self.W is None:
            raise RuntimeError("[SLIM] No weights to save.")
        
        np.save(filepath, self.W)
        print(f"[SLIM] Weights saved to {filepath}")

    def load_weights(self, filepath):
        """Load weight matrix from disk"""
        self.W = np.load(filepath)
        self.W_tensor = None  # Reset cache
        print(f"[SLIM] Weights loaded from {filepath}")