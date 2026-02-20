import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.csar.csar_layers import CSAR_basic

def test_csar_basic_grad():
    print("Testing CSAR_basic differentiability (with Symmetric Normalization)...")
    num_interests = 16
    embedding_dim = 32
    reg_lambda = 100.0
    
    layer = CSAR_basic(num_interests, embedding_dim, reg_lambda=reg_lambda, normalize=True)
    
    # Fake user/item embeddings
    user_embs = torch.randn(8, embedding_dim, requires_grad=True)
    item_embs = torch.randn(20, embedding_dim, requires_grad=True)
    
    # 1. Gram Matrix (Ridge Prop)
    G, d_inv_sqrt = layer.get_gram_matrix()
    
    # 2. Membership with Norm
    user_mem = layer.get_membership(user_embs, d_inv_sqrt)
    item_mem = layer.get_membership(item_embs, d_inv_sqrt)
    
    # 3. Score calculation
    user_prop = torch.matmul(user_mem, G)
    scores = torch.matmul(user_prop, item_mem.t())
    
    # 4. Dummy Loss
    loss = scores.sum()
    
    # 5. Backward
    loss.backward()
    
    print(f"Loss: {loss.item():.4f}")
    print(f"User Embs Grad: {user_embs.grad is not None and user_embs.grad.abs().sum().item() > 0}")
    print(f"Item Embs Grad: {item_embs.grad is not None and item_embs.grad.abs().sum().item() > 0}")
    print(f"Layer Keys Grad: {layer.interest_keys.grad is not None and layer.interest_keys.grad.abs().sum().item() > 0}")
    print(f"Layer Scale Grad: {layer.scale.grad is not None and layer.scale.grad.abs().sum().item() > 0}")
    
    if user_embs.grad is not None and layer.interest_keys.grad is not None:
        print(">>> SUCCESS: Gradients are flowing correctly through Prop and Norm!")
    else:
        print(">>> FAILURE: Gradients are blocked!")

if __name__ == "__main__":
    test_csar_basic_grad()
