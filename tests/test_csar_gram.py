
import torch
import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.models.csar.csar_layers import CSAR_basic

def test_gram_inverse():
    num_interests = 10
    embedding_dim = 32
    layer = CSAR_basic(num_interests, embedding_dim)
    
    # Test with default reg_lambda
    G_inv = layer.get_gram_matrix()
    
    print(f"G_inv shape: {G_inv.shape}")
    assert G_inv.shape == (num_interests, num_interests)
    
    # Check if symmetric
    is_symmetric = torch.allclose(G_inv, G_inv.t(), atol=1e-5)
    print(f"Is symmetric: {is_symmetric}")
    assert is_symmetric
    
    # Check with different reg_lambda
    G_inv_large = layer.get_gram_matrix(reg_lambda=100.0)
    print(f"Mean value with reg_lambda=1.0: {G_inv.abs().mean().item():.6f}")
    print(f"Mean value with reg_lambda=100.0: {G_inv_large.abs().mean().item():.6f}")
    
    assert G_inv_large.abs().mean() < G_inv.abs().mean()
    
    print("Test passed!")

if __name__ == "__main__":
    test_gram_inverse()
