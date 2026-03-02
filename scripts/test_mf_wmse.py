import torch
import torch.nn as nn
import sys
import os

# Framework 경로 추가
sys.path.append(os.path.abspath('.'))

from src.models.general.mf import MF

class MockDataLoader:
    def __init__(self):
        self.n_users = 10
        self.n_items = 20

def test_mf_pointwise_loss():
    print("Testing MF Pointwise Loss (All-Item Weighted MSE)...")
    
    config = {
        'model': {'embedding_dim': 8},
        'train': {
            'loss_type': 'pointwise',
            'w_mse': 5.0
        },
        'device': 'cpu'
    }
    
    data_loader = MockDataLoader()
    model = MF(config, data_loader)
    
    batch_size = 2
    batch_data = {
        'user_id': torch.tensor([0, 1]),
        'item_id': torch.tensor([5, 10]) # Positive items
    }
    
    # calc_loss 호출
    (loss,), _ = model.calc_loss(batch_data)
    
    print(f"Calculated Loss: {loss.item()}")
    
    # 수동 검증
    all_scores = model.forward(batch_data['user_id'])
    targets = torch.zeros_like(all_scores)
    targets[0, 5] = 1.0
    targets[1, 10] = 1.0
    
    weights = targets * (5.0 - 1.0) + 1.0
    expected_loss = (torch.pow(all_scores - targets, 2) * weights).mean()
    
    print(f"Expected Loss: {expected_loss.item()}")
    
    assert torch.allclose(loss, expected_loss), "Loss calculation mismatch!"
    print("Test Passed!")

if __name__ == "__main__":
    test_mf_pointwise_loss()
