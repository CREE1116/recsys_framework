
import torch
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.general.macr import MACR

class MockDataLoader:
    def __init__(self):
        self.n_users = 100
        self.n_items = 100

def test_macr_loss():
    print("Testing MACR calc_loss...")
    
    # Mock config and data loader
    config = {
        'model': {
            'embedding_dim': 32,
            'c': 40.0,
            'alpha': 1e-3,
            'beta': 1e-3
        },
        'device': 'cpu'  # Added device key
    }
    data_loader = MockDataLoader()
    
    # Initialize model
    model = MACR(config, data_loader)
    
    # Create dummy batch data (simulating BPR/Pairwise loader output)
    batch_size = 4
    batch_data = {
        'user_id': torch.randint(0, 100, (batch_size,)),
        'pos_item_id': torch.randint(0, 100, (batch_size,)), # Key changed from item_id to pos_item_id
        'neg_item_id': torch.randint(0, 100, (batch_size,))
    }
    
    # Run calc_loss
    try:
        losses, log_info = model.calc_loss(batch_data)
        print("Successfully calculated loss.")
        print(f"Losses: {losses}")
        print(f"Log Info: {log_info}")
        return True
    except KeyError as e:
        print(f"KeyError: {e}")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if test_macr_loss():
        print("Verification PASSED")
        sys.exit(0)
    else:
        print("Verification FAILED")
        sys.exit(1)
