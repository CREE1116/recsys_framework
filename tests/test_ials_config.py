
import yaml
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.general.ials import iALS

class MockDataLoader:
    def __init__(self):
        self.n_users = 100
        self.n_items = 100

def test_ials_config():
    config_path = 'configs/model/general/ials.yaml'
    print(f"Testing config file: {config_path}")
    
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Check scalars in yaml
    embedding_dim = yaml_config['model']['embedding_dim']
    print(f"embedding_dim: {embedding_dim} (Type: {type(embedding_dim)})")
    
    if isinstance(embedding_dim, list):
        print("FAIL: embedding_dim in yaml is still a list!")
        return False
        
    # Test Robust Initialization with List (Simulation of potential bad config)
    print("\nTesting robust initialization with list inputs...")
    bad_config = {
        'model': {
            'embedding_dim': [128],
            'reg_lambda': [0.01],
            'alpha': [40],
            'max_iter': [15]
        },
        'device': 'cpu'
    }
    data_loader = MockDataLoader()
    
    try:
        model = iALS(bad_config, data_loader)
        print(f"Model initialized successfully with list inputs.")
        print(f"  -> model.embedding_dim: {model.embedding_dim} (Type: {type(model.embedding_dim)})")
        
        if isinstance(model.embedding_dim, list):
             print("FAIL: Model did not unwrap the list!")
             return False
        
        return True
    except Exception as e:
        print(f"FAIL: Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if test_ials_config():
        print("\nVerification PASSED")
        sys.exit(0)
    else:
        print("\nVerification FAILED")
        sys.exit(1)
