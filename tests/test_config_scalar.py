
import yaml
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_yaml_config():
    config_path = 'configs/model/csar/csar_basic.yaml'
    print(f"Testing config file: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check batch_size
    batch_size = config['train']['batch_size']
    print(f"batch_size: {batch_size} (Type: {type(batch_size)})")
    
    if isinstance(batch_size, list):
        print("FAIL: batch_size is still a list!")
        return False
    
    if not isinstance(batch_size, int):
        print("FAIL: batch_size is not an int!")
        return False
        
    # Check num_interests
    num_interests = config['model']['num_interests']
    print(f"num_interests: {num_interests} (Type: {type(num_interests)})")
    
    if isinstance(num_interests, list):
        print("FAIL: num_interests is still a list!")
        return False

    print("PASS: batch_size and num_interests are scalars.")
    return True

if __name__ == "__main__":
    if test_yaml_config():
        sys.exit(0)
    else:
        sys.exit(1)
