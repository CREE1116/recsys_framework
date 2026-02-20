import unittest
import torch
import pandas as pd
import tempfile
import shutil
import os
import yaml
from src.data_loader import DataLoader
from src.models import get_model

class TestBaselines(unittest.TestCase):
    def setUp(self):
        # Create dummy data
        self.test_dir = tempfile.mkdtemp()
        self.data_path = os.path.join(self.test_dir, 'test.inter')
        self.config_path = os.path.join(self.test_dir, 'config.yaml')
        
        # 10 users, 20 items, 50 interactions
        data = {
            'user_id': range(50),
            'item_id': range(50),
            'rating': [5] * 50,
            'timestamp': range(50)
        }
        # Remap to range
        df = pd.DataFrame(data)
        df['user_id'] = df['user_id'] % 10
        df['item_id'] = df['item_id'] % 20
        df.to_csv(self.data_path, index=False)
        
        # Base config
        self.config = {
            'dataset_name': 'test_data',
            'data_path': self.data_path,
            'separator': ',',
            'columns': ['user_id', 'item_id', 'rating', 'timestamp'],
            'has_header': True,
            'min_user_interactions': 1,
            'min_item_interactions': 1,
            'split_method': 'random',
            'train_ratio': 0.8,
            'valid_ratio': 0.1,
            'data_cache_path': self.test_dir,
            'model': {},
            'train': {
                'epochs': 1,
                'learning_rate': 0.01,
                'batch_size': 4,
                'loss_type': 'bce', # Default
            },
            'evaluation': {
                'batch_size': 4,
                'metrics': ['NDCG'],
                'top_k': [5],
                'main_metric': 'NDCG@5'
            },
            'device': 'cpu'
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _test_model(self, model_name, model_config):
        self.config['model'] = model_config
        self.config['model']['name'] = model_name
        
        # Load Data
        dl = DataLoader(self.config)
        
        # Init Model
        model = get_model(model_name, self.config, dl)
        
        # Fit
        model.fit(dl)
        
        # Forward/Predict
        user_ids = torch.tensor([0, 1])
        scores = model.forward(user_ids)
        self.assertEqual(scores.shape, (2, dl.n_items))
        
        # Calc Loss (if applicable)
        batch = {
            'user_id': torch.tensor([0, 1]),
            'pos_item_id': torch.tensor([1, 2]),
            'neg_item_id': torch.tensor([3, 4])
        }
        loss, _ = model.calc_loss(batch)
        print(f"[{model_name}] Loss: {loss.item()}")
        
    def test_cooccurrence(self):
        print("\nTesting CoOccurrence...")
        self._test_model('cooccurrence', {'similarity_metric': 'cosine'})
        
    def test_slim(self):
        print("\nTesting SLIM...")
        self._test_model('slim', {'alpha': 0.1, 'l1_ratio': 0.5, 'max_iter': 5})
        
    def test_elsa(self):
        print("\nTesting ELSA...")
        self._test_model('elsa', {'rank': 8, 'reg_lambda': 0.01})
        
    def test_ultragcn(self):
        print("\nTesting UltraGCN...")
        self._test_model('ultragcn', {'embedding_size': 16, 'ii_neighbor_num': 2})

if __name__ == '__main__':
    unittest.main()
