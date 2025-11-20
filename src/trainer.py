import torch
import torch.optim as optim
import yaml
import os
import numpy as np
from tqdm import tqdm
import collections
import json
import time
import copy

from .evaluation import evaluate_metrics  # 함수 임포트
from .utils import plot_results

def _convert_numpy_types_to_python_types(obj):
    """
    Numpy 타입(예: np.float64)을 Python 기본 타입(예: float)으로 재귀적으로 변환합니다.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_numpy_types_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_types_to_python_types(elem) for elem in obj]
    else:
        return obj

class Trainer:
    """
    추천 모델의 학습 과정을 관리하는 클래스.
    """
    def __init__(self, config, model, data_loader):
        self.config = config
        self.model = model
        self.data_loader = data_loader
        self.device = model.device
        self.model.to(self.device)

        model_name = self.config['model']['name']
        dataset_name = self.config['dataset_name']
        run_name = self.config.get('run_name')
        
        base_path = os.path.join('trained_model', dataset_name)
        
        if run_name and run_name != 'default':
            experiment_folder_name = f"{model_name}__{run_name}"
            self.output_path = os.path.join(base_path, experiment_folder_name)
        else:
            self.output_path = os.path.join(base_path, model_name)
            
        os.makedirs(self.output_path, exist_ok=True)

        with open(os.path.join(self.output_path, 'config.yaml'), 'w') as f:
            config_to_save = copy.deepcopy(self.config)
            config_to_save.pop('run_name', None)
            yaml.dump(config_to_save, f, default_flow_style=False)

        # BUG FIX: 학습 관련 설정은 'train' 블록이 있을 때만 초기화
        if 'train' in self.config:
            self.optimizer = self._get_optimizer(self.config['train']['optimizer'])
            self.epochs = self.config['train']['epochs']
            self.early_stop_patience = self.config['train']['early_stop_patience']
            
            self.train_losses = collections.defaultdict(list)
            self.eval_metrics = collections.defaultdict(list)
            self.tracked_params = collections.defaultdict(list)

            self.best_metric_value = -float('inf')
            self.patience_counter = 0
            
            print("Initializing data loaders for training...")
            self.train_loader = self.data_loader.get_train_loader(self.config['train']['batch_size'])
            self.validation_loader = self.data_loader.get_validation_loader(self.config['train']['batch_size'] * 2)
            print("Data loaders initialized.")

    def _get_optimizer(self, optimizer_name):
        # 이 메소드는 self.config['train']이 존재할 때만 호출됨
        lr = self.config['train']['lr']
        weight_decay = self.config['train']['l2_regularization']
        if optimizer_name.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def train(self):
        # train이 호출되면, 'train' config이 있다고 가정
        if 'train' not in self.config:
            print("[Error] 'train' configuration not found. Cannot start training.")
            return

        print(f"Training started on device: {self.device}")
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_losses = collections.defaultdict(float)
            epoch_params = collections.defaultdict(list)

            for batch_data in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                batch_data = {k: v.to(self.device) for k, v in batch_data.items()}

                if self.config['train'].get('loss_type') == 'pairwise' and self.config['train'].get('use_hard_negatives', False):
                    with torch.no_grad():
                        num_hard_negatives = self.config['train'].get('num_hard_negatives', 1)
                        hard_neg_items = self.model.sample_hard_negatives(
                            batch_data['user_id'], 
                            batch_data['pos_item_id'],
                            num_hard_negatives
                        )
                        batch_data['neg_item_id'] = hard_neg_items

                self.optimizer.zero_grad()
                losses_tuple, params_to_log = self.model.calc_loss(batch_data)

                total_loss_for_backward = sum(losses_tuple)
                total_loss_for_backward.backward()
                self.optimizer.step()

                epoch_losses['total_loss'] += total_loss_for_backward.item()
                for i, l in enumerate(losses_tuple):
                    epoch_losses[f'loss_{i}'] += l.item()
                
                if params_to_log:
                    for k, v in params_to_log.items():
                        epoch_params[k].append(v)

            avg_total_loss = epoch_losses['total_loss'] / len(self.train_loader)
            self.train_losses['total_loss'].append(avg_total_loss)
            avg_main_loss = epoch_losses['loss_0'] / len(self.train_loader) if 'loss_0' in epoch_losses else 0.0
            self.train_losses['loss_0'].append(avg_main_loss)
            for k, v in epoch_losses.items():
                if k != 'total_loss' and k != 'loss_0':
                    self.train_losses[k].append(v / len(self.train_loader))

            if epoch_params:
                for k, v_list in epoch_params.items():
                    self.tracked_params[k].append(np.mean(v_list))
            
            eval_config_for_val = self.config.get('evaluation', {})
            main_metric = eval_config_for_val.get('main_metric', 'NDCG')
            main_metric_k = eval_config_for_val.get('main_metric_k', 10)
            
            # [최적화] 미리 생성된 검증 로더를 전달
            current_metrics = self.evaluate(
                loader=self.validation_loader,
                is_final_evaluation=False
            )
            
            for k, v in current_metrics.items():
                self.eval_metrics[k].append(v)
            
            main_metric_name = f"{main_metric}@{main_metric_k}"
            main_metric_value = current_metrics.get(main_metric_name, 0.0)
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_total_loss:.4f} (Main: {avg_main_loss:.4f}) - Val {main_metric_name}: {main_metric_value:.4f}")

            if self._check_early_stopping(current_metrics):
                break
        
        print("Training finished. Performing final evaluation...")
        self.evaluate(is_final_evaluation=True)
        
        self._visualize_results()

    def evaluate(self, loader=None, is_final_evaluation=False):
        self.model.eval()
        
        eval_config = self.config.get('evaluation', {})
        
        if loader is None:
            print("Creating a new loader for evaluation...")
            batch_size = self.config.get('train', {}).get('batch_size', 512) * 2
            if is_final_evaluation:
                loader = self.data_loader.get_final_loader(batch_size)
            else: 
                loader = self.data_loader.get_validation_loader(batch_size)

        # 평가에 사용할 설정 결정
        if is_final_evaluation:
            # 최종 평가 시에는 YAML에 정의된 evaluation 블록을 그대로 사용
            config_for_eval = eval_config
        else: # Validation during training
            # 학습 중 검증 시에는 main_metric만 계산
            config_for_eval = {
                'method': eval_config.get('validation_method', 'uni99'),
                'metrics': [eval_config.get('main_metric', 'NDCG')],
                'top_k': [eval_config.get('main_metric_k', 10)]
            }

        current_metrics = evaluate_metrics(self.model, self.data_loader, config_for_eval, self.device, test_loader=loader)
        
        if is_final_evaluation:
            print(f"Final Evaluation: {current_metrics}")
            metrics_path = os.path.join(self.output_path, 'final_metrics.json')
            dumpable_metrics = _convert_numpy_types_to_python_types(current_metrics)
            with open(metrics_path, 'w') as f:
                json.dump(dumpable_metrics, f, indent=4)
            print(f"Final metrics saved to {metrics_path}")

        return current_metrics

    def _check_early_stopping(self, current_metrics):
        eval_config = self.config.get('evaluation', {})
        main_metric = eval_config.get('main_metric', 'NDCG')
        main_metric_k = eval_config.get('main_metric_k', 10)
        main_metric_name = f"{main_metric}@{main_metric_k}"
        current_main_metric = current_metrics.get(main_metric_name)

        if current_main_metric is None:
            print(f"[Warning] Main metric '{main_metric_name}' not found. Skipping early stopping.")
            return False

        if current_main_metric > self.best_metric_value:
            self.best_metric_value = current_main_metric
            self.patience_counter = 0
            self._save_checkpoint()
        else:
            self.patience_counter += 1
        
        if self.patience_counter >= self.early_stop_patience:
            print(f"Early stopping triggered at epoch {len(self.train_losses['total_loss'])}")
            return True
        return False

    def _save_checkpoint(self):
        checkpoint_file = os.path.join(self.output_path, "best_model.pt")
        torch.save(self.model.state_dict(), checkpoint_file)
        print(f"Best performance updated. Checkpoint saved to {checkpoint_file}")

    def _visualize_results(self):
        if not hasattr(self, 'train_losses') or not self.train_losses:
            return 
            
        losses_history_path = os.path.join(self.output_path, 'losses_history.json')
        with open(losses_history_path, 'w') as f:
            json.dump(_convert_numpy_types_to_python_types(dict(self.train_losses)), f, indent=4)

        metrics_history_path = os.path.join(self.output_path, 'metrics_history.json')
        with open(metrics_history_path, 'w') as f:
            json.dump(_convert_numpy_types_to_python_types(dict(self.eval_metrics)), f, indent=4)

        params_history_path = os.path.join(self.output_path, 'params_history.json')
        with open(params_history_path, 'w') as f:
            json.dump(_convert_numpy_types_to_python_types(dict(self.tracked_params)), f, indent=4)

        plot_results(
            data_dict={'total_loss': self.train_losses['total_loss']}, 
            title="Total Training Loss", 
            xlabel="Epoch", 
            ylabel="Loss",
            file_path=os.path.join(self.output_path, "total_loss_plot.png")
        )
        
        loss_plot_dict = {k: v for k, v in self.train_losses.items() if k.startswith('loss_')}
        plot_results(
            data_dict=loss_plot_dict, 
            title="Training Losses", 
            xlabel="Epoch", 
            ylabel="Loss",
            secondary_y_keys=[k for k in loss_plot_dict.keys() if k != 'loss_0'], 
            file_path=os.path.join(self.output_path, "loss_plot.png")
        )
        
        if self.tracked_params:
            plot_results(
                data_dict=self.tracked_params, 
                title="Tracked Parameters", 
                xlabel="Epoch", 
                ylabel="Value",
                file_path=os.path.join(self.output_path, "params_plot.png")
            )
        
        if self.eval_metrics:
            plot_results(
                data_dict=self.eval_metrics, 
                title="Validation Metrics", 
                xlabel="Epoch", 
                ylabel="Metric Score",
                file_path=os.path.join(self.output_path, "metrics_plot.png")
            )
