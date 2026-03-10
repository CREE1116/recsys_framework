import torch
import torch.optim as optim
import yaml
import os
import numpy as np
from tqdm import tqdm
import collections
import json
import copy
from .evaluation import evaluate_metrics
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

        self._cache_registry = getattr(model, 'cache_registry', None)
        self.output_path = model.output_path
        os.makedirs(self.output_path, exist_ok=True)

        self.best_metric_value = -float('inf')
        self.best_val_metrics = {}
        self.best_epoch = -1

        with open(os.path.join(self.output_path, 'config.yaml'), 'w', encoding='utf-8') as f:
            config_to_save = copy.deepcopy(self.config)
            config_to_save.pop('run_name', None)
            yaml.dump(config_to_save, f, default_flow_style=False)

        self.is_trainable = 'train' in self.config
        self.train_losses = collections.defaultdict(list)
        self.eval_metrics = collections.defaultdict(list)
        self.tracked_params = collections.defaultdict(list)

        if self.is_trainable:
            self.epochs = self.config['train'].get('epochs', 0)
            self.early_stop_patience = self.config['train'].get('early_stop_patience', 10)
            
            if 'optimizer' in self.config['train']:
                self.optimizer = self._get_optimizer(self.config['train']['optimizer'])
            else:
                self.optimizer = None

            self.patience_counter = 0
            
            print("Initializing data loaders for training...")
            self.train_loader = self.data_loader.get_train_loader(self.config['train']['batch_size'])
            self.validation_loader = self.data_loader.get_validation_loader(self.config['train']['batch_size'] * 2)
            print("Data loaders initialized.")

            self.use_amp = self.config['train'].get('use_amp', True) and self.device.type in ['cuda', 'mps']
            self.scaler = torch.amp.GradScaler(self.device.type) if self.use_amp else None

    def _get_optimizer(self, optimizer_name):
        lr = self.config['train'].get('lr', self.config['train'].get('learning_rate'))
        if lr is None:
            raise ValueError("Learning rate not found in config. Please specify 'lr' or 'learning_rate'.")
        lr = float(lr)

        weight_decay = float(self.config['train'].get('weight_decay', 0.0))

        # nn.Embedding params use per-batch L2 reg; exclude from weight_decay
        no_decay_params = []
        decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'embedding' in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

        if optimizer_name.lower() == 'adam':
            return optim.Adam(param_groups, lr=lr)
        elif optimizer_name.lower() == 'sgd':
            return optim.SGD(param_groups, lr=lr)
        elif optimizer_name.lower() == 'adamw':
            return optim.AdamW(param_groups, lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def run(self):
        """
        통합 실행 오케스트레이터: fit → train loop → final evaluation
        - fit()이 있으면 먼저 호출 (EASE, ItemKNN, UltraGCN 등)
        - train 설정이 있으면 SGD 학습 루프 실행
        - 없으면 비학습 모델로 간주하고 바로 최종 평가
        """
        best_model_path = os.path.join(self.output_path, "best_model.pt")
        if self.config.get('skip_fit', False) and os.path.exists(best_model_path):
            print(f"Loading cached model from {best_model_path} and skipping fit/train.")
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        elif hasattr(self.model, 'fit'):
            print(f"Model has 'fit' method. Calling fit()...")
            self.model.fit(self.data_loader)

        if self._cache_registry:
            self._cache_registry.log_status()

        if self.is_trainable:
            return self._train_loop()
        else:
            print("Non-trainable model. Proceeding directly to evaluation...")
            return self.evaluate(is_final_evaluation=not self.config.get('hpo_mode', False))

    def _train_loop(self):
        print(f"Training started on device: {self.device}")

        for epoch in range(self.epochs):
            if hasattr(self.model, 'on_epoch_start'):
                self.model.on_epoch_start(epoch)

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

                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    losses_tuple, params_to_log = self.model.calc_loss(batch_data)
                    total_loss_for_backward = sum(losses_tuple)

                if self.scaler:
                    self.scaler.scale(total_loss_for_backward).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total_loss_for_backward.backward()
                    self.optimizer.step()

                if hasattr(self.model, 'on_step_end'):
                    self.model.on_step_end()

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

            current_metrics = self.evaluate(
                loader=self.validation_loader,
                is_final_evaluation=False
            )

            for k, v in current_metrics.items():
                self.eval_metrics[k].append(v)

            main_metric_name = f"{main_metric}@{main_metric_k}"
            main_metric_value = current_metrics.get(main_metric_name, 0.0)
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_total_loss:.4f} (Main: {avg_main_loss:.4f}) - Val {main_metric_name}: {main_metric_value:.4f}")

            if self._check_early_stopping(current_metrics, epoch + 1):
                break
        
        print("Training finished. Loading best model for final evaluation...")
        best_model_path = os.path.join(self.output_path, "best_model.pt")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            print(f"Best model loaded from {best_model_path}")
        else:
            print("[Warning] Best model checkpoint not found. Using last epoch model.")

        if self.config.get('hpo_mode', False):
            print("[HPO-Mode] Skipping TEST evaluation. Using best validation metrics.")
            if self.best_val_metrics:
                return self.best_val_metrics
            return self.evaluate(is_final_evaluation=False, update_best_snapshot=True)

        current_metrics = self.evaluate(is_final_evaluation=True)
        self._visualize_results()
        return current_metrics

    def evaluate(self, loader=None, is_final_evaluation=False, update_best_snapshot=False):
        self.model.eval()

        eval_config = self.config.get('evaluation', {})

        if loader is None:
            print("Creating a new loader for evaluation...")
            eval_batch_size = eval_config.get('batch_size')
            if eval_batch_size is None:
                eval_batch_size = self.config.get('train', {}).get('batch_size', 512) * 2

            if is_final_evaluation:
                loader = self.data_loader.get_final_loader(eval_batch_size)
            else:
                loader = self.data_loader.get_validation_loader(eval_batch_size)

        if is_final_evaluation:
            config_for_eval = eval_config.copy()
            config_for_eval['method'] = eval_config.get('final_method', 'full')
        else:
            config_for_eval = {
                'method': eval_config.get('validation_method', 'full'),
                'metrics': [eval_config.get('main_metric', 'NDCG')],
                'top_k': [eval_config.get('main_metric_k', 10)]
            }

        current_metrics = evaluate_metrics(self.model, self.data_loader, config_for_eval, self.device, test_loader=loader, is_final=is_final_evaluation)
        
        if not is_final_evaluation and update_best_snapshot:
            # [추가] 검증 지표 스냅샷 업데이트 (비학습 모델이나 명시적 검증 호출 대응)
            eval_config_val = self.config.get('evaluation', {})
            m_name = f"{eval_config_val.get('main_metric', 'NDCG')}@{eval_config_val.get('main_metric_k', 10)}"
            m_val = current_metrics.get(m_name, 0.0)
            if m_val >= self.best_metric_value:
                self.best_metric_value = m_val
                if not self.is_trainable or self.best_epoch == -1:
                    self.best_epoch = 0
                self.best_val_metrics = copy.deepcopy(current_metrics)
                self._save_val_metrics(current_metrics)
                self._save_checkpoint()

        if is_final_evaluation:
            if not self.best_val_metrics:
                print("No validation metrics found. Running validation before final evaluation...")
                self.evaluate(is_final_evaluation=False, update_best_snapshot=True)

            print(f"Final Evaluation (Test): {current_metrics}")

            if hasattr(self, 'best_val_metrics') and self.best_val_metrics:
                for k, v in self.best_val_metrics.items():
                    current_metrics[f"val_{k}"] = v

            metrics_path = os.path.join(self.output_path, 'final_metrics.json')
            dumpable_metrics = _convert_numpy_types_to_python_types(current_metrics)
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(dumpable_metrics, f, indent=4)
            print(f"Final metrics saved to {metrics_path}")
        elif self.config.get('hpo_mode', False):
            metrics_path = os.path.join(self.output_path, 'final_metrics.json')
            dumpable_metrics = _convert_numpy_types_to_python_types(current_metrics)
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(dumpable_metrics, f, indent=4)

        return current_metrics

    def _save_val_metrics(self, metrics):
        val_metrics_path = os.path.join(self.output_path, 'val_metrics.json')
        dumpable_val = _convert_numpy_types_to_python_types(metrics)
        dumpable_val['best_epoch'] = self.best_epoch
        with open(val_metrics_path, 'w', encoding='utf-8') as f:
            json.dump(dumpable_val, f, indent=4)

    def _check_early_stopping(self, current_metrics, epoch):
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
            self.best_epoch = epoch
            self.best_val_metrics = copy.deepcopy(current_metrics)
            self.patience_counter = 0
            self._save_val_metrics(current_metrics)
            self._save_checkpoint()
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.early_stop_patience:
            print(f"Early stopping triggered at epoch {len(self.train_losses['total_loss'])}")
            return True
        return False

    def _save_checkpoint(self):
        os.makedirs(self.output_path, exist_ok=True)
        checkpoint_file = os.path.join(self.output_path, "best_model.pt")
        torch.save(self.model.state_dict(), checkpoint_file)
        print(f"Best performance updated. Checkpoint saved to {checkpoint_file}")

    def _visualize_results(self):
        if not hasattr(self, 'train_losses') or not self.train_losses:
            return 
            
        losses_history_path = os.path.join(self.output_path, 'losses_history.json')
        os.makedirs(self.output_path, exist_ok=True)
        with open(losses_history_path, 'w', encoding='utf-8') as f:
            json.dump(_convert_numpy_types_to_python_types(dict(self.train_losses)), f, indent=4)

        metrics_history_path = os.path.join(self.output_path, 'metrics_history.json')
        with open(metrics_history_path, 'w', encoding='utf-8') as f:
            json.dump(_convert_numpy_types_to_python_types(dict(self.eval_metrics)), f, indent=4)

        params_history_path = os.path.join(self.output_path, 'params_history.json')
        with open(params_history_path, 'w', encoding='utf-8') as f:
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


