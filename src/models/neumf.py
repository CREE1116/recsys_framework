import torch
import torch.nn as nn
from .base_model import BaseModel
from src.loss import BPRLoss

class NeuMF(BaseModel):
    """
    Neural Matrix Factorization 모델.
    GMF (Generalized Matrix Factorization)와 MLP (Multi-Layer Perceptron) 경로를 결합합니다.
    """
    def __init__(self, config, data_loader):
        super(NeuMF, self).__init__(config, data_loader)

        self.embedding_dim_gmf = config['model']['embedding_dim_gmf']
        self.embedding_dim_mlp = config['model']['embedding_dim_mlp']
        self.mlp_layers = config['model']['mlp_layers']
        
        self.n_users = self.data_loader.n_users
        self.n_items = self.data_loader.n_items

        # GMF 임베딩
        self.user_embedding_gmf = nn.Embedding(self.n_users, self.embedding_dim_gmf)
        self.item_embedding_gmf = nn.Embedding(self.n_items, self.embedding_dim_gmf)

        # MLP 임베딩
        self.user_embedding_mlp = nn.Embedding(self.n_users, self.embedding_dim_mlp)
        self.item_embedding_mlp = nn.Embedding(self.n_items, self.embedding_dim_mlp)

        # MLP 레이어
        mlp_modules = []
        input_size = self.embedding_dim_mlp * 2
        for hidden_size in self.mlp_layers:
            mlp_modules.append(nn.Linear(input_size, hidden_size))
            mlp_modules.append(nn.ReLU())
            input_size = hidden_size
        self.mlp_layers_module = nn.Sequential(*mlp_modules)

        # 최종 예측 레이어
        # GMF 출력 차원 + MLP 출력 차원
        predict_input_dim = self.embedding_dim_gmf + self.mlp_layers[-1]
        self.predict_layer = nn.Linear(predict_input_dim, 1)

        self.loss_fn = BPRLoss()
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)
        
        for m in self.mlp_layers_module:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

    def forward(self, users):
        """
        주어진 사용자들에 대한 모든 아이템의 점수를 계산합니다.
        """
        batch_size = users.size(0)
        all_items = torch.arange(self.n_items, device=self.device)
        
        # 모든 (사용자, 아이템) 쌍 생성
        user_ids_repeated = users.repeat_interleave(self.n_items)
        item_ids_tiled = all_items.repeat(batch_size)
        
        # predict_for_pairs를 사용하여 모든 쌍의 점수 계산
        scores = self.predict_for_pairs(user_ids_repeated, item_ids_tiled)
        
        # 결과를 [batch_size, n_items] 형태로 재구성
        return scores.view(batch_size, self.n_items)

    def predict_for_pairs(self, users, items):
        """
        주어진 (사용자, 아이템) 쌍에 대한 점수를 계산합니다.
        """
        # GMF 경로
        user_embed_gmf = self.user_embedding_gmf(users)
        item_embed_gmf = self.item_embedding_gmf(items)
        gmf_output = user_embed_gmf * item_embed_gmf

        # MLP 경로
        user_embed_mlp = self.user_embedding_mlp(users)
        item_embed_mlp = self.item_embedding_mlp(items)
        mlp_input = torch.cat((user_embed_mlp, item_embed_mlp), dim=-1)
        mlp_output = self.mlp_layers_module(mlp_input)

        # GMF와 MLP 출력 결합
        concat_output = torch.cat((gmf_output, mlp_output), dim=-1)
        
        # 최종 점수
        prediction = self.predict_layer(concat_output)
        return prediction.squeeze(-1)

    def get_embeddings(self):
        """
        GMF와 MLP 임베딩을 결합하여 최종 사용자 및 아이템 임베딩을 반환합니다.
        evaluation.py에서 ILD 계산 등에 사용됩니다.
        """
        final_user_embeddings = torch.cat((self.user_embedding_gmf.weight, self.user_embedding_mlp.weight), dim=1)
        final_item_embeddings = torch.cat((self.item_embedding_gmf.weight, self.item_embedding_mlp.weight), dim=1)
        return final_user_embeddings, final_item_embeddings

    def get_final_item_embeddings(self):
        """
        [호환성 유지] NeuMF의 최종 아이템 임베딩을 반환합니다.
        get_embeddings() 사용이 권장됩니다.
        """
        _, item_embeds = self.get_embeddings()
        return item_embeds.detach()

    def calc_loss(self, batch_data):
        users = batch_data['user_id'].squeeze(-1)
        pos_items = batch_data['pos_item_id'].squeeze(-1)
        neg_items = batch_data['neg_item_id'].squeeze(-1)

        pos_scores = self.predict_for_pairs(users, pos_items)
        neg_scores = self.predict_for_pairs(users, neg_items)

        loss = self.loss_fn(pos_scores, neg_scores)
        return (loss,), None

    def __str__(self):
        return f"NeuMF(gmf_dim={self.embedding_dim_gmf}, mlp_dim={self.embedding_dim_mlp}, layers={self.mlp_layers})"
