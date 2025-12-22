import torch
import torch.nn as nn
from ..base_model import BaseModel

class RandomRec(BaseModel):
    """
    무작위로 아이템을 추천하는 베이스라인 모델.
    이 모델은 학습되지 않습니다.
    """
    def __init__(self, config, data_loader):
        super(RandomRec, self).__init__(config, data_loader)
        self.n_items = self.data_loader.n_items
        print("RandomRec model initialized.")

    def fit(self, data_loader):
        """
        RandomRec 모델은 학습이 필요하지 않습니다.
        """
        print("RandomRec model fitted successfully (no-op).")
        return

    def forward(self, users):
        """
        모든 사용자에 대해 [0, 1) 범위의 랜덤 점수를 반환합니다.
        """
        batch_size = users.size(0)
        # Random scores for all items
        return torch.rand(batch_size, self.n_items).to(self.device)

    def predict_for_pairs(self, user_ids, item_ids):
        """
        주어진 사용자-아이템 쌍에 대해 랜덤 점수를 반환합니다.
        """
        return torch.rand(len(user_ids)).to(self.device)

    def get_embeddings(self):
        """
        RandomRec 모델은 임베딩이 없습니다.
        """
        return None, None

    def get_final_item_embeddings(self):
        """
        RandomRec uses random scores, so no meaningful embeddings.
        Return random or zero for placeholder.
        """
        return torch.rand(self.n_items, 64, device=self.device)

    def calc_loss(self, batch_data):
        """
        이 모델은 학습되지 않으므로 손실은 0입니다.
        """
        return (torch.tensor(0.0, device=self.device),), None

    def __str__(self):
        return "RandomRec"
