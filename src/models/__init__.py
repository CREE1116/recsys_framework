# 모델 이름과 클래스를 매핑하는 레지스트리
# main.py에서 config 파일의 모델 이름을 기반으로 동적으로 클래스를 가져오는 데 사용됩니다.

from .mf import MF
from .lightgcn import LightGCN
from .CSAR import CSAR
from .CSAR_R import CSAR_R
from .CSAR_R_Softmax import CSAR_R_Softmax
from .CSAR_contrastive import CSAR_contrastive
from .CSAR_BPR import CSAR_BPR
from .CSAR_Deep import CSAR_Deep
from .most_popular import MostPopular
from .item_knn import ItemKNN
from .neumf import NeuMF
from .rerank_wrapper import ReRankWrapper

MODEL_REGISTRY = {
    'mf': MF,
    'lightgcn': LightGCN,
    'csar' : CSAR,
    'csar-r' : CSAR_R,
    'csar-r-softmax' : CSAR_R_Softmax,
    'csar-contrastive' : CSAR_contrastive,
    'csar-bpr' : CSAR_BPR,
    'csar-deep' : CSAR_Deep,
    'most-popular': MostPopular,
    'ItemKNN': ItemKNN,
    'NeuMF': NeuMF,
    'ReRankWrapper': ReRankWrapper,
}

def get_model(model_name, config, data_loader):
    """
    모델 이름에 해당하는 모델 클래스의 인스턴스를 생성하여 반환합니다.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class(config, data_loader)
