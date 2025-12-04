# 모델 이름과 클래스를 매핑하는 레지스트리
# main.py에서 config 파일의 모델 이름을 기반으로 동적으로 클래스를 가져오는 데 사용됩니다.

from .mf import MF
from .lightgcn import LightGCN
from .CSAR import CSAR
from .CSAR_R import CSAR_R
from .CSAR_R_Softmax import CSAR_R_Softmax
# from .CSAR_contrastive import CSAR_contrastive
from .CSAR_BPR import CSAR_BPR
from .CSAR_Deep import CSAR_Deep
from .most_popular import MostPopular
from .item_knn import ItemKNN
from .neumf import NeuMF
from .rerank_wrapper import ReRankWrapper
from .CSAR_R_BPR import CSAR_R_BPR
from .CSAR_R_contrastive import CSAR_R_contrastive
from .CSAR_R_Confidence import CSAR_R_Confidence
from .CSAR_R_Lasso import CSAR_R_Lasso
from .CSAR_Lasso import CSAR_Lasso
from .CSAR_STE import CSAR_STE
from .CSAR_R_L0 import CSAR_R_L0
from .CSAR_R_KD import CSAR_R_KD
from .CSAR_R_UBR import CSAR_R_UBR
from .CSAR_R_CCBPR import CSAR_R_CCBPR
from .ACF_NLL import ACF_NLL, ACF_BPR
from .CSAR_V_BPR import CSAR_V_BPR
from .CSAR_V import CSAR_V
from .CSAR_VR import CSAR_VR


MODEL_REGISTRY = {
    'mf': MF,
    'lightgcn': LightGCN,
    'csar' : CSAR,
    'csar-r' : CSAR_R,
    'csar-r-softmax' : CSAR_R_Softmax,
    # 'csar-contrastive' : CSAR_contrastive,
    'csar-bpr' : CSAR_BPR,
    'csar-deep' : CSAR_Deep,
    'most-popular': MostPopular,
    'ItemKNN': ItemKNN,
    'NeuMF': NeuMF,
    'ReRankWrapper': ReRankWrapper,
    'csar-r-bpr': CSAR_R_BPR,
    'csar-r-contrastive': CSAR_R_contrastive,
    'csar-r-confidence': CSAR_R_Confidence,
    'csar-r-lasso': CSAR_R_Lasso,
    'csar-lasso': CSAR_Lasso,
    'csar-ste': CSAR_STE,
    'csar-r-l0': CSAR_R_L0,
    'csar-r-kd': CSAR_R_KD,
    'csar-r-ubr': CSAR_R_UBR,
    'csar-r-ccbpr': CSAR_R_CCBPR,
    'acf-nll': ACF_NLL,
    'acf-bpr': ACF_BPR,
    'csar-v-bpr': CSAR_V_BPR,
    'csar-v': CSAR_V,
    'csar-vr': CSAR_VR
}

def get_model(model_name, config, data_loader):
    """
    모델 이름에 해당하는 모델 클래스의 인스턴스를 생성하여 반환합니다.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class(config, data_loader)
