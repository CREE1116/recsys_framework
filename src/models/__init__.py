# 모델 이름과 클래스를 매핑하는 레지스트리
# main.py에서 config 파일의 모델 이름을 기반으로 동적으로 클래스를 가져오는 데 사용됩니다.

# General Models
from .general.mf import MF
from .general.lightgcn import LightGCN
from .general.most_popular import MostPopular
from .general.random_rec import RandomRec
from .general.item_knn import ItemKNN
from .general.neumf import NeuMF
from .general.ACF_NLL import ACF_NLL, ACF_BPR
from .general.softplusmf import softplusMF
from .general.ease import EASE
from .general.norm_ease import NormEASE
from .general.protomf import ProtoMF
from .general.macr import MACR
from .general.pure_svd import PureSVD
from .general.naive_bayes import NaiveBayes
from .general.cooccurrence import CoOccurrence
from .general.slim import SLIM
from .general.elsa import ELSA
from .general.ultragcn import UltraGCN
from .general.simgcl import SimGCL

# CSAR Models
from .csar.CSAR_Basic import CSAR_Basic
from .csar.MinimalCSAR import MinimalCSAR
from .csar.ClosedCSAR import ClosedCSAR


from .csar.CSAR import CSAR
from .csar.CSAR_BPR import CSAR_BPR
from .csar.LIRA import LIRA
from .csar.LightLIRA import LightLIRA
from .csar.DNILIRA import DNILIRA
from .csar.SVD_DNILIRA import SVD_DNILIRA
from .csar.DLIRA import DLIRA
from .csar.SMFLIRA import SMFLIRA
from .csar.SpectralDiffusionLIRA import SpectralDiffusionLIRA
from .csar.SymmetricMSLIRA import SymmetricMSLIRA


from .general.sansa import SANSA
from .general.rlae import RLAE
from .general.infinity_ae import Infinity_AE
from .general.svd_ae import SVD_AE
from .general.ials import iALS

MODEL_REGISTRY = {
    # General Models
    'mf': MF,
    'lightgcn': LightGCN,
    'most-popular': MostPopular,
    'ItemKNN': ItemKNN,
    'itemknn': ItemKNN,
    'NeuMF': NeuMF,
    'neumf': NeuMF,
    'ease': EASE,
    'norm_ease': NormEASE,
    'sansa': SANSA,   # Registered
    'rlae': RLAE,     # Registered
    'infinity_ae': Infinity_AE, # Registered
    'svd_ae': SVD_AE, # Registered
    'ials': iALS,     # Registered
    'random-rec': RandomRec,
    'naive-bayes': NaiveBayes,
    'pure-svd': PureSVD,
    'acf-nll': ACF_NLL,
    'acf-bpr': ACF_BPR,
    'softplusmf': softplusMF,
    'macr': MACR,
    'protomf': ProtoMF,
    'cooccurrence': CoOccurrence,
    'slim': SLIM,
    'elsa': ELSA,
    'ultragcn': UltraGCN,
    'rlae': RLAE,
    'simgcl': SimGCL,
    
    # CSAR Models
    'csar_basic': CSAR_Basic,
    'csar_bpr': CSAR_BPR,
    'csar': CSAR,
    'lira': LIRA,
    'light_lira': LightLIRA,
    'minimal_csar': MinimalCSAR,
    'closed_csar': ClosedCSAR,
    'dnilira': DNILIRA,
    'svd_dnilira': SVD_DNILIRA,
    'dlira': DLIRA,
    'smflira': SMFLIRA,
    'spectral_diffusion_lira': SpectralDiffusionLIRA,
    'symmetric_ms_lira': SymmetricMSLIRA,
}

def get_model(model_name, config, data_loader):
    """
    모델 이름에 해당하는 모델 클래스의 인스턴스를 생성하여 반환합니다.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class(config, data_loader)
