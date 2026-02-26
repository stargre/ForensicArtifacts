"""
信号自然性特征提取与编码模块
"""
from .signal_encoder import SignalEncoder, SignalFeatureExtractor
from .Local_Spectral import extract_spectral_feature
from .Laplacian import extract_resampling_feature
from .JPEG import extract_jpeg_feature

__all__ = [
    'SignalEncoder',
    'SignalFeatureExtractor',
    'extract_spectral_feature',
    'extract_resampling_feature',
    'extract_jpeg_feature'
]