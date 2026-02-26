"""
场景一致性特征提取与编码模块
"""
from .scene_encoder import SceneEncoder, SceneFeatureExtractor
from .Semantic_Illusion import extract_semantic_feature
from .Geo_consistency import extract_geometric_feature
from .Lighting_shadow_anomaly import extract_lighting_feature
from .Layout import extract_layout_feature

__all__ = [
    'SceneEncoder',
    'SceneFeatureExtractor',
    'extract_semantic_feature',
    'extract_geometric_feature',
    'extract_lighting_feature',
    'extract_layout_feature'
]
