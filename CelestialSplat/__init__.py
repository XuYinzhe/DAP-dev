"""
CelestialSplat: Feed-forward 360° Gaussian Splatting with Cross-View Attention

A transformer-based sparse 360° panorama 3D Gaussian Splatting reconstruction system.
"""

from .model import (
    CelestialSplat,
    CelestialSplatConfig,
    DAPFeatureAdapter,
    CrossViewTransformer,
    GSDecoder,
    GaussianFusion,
    build_celestial_splat
)

try:
    from .integrate_dap import (
        load_dap_model,
        build_celestial_splat_with_dap,
        get_training_strategy
    )
    HAS_DAP_INTEGRATION = True
except ImportError:
    HAS_DAP_INTEGRATION = False

try:
    from .dataset import (
        SequenceChunkDataset,
        TartanAir360Dataset,
        create_dataloaders,
    )
    HAS_DATASET = True
except ImportError:
    HAS_DATASET = False

__version__ = "0.2.0"
__all__ = [
    "CelestialSplat",
    "CelestialSplatConfig",
    "DAPFeatureAdapter",
    "CrossViewTransformer", 
    "GSDecoder",
    "GaussianFusion",
    "build_celestial_splat"
]

if HAS_DAP_INTEGRATION:
    __all__.extend([
        "load_dap_model",
        "build_celestial_splat_with_dap",
        "get_training_strategy"
    ])

if HAS_DATASET:
    __all__.extend([
        "SequenceChunkDataset",
        "TartanAir360Dataset",
        "create_dataloaders",
    ])
