"""
PyTorch implementation of a linear oscillatory state-space model.

Key components:
- LinOSSLayer: Single state-space layer with implicit (IM) discretization (Eq. 3-4)
- LinOSSBlock: Layer with normalization, gating, and residual connections
- LinOSS: Full model with stacked blocks

Example usage:
    >>> from linoss_pytorch import LinOSS
    >>> model = LinOSS(input_dim=32, output_dim=32, ssm_size=64, feature_dim=64, num_blocks=2)
    >>> x = torch.randn(100, 32)  # Batch of 100 sequences, 32 features
    >>> y = model(x)  # Output shape: [100, 32]
"""

from .linoss_pytorch import (
    LinOSS,
    LinOSSBlock,
    LinOSSLayer,
    apply_linoss_im,
    apply_linoss_imex,
    LinearAttention,
    binary_operator,
)

__version__ = "0.1.0"
__all__ = [
    "LinOSS",
    "LinOSSBlock", 
    "LinOSSLayer",
    "apply_linoss_im",
    "apply_linoss_imex",
    "LinearAttention",
    "binary_operator",
]
