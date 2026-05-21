"""
MBCTD — Multi-Label Building Change Type Detection

ConvNeXt-based model that predicts three independent per-pixel masks
(unchanged, demolished, new). Pixels can belong to more than one mask
(e.g. replacements where demolished and new overlap).
"""

import torch
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MBCTDConfig:
    """Configuration for MBCTD."""

    # Device
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

    # ============== Model Architecture ==============

    convnext_weights: str = "convnext_base.dinov3_lvd1689m"

    convnext_dims: List[int] = field(default_factory=lambda: [128, 256, 512, 1024])

    input_size: int = 256

    pretrained: bool = True

    # Number of output channels (3 building classes, no background)
    # 0=unchanged, 1=demolished, 2=new
    # Each channel is an independent binary mask (overlap allowed for replacements).
    num_classes: int = 3

    # Multi-label only. Kept for backward-compat reads; ignored by new code.
    multi_label: bool = True

    # Decoder channels at each upsampling stage
    decoder_channels: List[int] = field(default_factory=lambda: [256, 128, 64, 32])

    # Fused channels in the per-scale change-fusion module (single shared stream).
    # If None, defaults to encoder dim per stage.
    fusion_channels: Optional[int] = None

    # Decision threshold for predict_multilabel.
    pred_threshold: float = 0.5

    # ============== Checkpoint ==============

    checkpoint_path: str = ""
    strict_load: bool = False

    @property
    def encoder_dims(self) -> List[int]:
        return self.convnext_dims

    @property
    def encoder_weights(self) -> str:
        return self.convnext_weights


# Backward-compat alias for checkpoints pickled under the old class name.
BCTDConfig = MBCTDConfig


def get_config(**kwargs) -> MBCTDConfig:
    return MBCTDConfig(**kwargs)
