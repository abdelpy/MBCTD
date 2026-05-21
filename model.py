"""
MBCTD — Multi-Label Building Change Type Detection

Multi-label segmentation producing 3 independent binary masks
(unchanged, demolished, new). A pixel may belong to more than one
mask (e.g. replacements where demolished and new overlap).

Architecture:
1. Siamese ConvNeXt encoder (shared weights for before/after)
2. Single-stream change fusion at each scale: 1x1 conv over
   [before, after, before-after, |before-after|]
3. U-Net decoder with PixelShuffle upsampling and full-resolution
   skips drawn from the raw input images
4. 1x1 head producing 3 logits per pixel
"""

import torch
import torch.nn as nn
import timm
from typing import List, Optional, Tuple

from config import MBCTDConfig


class ConvNeXtSiameseEncoder(nn.Module):
    """Siamese ConvNeXt encoder for change detection.

    Returns features at 1/4, 1/8, 1/16, 1/32 of input resolution.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.feature_dims = [128, 256, 512, 1024]
        self.backbone = timm.create_model(
            "convnext_base.dinov3_lvd1689m",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

    def forward(
        self, before: torch.Tensor, after: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        feats_before = self.backbone(before)
        feats_after = self.backbone(after)
        return list(zip(feats_before, feats_after))


class ChangeFusion(nn.Module):
    """Fuses before/after features by concatenating [before, after, diff, |diff|] and projecting to out_channels."""

    def __init__(self, in_channels: int, out_channels: Optional[int] = None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.out_channels = out_channels

    def forward(self, feat_before: torch.Tensor, feat_after: torch.Tensor) -> torch.Tensor:
        diff = feat_before - feat_after
        x = torch.cat([feat_before, feat_after, diff, diff.abs()], dim=1)
        return self.fuse(x)


class PixelShuffleUpsample(nn.Module):
    """Learned 2x upsampling via PixelShuffle."""

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pixel_shuffle(self.conv(x))))


class DecoderBlock(nn.Module):
    """PixelShuffle 2x upsample + concat skip + two 3x3 conv refinement."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = PixelShuffleUpsample(in_channels, in_channels, scale_factor=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class FullResolutionDecoder(nn.Module):
    """U-Net decoder with encoder skips at 1/4–1/32 and high-res skips at 1/2 and 1/1 from the raw input pair."""

    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int],
        num_classes: int = 3,
        highres_channels: Tuple[int, int] = (64, 32),
    ):
        super().__init__()
        assert len(encoder_channels) == len(decoder_channels), \
            "encoder_channels and decoder_channels must have the same length"

        self.num_stages = len(encoder_channels)

        # High-res skips from raw input pair
        self.highres_1x = nn.Sequential(
            nn.Conv2d(6, highres_channels[1], 3, padding=1),
            nn.BatchNorm2d(highres_channels[1]),
            nn.GELU(),
            nn.Conv2d(highres_channels[1], highres_channels[1], 3, padding=1),
            nn.BatchNorm2d(highres_channels[1]),
            nn.GELU(),
        )
        self.highres_2x = nn.Sequential(
            nn.Conv2d(6, highres_channels[0], 3, stride=2, padding=1),
            nn.BatchNorm2d(highres_channels[0]),
            nn.GELU(),
            nn.Conv2d(highres_channels[0], highres_channels[0], 3, padding=1),
            nn.BatchNorm2d(highres_channels[0]),
            nn.GELU(),
        )

        # Project deepest encoder features
        self.initial_proj = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], decoder_channels[0], 1),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.GELU(),
        )

        # Encoder-scale decoder blocks: 1/32 -> 1/16 -> 1/8 -> 1/4
        self.decoder_blocks = nn.ModuleList()
        for i in range(self.num_stages - 1):
            in_ch = decoder_channels[i]
            skip_ch = encoder_channels[-(i + 2)]
            out_ch = decoder_channels[i + 1]
            self.decoder_blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))

        # High-res blocks: 1/4 -> 1/2 -> 1/1
        self.highres_block_2x = DecoderBlock(decoder_channels[-1], highres_channels[0], decoder_channels[-1])
        self.highres_block_1x = DecoderBlock(decoder_channels[-1], highres_channels[1], decoder_channels[-1])

        # Final head: 3 independent logits per pixel
        self.mask_head = nn.Conv2d(decoder_channels[-1], num_classes, 1)

    def forward(
        self,
        encoder_features: List[torch.Tensor],
        before: torch.Tensor,
        after: torch.Tensor,
    ) -> torch.Tensor:
        input_concat = torch.cat([before, after], dim=1)
        skip_1x = self.highres_1x(input_concat)
        skip_2x = self.highres_2x(input_concat)

        x = self.initial_proj(encoder_features[-1])
        for i, block in enumerate(self.decoder_blocks):
            skip = encoder_features[-(i + 2)]
            x = block(x, skip)

        x = self.highres_block_2x(x, skip_2x)
        x = self.highres_block_1x(x, skip_1x)

        return self.mask_head(x)


class MBCTD(nn.Module):
    """ConvNeXt-based change detection model. Returns (B, num_classes, H, W) logits."""

    def __init__(self, config: MBCTDConfig):
        super().__init__()
        self.config = config

        self.encoder = ConvNeXtSiameseEncoder(pretrained=config.pretrained)

        encoder_dims = config.encoder_dims
        fusion_out = config.fusion_channels
        self.fusion_modules = nn.ModuleList([
            ChangeFusion(dim, fusion_out) for dim in encoder_dims
        ])
        fused_dims = [m.out_channels for m in self.fusion_modules]

        self.decoder = FullResolutionDecoder(
            encoder_channels=fused_dims,
            decoder_channels=config.decoder_channels,
            num_classes=config.num_classes,
        )

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        feature_pairs = self.encoder(before, after)
        fused = [self.fusion_modules[i](b, a) for i, (b, a) in enumerate(feature_pairs)]
        return self.decoder(fused, before, after)

    def predict_multilabel(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Per-channel binary masks at the given probability threshold."""
        logits = self.forward(before, after)
        return (torch.sigmoid(logits) > threshold).float()

    def predict_probs(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        """Per-channel sigmoid probabilities (independent)."""
        return torch.sigmoid(self.forward(before, after))

    @classmethod
    def from_config(cls, config: MBCTDConfig) -> "MBCTD":
        return cls(config)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: Optional[MBCTDConfig] = None,
        map_location: str = "cpu",
    ) -> "MBCTD":
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        if config is None:
            config = checkpoint.get("config") or MBCTDConfig()
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model


def create_model(input_size: int = 256, pretrained: bool = True, **kwargs) -> MBCTD:
    config = MBCTDConfig(input_size=input_size, pretrained=pretrained, **kwargs)
    return MBCTD(config)
