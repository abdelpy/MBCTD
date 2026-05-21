"""
Core inference utilities for MBCTD building change detection.

The model is multi-label: each pixel can independently belong to any of
  channel 0 = unchanged
  channel 1 = demolished
  channel 2 = new
A pixel may belong to more than one (replacement = demolished AND new).

Visual encoding:
  background  (no class above threshold)  – transparent
  unchanged                                – light blue
  demolished                               – red
  new                                      – green
  replacement (demolished + new)           – yellow
"""

import numpy as np
import torch
import albumentations as A
from PIL import Image

from config import MBCTDConfig
from model import MBCTD

CLASS_COLORS = {
    "background":  (0,   0,   0,   0  ),
    "unchanged":   (100, 200, 255, 160),
    "demolished":  (220, 50,  50,  180),
    "new":         (50,  205, 50,  180),
    "replacement": (255, 200, 0,   200),
}

# class_map id -> name. Priority order when collapsing multi-label to a single id:
# replacement > demolished > new > unchanged > background.
CLASS_ID_TO_NAME = {
    0: "background",
    1: "unchanged",
    2: "demolished",
    3: "new",
    4: "replacement",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_model_cache: dict = {}   # keyed by checkpoint path
_normalize = A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def load_model(checkpoint_path: str, device: torch.device) -> MBCTD:
    """Load and cache the MBCTD model from a checkpoint file."""
    key = str(checkpoint_path)
    if key in _model_cache:
        return _model_cache[key]

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", MBCTDConfig())
    config.device = device

    model = MBCTD(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    _model_cache[key] = model
    return model


def _to_tensor(img_np: np.ndarray) -> torch.Tensor:
    """ImageNet normalise only (no resize) -> (1, 3, H, W)."""
    out = _normalize(image=img_np)["image"]
    return torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)


def logits_to_binary(logits: torch.Tensor, threshold: float) -> np.ndarray:
    """(1, 3, H, W) raw logits -> (3, H, W) uint8 binary masks."""
    probs = torch.sigmoid(logits[0])
    return (probs > threshold).cpu().numpy().astype(np.uint8)


def build_class_map(binary: np.ndarray) -> np.ndarray:
    """
    Reduce (3, H, W) binary masks to (H, W) uint8 class ids
    using priority: replacement > demolished > new > unchanged > background.
    """
    unchanged  = binary[0].astype(bool)
    demolished = binary[1].astype(bool)
    new        = binary[2].astype(bool)

    class_map = np.zeros(unchanged.shape, dtype=np.uint8)
    class_map[unchanged]        = 1
    class_map[new]              = 3
    class_map[demolished]       = 2
    class_map[demolished & new] = 4
    return class_map


def draw_overlay(after_rgb: np.ndarray, class_map: np.ndarray) -> np.ndarray:
    """
    Blend colored class masks over the after image.

    after_rgb  : (H, W, 3) uint8 original after image (any size)
    class_map  : (H_model, W_model) uint8 class ids
    Returns    : (H, W, 3) uint8 composited image
    """
    h, w = after_rgb.shape[:2]

    class_img = Image.fromarray(class_map).resize((w, h), Image.Resampling.NEAREST)
    class_map_full = np.array(class_img)

    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    for cls_id, name in CLASS_ID_TO_NAME.items():
        mask = class_map_full == cls_id
        overlay[mask] = CLASS_COLORS[name]

    alpha = overlay[..., 3:4].astype(np.float32) / 255.0
    after_f = after_rgb.astype(np.float32)
    over_f  = overlay[..., :3].astype(np.float32)
    result  = (after_f * (1 - alpha) + over_f * alpha).clip(0, 255).astype(np.uint8)
    return result


def colorize_mask(class_map: np.ndarray) -> np.ndarray:
    """Return a solid-color (H, W, 3) visualization of the class map."""
    h, w = class_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, name in CLASS_ID_TO_NAME.items():
        r, g, b, _ = CLASS_COLORS[name]
        rgb[class_map == cls_id] = (r, g, b)
    return rgb


def predict_patch(
    before_patch: np.ndarray,
    after_patch: np.ndarray,
    model: MBCTD,
    threshold: float = 0.7,
) -> dict:
    """
    Run change detection on a single before/after image patch.

    Designed for 256×256 inputs (the model's native resolution), but accepts
    any spatial size — no resizing is applied. Use load_model() to get a model instance.

    Parameters
    ----------
    before_patch : (H, W, 3) uint8 RGB array
    after_patch  : (H, W, 3) uint8 RGB array
    model        : loaded MBCTD instance
    threshold    : per-class sigmoid threshold (default 0.7)

    Returns
    -------
    dict:
        binary    : (3, H, W) uint8 – per-class binary masks (unchanged / demolished / new)
        class_map : (H, W) uint8   – collapsed class ids (see CLASS_ID_TO_NAME)
        overlay   : (H, W, 3) uint8 – class colors blended over after_patch
        mask_rgb  : (H, W, 3) uint8 – solid-color class visualization
    """
    device   = next(model.parameters()).device
    before_t = _to_tensor(before_patch).to(device)
    after_t  = _to_tensor(after_patch).to(device)
    with torch.no_grad():
        logits = model(before_t, after_t)
    binary    = logits_to_binary(logits, threshold)
    class_map = build_class_map(binary)
    return {
        "binary":    binary,
        "class_map": class_map,
        "overlay":   draw_overlay(after_patch, class_map),
        "mask_rgb":  colorize_mask(class_map),
    }


def infer_patches(
    model: MBCTD,
    before_img: np.ndarray,
    after_img: np.ndarray,
    patch_size: int,
    threshold: float,
) -> np.ndarray:
    """
    Split before/after into non-overlapping patch_size×patch_size tiles,
    run inference on each, stitch (3, H, W) binary masks back to original resolution.

    Images are reflected-padded to the next multiple of patch_size so every
    tile is exactly patch_size×patch_size.  Padding is cropped after stitching.
    """
    H, W = before_img.shape[:2]

    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    def pad(img):
        return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    before_p = pad(before_img)
    after_p  = pad(after_img)
    H_p, W_p = before_p.shape[:2]

    binary = np.zeros((3, H_p, W_p), dtype=np.uint8)

    for y in range(0, H_p, patch_size):
        for x in range(0, W_p, patch_size):
            pb = before_p[y : y + patch_size, x : x + patch_size]
            pa = after_p [y : y + patch_size, x : x + patch_size]
            binary[:, y : y + patch_size, x : x + patch_size] = (
                predict_patch(pb, pa, model, threshold)["binary"]
            )

    return binary[:, :H, :W]
