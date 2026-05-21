"""Gradio demo for MBCTD change detection."""

import argparse
import numpy as np
import torch
import albumentations as A
from PIL import Image
import gradio as gr

from inference import (
    load_model,
    predict_patch,
    infer_patches,
    build_class_map,
    draw_overlay,
    colorize_mask,
    CLASS_ID_TO_NAME,
)
from config import MBCTDConfig


def run_inference(
    before_img: np.ndarray,
    after_img:  np.ndarray,
    threshold: float,
    use_patches: bool,
    full_res: bool,
) -> tuple:
    """Gradio callback: run the model and return (overlay, mask, info_string)."""
    if before_img is None or after_img is None:
        return None, None, "Please upload both images."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = load_model(CHECKPOINT_PATH, device)
    except Exception as e:
        return None, None, f"Failed to load checkpoint: {e}"

    config: MBCTDConfig = model.config
    input_size = config.input_size
    H, W = before_img.shape[:2]
    needs_patching = use_patches and (H > input_size or W > input_size)

    if full_res:
        binary   = predict_patch(before_img, after_img, model, threshold)["binary"]
        mode_str = f"full resolution ({W}×{H})"
    elif needs_patching:
        n_tiles = (
            ((H + input_size - 1) // input_size) *
            ((W + input_size - 1) // input_size)
        )
        binary = infer_patches(model, before_img, after_img, input_size, threshold)
        mode_str = f"patch ({input_size}px tiles, {n_tiles} total)"
    else:
        resizer  = A.Resize(input_size, input_size)
        before_r = resizer(image=before_img)["image"]
        after_r  = resizer(image=after_img)["image"]
        binary   = predict_patch(before_r, after_r, model, threshold)["binary"]
        mode_str = "resize"

    class_map = build_class_map(binary)
    composited = draw_overlay(after_img, class_map)

    h, w = after_img.shape[:2]
    mask_full = np.array(Image.fromarray(class_map).resize((w, h), Image.Resampling.NEAREST))
    mask_rgb  = colorize_mask(mask_full)

    total = class_map.size
    coverage = {
        CLASS_ID_TO_NAME[i]: 100.0 * (class_map == i).sum() / total
        for i in (1, 2, 3, 4)
    }
    coverage_str = " | ".join(f"{k}: {v:.2f}%" for k, v in coverage.items())

    info = (
        f"Device: {device} | Model: MBCTD | "
        f"Threshold: {threshold:.2f} | Mode: {mode_str}\n"
        f"Image: {W}×{H} | {coverage_str}"
    )
    return composited, mask_rgb, info


LEGEND_MD = """
**Legend** (multi-label: a pixel can belong to multiple classes)

| Color | Class |
|-------|-------|
| ⬛ transparent | Background (no class above threshold) |
| 🔵 light blue | Unchanged building |
| 🔴 red | Demolished |
| 🟢 green | New / constructed |
| 🟡 yellow | Replacement (demolished + new) |
"""

parser = argparse.ArgumentParser(description="MBCTD change detection demo")
parser.add_argument("checkpoint", help="Path to model checkpoint (.pth)")
args = parser.parse_args()
CHECKPOINT_PATH = args.checkpoint

with gr.Blocks(title="MBCTD Change Detection") as demo:
    gr.Markdown("# MBCTD — Multi-Label Building Change Type Detection")
    gr.Markdown(
        "Upload a **before** and **after** satellite image "
        "and get change masks overlaid on the after image.\n\n"
        f"**Checkpoint:** `{CHECKPOINT_PATH}`"
    )

    with gr.Row():
        with gr.Column(scale=1):
            before_input = gr.Image(label="Before image", type="numpy")
            after_input  = gr.Image(label="After image",  type="numpy")
            threshold_slider = gr.Slider(
                0.1, 0.9, value=0.70, step=0.05,
                label="Confidence threshold (per-class sigmoid)",
            )
            patch_checkbox = gr.Checkbox(
                label="Patch inference (tile large images at input_size patches)",
                value=False,
            )
            full_res_checkbox = gr.Checkbox(
                label="Full resolution (no resize, no tiling — use with caution on large images)",
                value=False,
            )
            run_btn = gr.Button("Run inference", variant="primary")

        with gr.Column(scale=2):
            overlay_out = gr.Image(label="Overlay on after image", type="numpy")
            mask_out    = gr.Image(label="Predicted mask",          type="numpy")
            info_out    = gr.Textbox(label="Info", interactive=False)
            gr.Markdown(LEGEND_MD)

    patch_checkbox.change(
        fn=lambda checked: gr.update(value=False) if checked else gr.update(),
        inputs=[patch_checkbox],
        outputs=[full_res_checkbox],
    )
    full_res_checkbox.change(
        fn=lambda checked: gr.update(value=False) if checked else gr.update(),
        inputs=[full_res_checkbox],
        outputs=[patch_checkbox],
    )

    run_btn.click(
        fn=run_inference,
        inputs=[before_input, after_input, threshold_slider, patch_checkbox, full_res_checkbox],
        outputs=[overlay_out, mask_out, info_out],
    )

if __name__ == "__main__":
    demo.launch()
