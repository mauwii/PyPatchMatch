"""Gradio GUI for PyPatchMatch: draw a mask, then inpaint."""

from __future__ import annotations

import gradio as gr
import numpy as np

import patchmatch


def editor_mask(value: dict) -> np.ndarray:
    """Union of all editor layers; every pixel with any alpha is a hole (hard edges)."""
    mask = np.zeros(value["background"].shape[:2], dtype=bool)
    for layer in value["layers"] or []:
        mask |= layer[..., 3] > 0
    return mask


def on_inpaint(editor: dict | None, patch_size: int) -> np.ndarray:
    if editor is None or editor["background"] is None:
        raise gr.Error("Load an image first.")
    if not patchmatch.patchmatch_available:
        raise gr.Error("The PyPatchMatch native library is not available.")
    mask = editor_mask(editor)
    if not mask.any():
        gr.Warning("The mask is empty.")
    return patchmatch.inpaint(
        editor["background"][..., :3], mask, patch_size=int(patch_size)
    )


with gr.Blocks(title="PyPatchMatch GUI") as demo:
    with gr.Row():
        editor = gr.ImageEditor(
            label="Original + mask",
            type="numpy",
            format="png",
            brush=gr.Brush(colors=["#ff00ff"], color_mode="fixed"),
        )
        result = gr.Image(label="Result", type="numpy", format="png", interactive=False)
    with gr.Row():
        patch_size = gr.Slider(1, 15, value=3, step=1, label="patch_size")
        inpaint_btn = gr.Button("Inpaint", variant="primary", scale=3)

    gr.on([editor.upload, editor.clear], lambda: None, None, result)
    inpaint_btn.click(on_inpaint, [editor, patch_size], result)

if __name__ == "__main__":
    demo.launch()
