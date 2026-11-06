"""Deterministic grid cutting nodes for existing ComfyUI image pipelines."""
import json

import numpy as np
import torch

from .grid_core import split_sheet


class GridAssetSplitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "rows": ("INT", {"default": 0, "min": 0, "max": 20}),
                "columns": ("INT", {"default": 0, "min": 0, "max": 20}),
                "background_tolerance": ("INT", {"default": 24, "min": 1, "max": 128}),
                "min_content_ratio": ("FLOAT", {
                    "default": 0.0003, "min": 0.0, "max": 0.1, "step": 0.0001}),
                "border_ignore_ratio": ("FLOAT", {
                    "default": 0.025, "min": 0.0, "max": 0.1, "step": 0.005}),
                "trim_grid_lines": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "asset_list": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("assets", "kept_count", "report")
    FUNCTION = "split"
    CATEGORY = "image/asset_sheet"
    DESCRIPTION = "Cut row-specific asset-sheet cells and skip blanks; original pixels retained."

    def split(self, image, rows=0, columns=0, background_tolerance=24,
              min_content_ratio=0.0003, border_ignore_ratio=0.025,
              trim_grid_lines=True, asset_list=""):
        if image.ndim != 4 or image.shape[-1] not in (3, 4) or image.shape[0] < 1:
            raise ValueError("Expected non-empty IMAGE tensor [batch,height,width,channels]")
        count = 0
        if asset_list and asset_list.strip():
            text = asset_list.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                raise ValueError("asset_list must contain a JSON array, for example #69 text output")
            count = len(parsed)
            if not 1 <= count <= 20:
                raise ValueError("asset_list must contain between 1 and 20 assets")
        # Detect on uint8, then crop the original floating-point source pixels.
        original = image.detach().cpu()
        detection = (original.clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        output_source = original[..., :3]
        if original.shape[-1] == 4:
            alpha = original[..., 3:4].clamp(0, 1)
            output_source = output_source * alpha + (1 - alpha)
        pieces, reports = [], []
        for batch_index, pixels in enumerate(detection):
            _, report = split_sheet(
                pixels, rows, columns, count, background_tolerance,
                min_content_ratio, border_ignore_ratio, trim_grid_lines,
            )
            report["source_batch_index"] = batch_index
            reports.append(report)
            for record in report["cells"]:
                if record["kept"]:
                    x0, y0, x1, y1 = record["box"]
                    pieces.append(output_source[batch_index, y0:y1, x0:x1])
        if pieces:
            max_h = max(p.shape[0] for p in pieces)
            max_w = max(p.shape[1] for p in pieces)
            result = torch.ones((len(pieces), max_h, max_w, 3), dtype=original.dtype)
            for i, piece in enumerate(pieces):
                y = (max_h - piece.shape[0]) // 2
                x = (max_w - piece.shape[1]) // 2
                result[i, y:y + piece.shape[0], x:x + piece.shape[1]] = piece
            for i, record in enumerate(r for report in reports
                                       for r in report["cells"] if r["kept"]):
                record["output_batch_index"] = i
                record["padding_top"] = (max_h - pieces[i].shape[0]) // 2
                record["padding_left"] = (max_w - pieces[i].shape[1]) // 2
        else:
            result = torch.empty((0, original.shape[1], original.shape[2], 3),
                                 dtype=original.dtype)
        return result.to(device=image.device), len(pieces), json.dumps(
            {"sources": reports, "total_kept": len(pieces)}, ensure_ascii=False, indent=2
        )


class AutoGridSplit:
    """One image socket in, every non-empty crop in one IMAGE batch out."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "split"
    CATEGORY = "image/asset_sheet"
    DESCRIPTION = "输入一张网格图，逐行识别贯通格子线并排除灰色主体误切；支持不等列宽、行高和窄空白格。"

    def split(self, image):
        images, _, _ = GridAssetSplitter().split(image=image)
        return (images,)
