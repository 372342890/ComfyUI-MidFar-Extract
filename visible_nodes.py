"""Image/text previews for explicit native recognition → PE → image canvas nodes."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .visible_pipeline import prepare_asset, split_prompts
from .visible_batch import process_three
from .selection import selected_matte
from .image_handoff import _diagnostic_path


def _preview(image, prefix, suffix):
    import folder_paths
    output_dir = Path(folder_paths.get_output_directory()).resolve()
    path = _diagnostic_path(prefix, suffix, output_dir)
    pixels = np.round(image[0]*255).astype(np.uint8)
    Image.fromarray(pixels).save(path)
    return {"filename": path.name, "subfolder": str(path.relative_to(output_dir).parent),
            "type": "output"}


class BAMidFarVisiblePE:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "reference": ("IMAGE",),
            "recognition": ("STRING", {"forceInput": True}),
            "pe": ("STRING", {"forceInput": True}),
            "filename_prefix": ("STRING", {"default": "midfar-visible/asset"}),
        }, "optional": {
            "background_mode": (["key", "white"], {"default": "key"}),
        }}

    RETURN_TYPES = ("STRING",)*8
    RETURN_NAMES = ("prompt_1", "prompt_2", "prompt_3", "key_color",
                    "prefix_1", "prefix_2", "prefix_3", "record")
    FUNCTION = "run"
    CATEGORY = "MidFar Extract/Visible PE"
    OUTPUT_NODE = True
    DESCRIPTION = "Shows three actual PE prompts; original image goes directly to native image nodes."

    def run(self, reference, recognition, pe, filename_prefix, background_mode="key"):
        outputs = split_prompts(recognition, pe, reference.detach().cpu().numpy(),
                                filename_prefix, background_mode)
        import folder_paths
        path = _diagnostic_path(filename_prefix, "_PE分路.json", folder_paths.get_output_directory())
        path.write_text(outputs[-1], encoding="utf8")
        # ByteArtist already renders the declared result slots. A Comfy UI
        # payload adds the same prompts again as a second generic preview.
        return outputs


class BAMidFarVisibleAsset:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "reference": ("IMAGE",),
            "recognition": ("STRING", {"forceInput": True}),
            "pe": ("STRING", {"forceInput": True}),
            "asset_index": ("INT", {"default": 1, "min": 1, "max": 3}),
            "filename_prefix": ("STRING", {"default": "midfar-visible/asset"}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("reference_crop", "prompt", "key_color", "filename_prefix", "record")
    FUNCTION = "run"
    CATEGORY = "MidFar Extract/Visible PE"
    OUTPUT_NODE = True
    DESCRIPTION = "Shows the actual target crop and PE prompt. No hidden model calls."

    def run(self, reference, recognition, pe, asset_index, filename_prefix):
        crop, prompt, key, prefix, record = prepare_asset(
            recognition, pe, reference.detach().cpu().numpy(), asset_index, filename_prefix)
        preview = _preview(crop, prefix, "_定位.png")
        return {"ui": {"images": [preview], "text": [prompt, record]},
                "result": (torch.from_numpy(crop), prompt, key, prefix, record)}


class BAMidFarVisiblePNG:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",), "key_color": ("STRING", {"forceInput": True}),
            "filename_prefix": ("STRING", {"forceInput": True}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("rgba", "quality_record")
    FUNCTION = "run"
    CATEGORY = "MidFar Extract/Visible PE"
    OUTPUT_NODE = True
    DESCRIPTION = "Shows and saves one transparent PNG with visible numeric edge checks."

    def run(self, image, key_color, filename_prefix):
        rgba, quality = selected_matte(
            image.detach().cpu().numpy(), key_color, .06, 24, strict=False)
        if quality["warning"] or quality["removed_remote_pixels"]:
            raise ValueError("透明检查未通过：" + (quality["warning"] or
                             f"存在{quality['removed_remote_pixels']}个远离主体的实际碎片像素"))
        preview = _preview(rgba, filename_prefix, ".png")
        quality.update(status="success", png=preview, filename_prefix=filename_prefix)
        record = json.dumps(quality, ensure_ascii=False, indent=2)
        import folder_paths
        _diagnostic_path(filename_prefix, "_透明检查.json",
                         folder_paths.get_output_directory()).write_text(record, encoding="utf8")
        return torch.from_numpy(rgba), record


class BAMidFarVisibleThreePNG:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image_1": ("IMAGE",), "image_2": ("IMAGE",), "image_3": ("IMAGE",),
            "key_color": ("STRING", {"forceInput": True}),
            "prefix_1": ("STRING", {"forceInput": True}),
            "prefix_2": ("STRING", {"forceInput": True}),
            "prefix_3": ("STRING", {"forceInput": True}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("rgba_1", "rgba_2", "rgba_3", "quality_record")
    FUNCTION = "run"
    CATEGORY = "MidFar Extract/Visible PE"
    OUTPUT_NODE = True
    DESCRIPTION = "Shows three distinct white-background or transparent PNGs with duplicate checks."

    def run(self, image_1, image_2, image_3, key_color, prefix_1, prefix_2, prefix_3):
        prefixes = (prefix_1, prefix_2, prefix_3)
        if len(set(prefixes)) != 3:
            raise ValueError("三路输出路径重复，不能覆盖同一素材。")
        sources = [item.detach().cpu().numpy() for item in (image_1, image_2, image_3)]
        outputs, qualities = process_three(sources, key_color)
        import folder_paths
        for output, quality, prefix in zip(outputs, qualities, prefixes):
            preview = _preview(output, prefix, ".png")
            quality.update(png=preview, filename_prefix=prefix)
            _diagnostic_path(prefix, "_透明检查.json",
                             folder_paths.get_output_directory()).write_text(
                                 json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf8")
        record = json.dumps(
            {"status": "warning" if any(q["status"] == "warning" for q in qualities)
             else "success", "png_count": 3, "duplicate_check": "passed",
             "hidden_model_calls": 0, "outputs": qualities}, ensure_ascii=False, indent=2)
        _diagnostic_path(prefixes[0], "_三图记录.json",
                         folder_paths.get_output_directory()).write_text(record, encoding="utf8")
        return *[torch.from_numpy(output) for output in outputs], record
