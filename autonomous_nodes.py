"""Scalar ComfyUI wrappers; native ByteArtist nodes perform all model calls."""
import torch

from .autonomous_plan import candidate_review, split_three, asset_reference
from .autonomous_audit import inspect_asset, repair_prompt, choose_asset
from .target_guide import target_guide


def _array(image):
    return image.detach().cpu().numpy()


class BAMidFarCandidateReview:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan": ("STRING", {"forceInput": True}), "reference": ("IMAGE",),
            "edge_margin": ("FLOAT", {"default": .01, "min": 0, "max": .1, "step": .005}),
        }}
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("candidate_sheet", "candidate_manifest")
    FUNCTION = "run"
    CATEGORY = "MidFar Extract/Autonomous"

    def run(self, plan, reference, edge_margin):
        sheet, manifest = candidate_review(plan, _array(reference), edge_margin)
        return torch.from_numpy(sheet), manifest


class BAMidFarThreePlans:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "review": ("STRING", {"forceInput": True}),
            "candidate_manifest": ("STRING", {"forceInput": True}),
            "edge_margin": ("FLOAT", {"default": .01, "min": 0, "max": .1, "step": .005}),
        }}
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("plan_1", "plan_2", "plan_3", "selection_report")
    FUNCTION = "run"
    CATEGORY = "MidFar Extract/Autonomous"

    def run(self, review, candidate_manifest, edge_margin):
        return split_three(review, candidate_manifest, edge_margin)


class BAMidFarAutonomousReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan": ("STRING", {"forceInput": True}), "reference": ("IMAGE",),
            "filename_prefix": ("STRING", {"default": "midfar-autonomous"}),
            "crop_context": ("FLOAT", {"default": .025, "min": 0, "max": .15, "step": .005}),
        }}
    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "reference_crop", "key_color", "filename_prefix", "source_record")
    FUNCTION = "run"
    CATEGORY = "MidFar Extract/Autonomous"

    def run(self, plan, reference, filename_prefix, crop_context):
        prompt, crop, key, filename, record = asset_reference(
            plan, _array(reference), filename_prefix, crop_context)
        return prompt, torch.from_numpy(crop), key, filename, record


class BAMidFarAssetInspect:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",), "reference_crop": ("IMAGE",),
            "plan": ("STRING", {"forceInput": True}),
            "key_color": ("STRING", {"forceInput": True}),
            "softness": ("FLOAT", {"default": .06, "min": 0, "max": .2, "step": .01}),
            "padding": ("INT", {"default": 24, "min": 0, "max": 256}),
        }}
    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("inspection_sheet", "review_prompt", "candidate_rgba", "numeric_report")
    FUNCTION = "run"
    CATEGORY = "MidFar Extract/Autonomous"

    def run(self, image, reference_crop, plan, key_color, softness, padding):
        sheet, prompt, rgba, numeric = inspect_asset(
            _array(image), _array(reference_crop), plan, key_color, softness, padding)
        return torch.from_numpy(sheet), prompt, torch.from_numpy(rgba), numeric


class BAMidFarTargetGuide:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "reference_crop": ("IMAGE",), "plan": ("STRING", {"forceInput": True}),
            "localization": ("STRING", {"forceInput": True}),
            "key_color": ("STRING", {"forceInput": True}),
        }}
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("guide_image", "guide_prompt", "guide_record")
    FUNCTION = "run"
    CATEGORY = "MidFar Extract/Autonomous"

    def run(self, reference_crop, plan, localization, key_color):
        image, prompt, record = target_guide(_array(reference_crop), plan, localization, key_color)
        return torch.from_numpy(image), prompt, record


class BAMidFarRepairPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan": ("STRING", {"forceInput": True}),
            "key_color": ("STRING", {"forceInput": True}),
            "numeric_report": ("STRING", {"forceInput": True}),
            "visual_report": ("STRING", {"forceInput": True}),
            "attempt": ("INT", {"default": 1, "min": 1, "max": 2}),
        }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("repair_prompt",)
    FUNCTION = "run"
    CATEGORY = "MidFar Extract/Autonomous"

    def run(self, plan, key_color, numeric_report, visual_report, attempt):
        return (repair_prompt(plan, key_color, numeric_report, visual_report, attempt),)


class BAMidFarChooseAsset:
    @classmethod
    def INPUT_TYPES(cls):
        required = {}
        for i in range(1, 4):
            required[f"rgba_{i}"] = ("IMAGE",)
            required[f"numeric_{i}"] = ("STRING", {"forceInput": True})
            required[f"visual_{i}"] = ("STRING", {"forceInput": True})
        required["min_score"] = ("INT", {"default": 85, "min": 85, "max": 100})
        return {"required": required}
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("rgba", "selection_record")
    FUNCTION = "run"
    CATEGORY = "MidFar Extract/Autonomous"

    def run(self, rgba_1, numeric_1, visual_1, rgba_2, numeric_2, visual_2,
            rgba_3, numeric_3, visual_3, min_score):
        rgba, record = choose_asset(
            [_array(rgba_1), _array(rgba_2), _array(rgba_3)],
            [numeric_1, numeric_2, numeric_3],
            [visual_1, visual_2, visual_3], min_score)
        return torch.from_numpy(rgba), record
