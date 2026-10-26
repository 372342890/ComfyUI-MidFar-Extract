"""Scalar ComfyUI wrappers; native ByteArtist nodes perform all model calls."""
import json
import torch

from .autonomous_plan import candidate_review, split_three, asset_reference
from .autonomous_audit import inspect_asset, repair_prompt, choose_asset
from .target_guide import target_guide
from .foot_completion import completion_context, complete_foot
from .image_handoff import native_image_file, save_inspection, save_review_records
from .localization_view import localization_view


def _array(image):
    return image.detach().cpu().numpy()


class BAMidFarNativeImageFile:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("image_file",)
    FUNCTION = "run"
    CATEGORY = "MidFar Extract/Autonomous"

    def run(self, image):
        import folder_paths
        return (native_image_file(_array(image), folder_paths.get_input_directory()),)


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


class BAMidFarLocalizationView:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"reference_crop": ("IMAGE",),
                             "source_record": ("STRING", {"forceInput": True})}}
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("marked_crop", "localization_prompt", "localization_record")
    FUNCTION = "run"
    CATEGORY = "MidFar Extract/Autonomous"

    def run(self, reference_crop, source_record):
        image, prompt, record = localization_view(_array(reference_crop), source_record)
        return torch.from_numpy(image), prompt, record


class BAMidFarAssetInspect:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",), "reference_crop": ("IMAGE",),
            "plan": ("STRING", {"forceInput": True}),
            "key_color": ("STRING", {"forceInput": True}),
            "softness": ("FLOAT", {"default": .06, "min": 0, "max": .2, "step": .01}),
            "padding": ("INT", {"default": 24, "min": 0, "max": 256}),
        }, "optional": {"target_guide": ("IMAGE",),
                        "identity_record": ("STRING", {"default": "", "forceInput": True}),
                        "diagnostic_prefix": ("STRING", {"default": "", "forceInput": True}),
                        "version": ("INT", {"default": 1, "min": 1, "max": 3})}}
    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("inspection_sheet", "review_prompt", "candidate_rgba", "numeric_report")
    FUNCTION = "run"
    CATEGORY = "MidFar Extract/Autonomous"

    def run(self, image, reference_crop, plan, key_color, softness, padding, target_guide=None,
            diagnostic_prefix="", version=1, identity_record=""):
        sheet, prompt, rgba, numeric = inspect_asset(
            _array(image), _array(reference_crop), plan, key_color, softness, padding,
            None if target_guide is None else _array(target_guide), identity_record)
        if diagnostic_prefix:
            import folder_paths
            save_inspection(_array(image), sheet, rgba, numeric, prompt, diagnostic_prefix,
                            version, folder_paths.get_output_directory())
        return torch.from_numpy(sheet), prompt, torch.from_numpy(rgba), numeric


class BAMidFarTargetGuide:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "reference_crop": ("IMAGE",), "plan": ("STRING", {"forceInput": True}),
            "localization": ("STRING", {"forceInput": True}),
            "key_color": ("STRING", {"forceInput": True}),
        }, "optional": {"identity_record": ("STRING", {"default": "", "forceInput": True})}}
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "IMAGE", "MASK")
    RETURN_NAMES = ("guide_image", "guide_prompt", "guide_record", "completion_guide", "foot_mask")
    FUNCTION = "run"
    CATEGORY = "MidFar Extract/Autonomous"

    def run(self, reference_crop, plan, localization, key_color, identity_record=""):
        image, prompt, record = target_guide(
            _array(reference_crop), plan, localization, key_color, identity_record)
        marked, mask = completion_context(image, key_color)
        return torch.from_numpy(image), prompt, record, torch.from_numpy(marked), torch.from_numpy(mask)


class BAMidFarCompleteFoot:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",), "source_guide": ("IMAGE",), "foot_mask": ("MASK",),
            "key_color": ("STRING", {"forceInput": True}),
        }}
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("completed_image", "completion_record")
    FUNCTION = "run"
    CATEGORY = "MidFar Extract/Autonomous"

    def run(self, image, source_guide, foot_mask, key_color):
        completed, record = complete_foot(_array(image), _array(source_guide),
                                         _array(foot_mask), key_color)
        return torch.from_numpy(completed), json.dumps(record)


class BAMidFarRepairPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan": ("STRING", {"forceInput": True}),
            "key_color": ("STRING", {"forceInput": True}),
            "numeric_report": ("STRING", {"forceInput": True}),
            "visual_report": ("STRING", {"forceInput": True}),
            "attempt": ("INT", {"default": 1, "min": 1, "max": 2}),
        }, "optional": {"identity_record": ("STRING", {"default": "", "forceInput": True})}}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("repair_prompt",)
    FUNCTION = "run"
    CATEGORY = "MidFar Extract/Autonomous"

    def run(self, plan, key_color, numeric_report, visual_report, attempt, identity_record=""):
        return (repair_prompt(plan, key_color, numeric_report, visual_report, attempt,
                              identity_record),)


class BAMidFarChooseAsset:
    @classmethod
    def INPUT_TYPES(cls):
        required = {}
        for i in range(1, 4):
            required[f"rgba_{i}"] = ("IMAGE",)
            required[f"numeric_{i}"] = ("STRING", {"forceInput": True})
            required[f"visual_{i}"] = ("STRING", {"forceInput": True})
        required["min_score"] = ("INT", {"default": 85, "min": 85, "max": 100})
        return {"required": required, "optional": {
            "diagnostic_prefix": ("STRING", {"default": "", "forceInput": True})}}
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("rgba", "selection_record")
    FUNCTION = "run"
    CATEGORY = "MidFar Extract/Autonomous"

    def run(self, rgba_1, numeric_1, visual_1, rgba_2, numeric_2, visual_2,
            rgba_3, numeric_3, visual_3, min_score, diagnostic_prefix=""):
        if diagnostic_prefix:
            import folder_paths
            save_review_records(diagnostic_prefix, [numeric_1, numeric_2, numeric_3],
                                [visual_1, visual_2, visual_3], folder_paths.get_output_directory())
        rgba, record = choose_asset(
            [_array(rgba_1), _array(rgba_2), _array(rgba_3)],
            [numeric_1, numeric_2, numeric_3],
            [visual_1, visual_2, visual_3], min_score)
        return torch.from_numpy(rgba), record
