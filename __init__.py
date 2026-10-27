import json
import torch
from .core import parse_plan, choose_key, compile_prompt, matte
from .objects import parse_objects, object_prompt, object_filename, crop_asset, expand_workflow
from .terrain import parse_assets, asset_prompt, asset_filename, expand_terrain_workflow
from .selection import parse_selection, crop_reference, selected_prompt, expand_selection, selected_matte
from .autonomous_nodes import (
    BAMidFarCandidateReview, BAMidFarThreePlans, BAMidFarAutonomousReference,
    BAMidFarAssetInspect, BAMidFarRepairPrompt, BAMidFarChooseAsset, BAMidFarTargetGuide,
    BAMidFarCompleteFoot, BAMidFarNativeImageFile, BAMidFarLocalizationView,
)
from .three_d_nodes import BAMidFar3DModel, BAMidFar3DGenerate, BAMidFarGLBExport
from .compact_nodes import BAMidFarCompactExtract
from .reference_nodes import BAMidFarReferencePlan, BAMidFarReferenceExtract, BAMidFarReferenceGLB
from .visible_nodes import (
    BAMidFarVisibleAsset, BAMidFarVisiblePNG, BAMidFarVisiblePE, BAMidFarVisibleThreePNG,
)


class BAMidFarPlanPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan": ("STRING", {"forceInput": True}),
            "reference": ("IMAGE",),
            "layer": (["midground", "background"],),
            "key_mode": (["auto", "green", "magenta", "blue"],),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "key_color", "plan_summary")
    FUNCTION = "run"
    CATEGORY = "MidFar Extract"
    DESCRIPTION = "Compile a scene-specific extraction prompt and choose a low-conflict chroma key."

    def run(self, plan, reference, layer, key_mode):
        parsed = parse_plan(plan)
        key = choose_key(reference.detach().cpu().numpy(), key_mode)
        return compile_prompt(parsed, layer, key), key, json.dumps(parsed, ensure_ascii=False)


class BAMidFarMatte:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "key_color": ("STRING", {"forceInput": True}),
            "softness": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 0.2, "step": 0.01}),
            "despill": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("rgba", "foreground_mask", "preview", "quality")
    FUNCTION = "run"
    CATEGORY = "MidFar Extract"
    DESCRIPTION = "Soft chroma matte with foreground color recovery; MASK 1=opaque. Save rgba directly."

    def run(self, image, key_color, softness, despill):
        rgba, mask, preview, quality = matte(image.detach().cpu().numpy(), key_color, softness, despill)
        return torch.from_numpy(rgba), torch.from_numpy(mask), torch.from_numpy(preview), quality


class BAMidFarObjectPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan": ("STRING", {"forceInput": True}),
            "reference": ("IMAGE",),
            "layer": (["both", "midground", "background"],),
            "max_objects": ("INT", {"default": 8, "min": 1, "max": 32}),
            "key_mode": (["auto", "green", "magenta", "blue"],),
            "filename_prefix": ("STRING", {"default": "midfar-objects"}),
        }}

    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("object_prompt", "reference", "key_color", "filename_prefix", "manifest")
    OUTPUT_IS_LIST = (True, True, True, True, False)
    FUNCTION = "run"
    CATEGORY = "MidFar Extract"
    DESCRIPTION = "Dynamic individual instances: one list item per object, never one item per scenery layer."

    def run(self, plan, reference, layer, max_objects, key_mode, filename_prefix):
        parsed = parse_objects(plan, layer, max_objects)
        key = choose_key(reference.detach().cpu().numpy(), key_mode)
        objects = parsed["objects"]
        return ([object_prompt(parsed, o, key) for o in objects],
                [reference for _ in objects], [key for _ in objects],
                [object_filename(o, filename_prefix) for o in objects],
                json.dumps(parsed, ensure_ascii=False))


class BAMidFarObjectMatte:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "key_color": ("STRING", {"forceInput": True}),
            "filename_prefix": ("STRING", {"forceInput": True}),
            "softness": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 0.2, "step": 0.01}),
            "padding": ("INT", {"default": 24, "min": 0, "max": 256}),
            "strict": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("rgba", "preview", "filename_prefix", "quality")
    FUNCTION = "run"
    CATEGORY = "MidFar Extract"
    DESCRIPTION = "One RGBA asset, cropped without rescaling. Strict mode rejects empty/multiple/border-cut results."

    def run(self, image, key_color, filename_prefix, softness, padding, strict):
        source = image.detach().cpu().numpy()
        if source.shape[0] != 1:
            raise ValueError("Expected exactly one generated image per object; disable multiple variations.")
        rgba, _, _, matte_quality = matte(source, key_color, softness, True)
        rgba, quality = crop_asset(rgba, padding)
        quality["filename_prefix"] = filename_prefix
        quality["matte"] = json.loads(matte_quality)[0]
        if strict and quality["warning"]:
            raise ValueError(filename_prefix + ": " + quality["warning"])
        alpha = rgba[..., 3:4]
        yy, xx = torch.meshgrid(torch.arange(rgba.shape[1]), torch.arange(rgba.shape[2]), indexing="ij")
        check = torch.where(((xx//24 + yy//24) % 2) == 0, .72, .92).numpy()[None, ..., None]
        preview = rgba[..., :3]*alpha + check*(1-alpha)
        return (torch.from_numpy(rgba), torch.from_numpy(preview.astype("float32")),
                filename_prefix, json.dumps(quality, ensure_ascii=False))


class BAMidFarWorkflowExpand:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan": ("STRING", {"forceInput": True}),
            "workflow_template": ("STRING", {"multiline": True}),
            "reference_filename": ("STRING", {"default": "reference.png"}),
            "layer": (["both", "midground", "background"],),
            "max_objects": ("INT", {"default": 8, "min": 1, "max": 32}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("workflow_json", "manifest")
    FUNCTION = "run"
    CATEGORY = "MidFar Extract"
    DESCRIPTION = "ByteArtist compatibility: expand one independent image-generation carrier per detected object."

    def run(self, plan, workflow_template, reference_filename, layer, max_objects):
        parsed = parse_objects(plan, layer, max_objects)
        template = json.loads(workflow_template)
        workflow = expand_workflow(parsed, template, reference_filename)
        return json.dumps(workflow, ensure_ascii=False), json.dumps(parsed, ensure_ascii=False)


class BATerrainAssetPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan": ("STRING", {"multiline": True}),
            "style_reference": ("IMAGE",),
            "shape_reference": ("IMAGE",),
            "key_mode": (["auto", "green", "magenta", "blue"],),
            "filename_prefix": ("STRING", {"default": "terrain-assets"}),
        }}

    RETURN_TYPES = ("STRING", "IMAGE", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "style_reference", "shape_reference", "key_color", "filename_prefix", "manifest")
    FUNCTION = "run"
    CATEGORY = "MidFar Extract/Terrain Assets"
    DESCRIPTION = "One natural object. Image 1 controls style, image 2 controls silhouette and proportions."

    def run(self, plan, style_reference, shape_reference, key_mode, filename_prefix):
        parsed = parse_assets(plan)
        if parsed["total"] != 1:
            raise ValueError("Use BATerrainWorkflowExpand first: exactly one asset per image carrier.")
        target = parsed["assets"][0]
        # Both references participate in key selection, even when resolutions differ.
        import numpy as np
        samples = []
        for ref in (style_reference, shape_reference):
            pixels = ref.detach().cpu().numpy()[..., :3].reshape(-1, 3)
            samples.append(pixels[::max(1, len(pixels)//100000)])
        key = choose_key(np.concatenate(samples)[None, None, :, :], key_mode)
        return (asset_prompt(parsed, target, key), style_reference, shape_reference, key,
                asset_filename(target, filename_prefix), json.dumps(parsed, ensure_ascii=False))


class BATerrainWorkflowExpand:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan": ("STRING", {"forceInput": True}),
            "workflow_template": ("STRING", {"multiline": True}),
            "style_filename": ("STRING", {"default": "style-reference.png"}),
            "shape_filename": ("STRING", {"default": "shape-reference.png"}),
            "max_assets": ("INT", {"default": 6, "min": 1, "max": 32}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("workflow_json", "manifest")
    FUNCTION = "run"
    CATEGORY = "MidFar Extract/Terrain Assets"
    DESCRIPTION = "Expand a reference-guided asset plan into one independent two-image carrier per target."

    def run(self, plan, workflow_template, style_filename, shape_filename, max_assets):
        parsed = parse_assets(plan, max_assets)
        expanded = expand_terrain_workflow(parsed, json.loads(workflow_template), style_filename, shape_filename)
        return json.dumps(expanded, ensure_ascii=False), json.dumps(parsed, ensure_ascii=False)


class BAMidFarSelectionExpand:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan": ("STRING", {"forceInput": True}),
            "workflow_template": ("STRING", {"multiline": True}),
            "reference_filename": ("STRING", {"default": "reference.png"}),
            "max_assets": ("INT", {"default": 3, "min": 2, "max": 3}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("workflow_json", "manifest")
    FUNCTION = "run"
    CATEGORY = "MidFar Extract/Selected Assets"
    DESCRIPTION = "Select 2–3 ranked connected units and create independent cropped-reference carriers."

    def run(self, plan, workflow_template, reference_filename, max_assets):
        parsed = parse_selection(plan, max_assets)
        workflow = expand_selection(parsed, json.loads(workflow_template), reference_filename)
        return json.dumps(workflow, ensure_ascii=False), json.dumps(parsed, ensure_ascii=False)


class BAMidFarSelectedAsset:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan": ("STRING", {"multiline": True}),
            "reference": ("IMAGE",),
            "key_mode": (["auto", "green", "magenta", "blue"],),
            "filename_prefix": ("STRING", {"default": "midfar-selected"}),
            "crop_context": ("FLOAT", {"default": .02, "min": 0.0, "max": .15, "step": .01}),
        }}

    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "reference_crop", "key_color", "filename_prefix", "manifest")
    FUNCTION = "run"
    CATEGORY = "MidFar Extract/Selected Assets"
    DESCRIPTION = "One whole connected asset; crop from this image rather than editing the entire old scene."

    def run(self, plan, reference, key_mode, filename_prefix, crop_context):
        parsed = parse_selection(plan)
        if len(parsed["objects"]) != 1:
            raise ValueError("Use SelectionExpand first: exactly one selected asset per carrier.")
        target = parsed["objects"][0]
        crop, info = crop_reference(reference.detach().cpu().numpy(), target["bbox"], crop_context)
        key = choose_key(crop, key_mode)
        record = {"style": parsed["style"], "object": target, "crop": info}
        return (selected_prompt(parsed, target, key), torch.from_numpy(crop), key,
                object_filename(target, filename_prefix), json.dumps(record, ensure_ascii=False))


class BAMidFarSelectedMatte:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "key_color": ("STRING", {"forceInput": True}),
            "filename_prefix": ("STRING", {"forceInput": True}),
            "softness": ("FLOAT", {"default": .06, "min": 0.0, "max": .2, "step": .01}),
            "padding": ("INT", {"default": 24, "min": 0, "max": 256}),
            "strict": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("rgba", "preview", "filename_prefix", "quality")
    FUNCTION = "run"
    CATEGORY = "MidFar Extract/Selected Assets"
    DESCRIPTION = "Observed-screen alpha recovery; preserve soft edges and reject incomplete/disjoint outputs."

    def run(self, image, key_color, filename_prefix, softness, padding, strict):
        import numpy as np
        rgba, quality = selected_matte(image.detach().cpu().numpy(), key_color, softness, padding, strict)
        quality["filename_prefix"] = filename_prefix
        alpha = rgba[..., 3:4]
        yy, xx = np.indices(alpha.shape[1:3])
        check = np.where(((xx//24+yy//24) % 2)[None, ..., None] == 0, .72, .92)
        preview = rgba[..., :3]*alpha+check*(1-alpha)
        return (torch.from_numpy(rgba), torch.from_numpy(preview.astype("float32")),
                filename_prefix, json.dumps(quality, ensure_ascii=False))


NODE_CLASS_MAPPINGS = {
    "BAMidFarPlanPrompt": BAMidFarPlanPrompt,
    "BAMidFarMatte": BAMidFarMatte,
    "BAMidFarObjectPlan": BAMidFarObjectPlan,
    "BAMidFarObjectMatte": BAMidFarObjectMatte,
    "BAMidFarWorkflowExpand": BAMidFarWorkflowExpand,
    "BATerrainAssetPrompt": BATerrainAssetPrompt,
    "BATerrainWorkflowExpand": BATerrainWorkflowExpand,
    "BAMidFarSelectionExpand": BAMidFarSelectionExpand,
    "BAMidFarSelectedAsset": BAMidFarSelectedAsset,
    "BAMidFarSelectedMatte": BAMidFarSelectedMatte,
    "BAMidFarCandidateReview": BAMidFarCandidateReview,
    "BAMidFarThreePlans": BAMidFarThreePlans,
    "BAMidFarAutonomousReference": BAMidFarAutonomousReference,
    "BAMidFarAssetInspect": BAMidFarAssetInspect,
    "BAMidFarRepairPrompt": BAMidFarRepairPrompt,
    "BAMidFarChooseAsset": BAMidFarChooseAsset,
    "BAMidFarTargetGuide": BAMidFarTargetGuide,
    "BAMidFarCompleteFoot": BAMidFarCompleteFoot,
    "BAMidFarNativeImageFile": BAMidFarNativeImageFile,
    "BAMidFarLocalizationView": BAMidFarLocalizationView,
    "BAMidFar3DModel": BAMidFar3DModel,
    "BAMidFar3DGenerate": BAMidFar3DGenerate,
    "BAMidFarGLBExport": BAMidFarGLBExport,
    "BAMidFarCompactExtract": BAMidFarCompactExtract,
    "BAMidFarReferencePlan": BAMidFarReferencePlan,
    "BAMidFarReferenceExtract": BAMidFarReferenceExtract,
    "BAMidFarReferenceGLB": BAMidFarReferenceGLB,
    "BAMidFarVisibleAsset": BAMidFarVisibleAsset,
    "BAMidFarVisiblePNG": BAMidFarVisiblePNG,
    "BAMidFarVisiblePE": BAMidFarVisiblePE,
    "BAMidFarVisibleThreePNG": BAMidFarVisibleThreePNG,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BAMidFarPlanPrompt": "MidFar Scene Prompt / 通用分层提示",
    "BAMidFarMatte": "MidFar Soft Matte / 中远景软边抠图",
    "BAMidFarObjectPlan": "MidFar Individual Objects / 逐物体动态提取",
    "BAMidFarObjectMatte": "MidFar Single Asset Matte / 单物体透明裁切",
    "BAMidFarWorkflowExpand": "MidFar Workflow Expand / 单物体工作流自动展开",
    "BATerrainAssetPrompt": "Terrain Asset Prompt / 单体山石草丘提示",
    "BATerrainWorkflowExpand": "Terrain Workflow Expand / 双参考地形资产展开",
    "BAMidFarSelectionExpand": "MidFar Selection Expand / 精选2–3个完整资产",
    "BAMidFarSelectedAsset": "MidFar Selected Asset / 连体局部定位与提取",
    "BAMidFarSelectedMatte": "MidFar Selected Matte / 连体素材软边透明裁切",
    "BAMidFarCandidateReview": "MidFar Candidate Review / 候选完整性复核",
    "BAMidFarThreePlans": "MidFar Three Plans / 三份独立计划",
    "BAMidFarAutonomousReference": "MidFar Autonomous Reference / 当前图目标参考",
    "BAMidFarAssetInspect": "MidFar Asset Inspect / 完整性与透明检查",
    "BAMidFarRepairPrompt": "MidFar Repair Prompt / 自动审查反馈修正",
    "BAMidFarChooseAsset": "MidFar Choose Asset / 仅输出合格透明素材",
    "BAMidFarTargetGuide": "MidFar Target Guide / 自动轮廓定位参考",
    "BAMidFarCompleteFoot": "MidFar Complete Foot / 保留原形只合成山脚",
    "BAMidFarNativeImageFile": "MidFar Native Image File / 原生图片传递",
    "BAMidFarLocalizationView": "MidFar Localization View / 明确当前图目标",
    "BAMidFar3DModel": "MidFar 3D Model / 共享混元3D模型",
    "BAMidFar3DGenerate": "MidFar 3D Generate / 完整体积生成",
    "BAMidFarGLBExport": "MidFar GLB Export / 内嵌贴图GLB导出",
    "BAMidFarCompactExtract": "MidFar Compact / 单图自动三份素材",
    "BAMidFarReferencePlan": "MidFar Reference Plan / 单图统一识别",
    "BAMidFarReferenceExtract": "MidFar Reference Extract / 自动提取与检查",
    "BAMidFarReferenceGLB": "MidFar Reference GLB / 生成并导出模型",
    "BAMidFarVisibleAsset": "MidFar Visible Asset / 目标定位与PE提示",
    "BAMidFarVisiblePNG": "MidFar Visible PNG / 透明化并输出提取图",
    "BAMidFarVisiblePE": "MidFar Visible PE / 三份提示词自动分路",
    "BAMidFarVisibleThreePNG": "MidFar Visible Three PNG / 三图统一去重与输出",
}
