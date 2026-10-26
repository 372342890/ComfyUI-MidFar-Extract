import json
import torch
from .core import parse_plan, choose_key, compile_prompt, matte
from .objects import parse_objects, object_prompt, object_filename, crop_asset, expand_workflow


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


NODE_CLASS_MAPPINGS = {
    "BAMidFarPlanPrompt": BAMidFarPlanPrompt,
    "BAMidFarMatte": BAMidFarMatte,
    "BAMidFarObjectPlan": BAMidFarObjectPlan,
    "BAMidFarObjectMatte": BAMidFarObjectMatte,
    "BAMidFarWorkflowExpand": BAMidFarWorkflowExpand,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BAMidFarPlanPrompt": "MidFar Scene Prompt / 通用分层提示",
    "BAMidFarMatte": "MidFar Soft Matte / 中远景软边抠图",
    "BAMidFarObjectPlan": "MidFar Individual Objects / 逐物体动态提取",
    "BAMidFarObjectMatte": "MidFar Single Asset Matte / 单物体透明裁切",
    "BAMidFarWorkflowExpand": "MidFar Workflow Expand / 单物体工作流自动展开",
}
