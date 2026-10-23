import json
import torch
from .core import parse_plan, choose_key, compile_prompt, matte


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


NODE_CLASS_MAPPINGS = {
    "BAMidFarPlanPrompt": BAMidFarPlanPrompt,
    "BAMidFarMatte": BAMidFarMatte,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BAMidFarPlanPrompt": "MidFar Scene Prompt / 通用分层提示",
    "BAMidFarMatte": "MidFar Soft Matte / 中远景软边抠图",
}
