"""Shared planning and three independent native image-to-GLB branches."""
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .autonomous_plan import candidate_review, split_three, asset_reference
from .localization_view import localization_view
from .target_guide import target_guide
from .image_handoff import native_image_file, save_inspection, save_review_records, _diagnostic_path
from .compact_native import invoke, request_json
from .compact_pipeline import extract_until_accepted
from .core import validate_image
from .three_d_nodes import BAMidFar3DGenerate, BAMidFarGLBExport


class _NativeRun:
    def __init__(self, owner, prefix, suffix, reference, unique_id, prompt):
        import folder_paths
        self.owner = owner
        self.unique_id = unique_id
        self.prompt = prompt
        self.settings = json.loads(
            Path(__file__).with_name("compact_settings.json").read_text(encoding="utf8"))
        self.output_dir = folder_paths.get_output_directory()
        self.input_dir = folder_paths.get_input_directory()
        self.path = _diagnostic_path(prefix, suffix, self.output_dir)
        self.record = {"schema_version": 1, "status": "running",
                       "source_size": [reference.shape[2], reference.shape[1]],
                       "native_calls": []}
        self.checkpoint()

    def checkpoint(self):
        self.path.write_text(json.dumps(self.record, ensure_ascii=False, indent=2),
                             encoding="utf8")

    def fail(self, error):
        self.record.update(status="failed", error=str(error))
        calls = self.record["native_calls"]
        if calls and calls[-1]["status"] == "running":
            calls[-1].update(status="failed", error=str(error))
        self.checkpoint()

    def native(self, stage, images, texts=()):
        import nodes
        files = [str(Path(self.input_dir)/native_image_file(image, self.input_dir))
                 for image in images]
        image_stage = stage in ("generate", "repair")

        def request(feedback=""):
            call = {"stage": stage, "status": "running", "started": time.time(),
                    "format_correction": bool(feedback)}
            self.record["native_calls"].append(call)
            self.checkpoint()
            result = invoke(
                nodes.NODE_CLASS_MAPPINGS, "BALLMImage" if image_stage else "BALLMText",
                getattr(self.owner, "node_info", {}), self.unique_id, self.prompt,
                self.settings[stage], files, list(texts)+([feedback] if feedback else []))
            call.update(status="success", duration_seconds=round(time.time()-call["started"], 3))
            if not image_stage:
                call["response"] = result[0]
            self.checkpoint()
            return result if image_stage else result[0]

        if not image_stage:
            return request_json(request)
        value = request()[0]
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError("Native image generation must return exactly one image.")
            value = value[0]
        return validate_image(value.detach().cpu().numpy())


def _reference(reference):
    reference = validate_image(reference.detach().cpu().numpy())
    if reference.shape[0] != 1:
        raise ValueError("Upload exactly one current reference image.")
    return reference


class BAMidFarReferencePlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "reference": ("IMAGE",),
            "filename_prefix": ("STRING", {"default": "midfar-reference/asset"}),
        }, "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT"}}

    RETURN_TYPES = ("STRING",)*5
    RETURN_NAMES = ("plan_1", "plan_2", "plan_3", "filename_prefix", "audit")
    FUNCTION = "run"
    CATEGORY = "MidFar Extract/Reference"
    DESCRIPTION = "Identify and review three actual safe units from one current reference."

    def run(self, reference, filename_prefix, unique_id, prompt):
        reference = _reference(reference)
        run = _NativeRun(self, filename_prefix, "_plan.json", reference, unique_id, prompt)
        try:
            candidates = run.native("plan", [reference])
            run.record["candidate_plan"] = candidates
            sheet, manifest = candidate_review(candidates, reference)
            run.record["candidate_manifest"] = manifest
            run.checkpoint()
            selection = run.native("selection", [reference, sheet], [manifest])
            plans = split_three(selection, manifest)
            run.record.update(status="success", selection=json.loads(plans[3]),
                              plans=list(plans[:3]))
            run.checkpoint()
        except Exception as error:
            run.fail(error)
            raise
        return (*plans[:3], filename_prefix, json.dumps(run.record, ensure_ascii=False))


class BAMidFarReferenceExtract:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "reference": ("IMAGE",),
            "plan": ("STRING", {"forceInput": True}),
            "filename_prefix": ("STRING", {"forceInput": True}),
        }, "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT"}}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("rgba", "filename_prefix", "extraction_record")
    FUNCTION = "run"
    CATEGORY = "MidFar Extract/Reference"
    OUTPUT_NODE = True
    DESCRIPTION = "Localize, extract, review and conditionally repair one planned unit."

    def run(self, reference, plan, filename_prefix, unique_id, prompt):
        reference = _reference(reference)
        _, crop, key, prefix, source_record = asset_reference(plan, reference, filename_prefix)
        run = _NativeRun(self, prefix, "_run.json", reference, unique_id, prompt)
        run.record.update(plan=plan, source_record=json.loads(source_record), prefix=prefix)
        try:
            marked, localization_prompt, identity = localization_view(crop, source_record)
            localized = run.native("localization", [marked], [localization_prompt])
            guide, guide_prompt, guide_record = target_guide(crop, plan, localized, key, identity)
            run.record.update(localization=localized, guide_record=guide_record)
            run.checkpoint()

            def generate(version, previous, feedback):
                return run.native("generate" if version == 1 else "repair",
                                  [crop, guide] if version == 1 else [crop, guide, previous],
                                  [guide_prompt if version == 1 else feedback])

            def review(inspection, review_prompt, version):
                return run.native("review", [inspection], [review_prompt])

            def observe(version, raw, inspection, rgba, numeric, review_prompt, numerics, visuals):
                save_inspection(raw, inspection, rgba, numeric, review_prompt,
                                prefix, version, run.output_dir)
                save_review_records(prefix, numerics, visuals, run.output_dir)
                run.record["current_version"] = version
                run.checkpoint()

            rgba, audit = extract_until_accepted(
                generate, review, marked, plan, key, guide, identity, observe)
            path = _diagnostic_path(prefix, ".png", run.output_dir)
            Image.fromarray(np.round(rgba[0]*255).astype(np.uint8), "RGBA").save(path)
            relative = path.relative_to(Path(run.output_dir).resolve())
            run.record.update(status="success", png=str(relative), audit=json.loads(audit))
            run.checkpoint()
        except Exception as error:
            run.fail(error)
            raise
        record = json.dumps(run.record, ensure_ascii=False)
        return {"ui": {"images": [{"filename": path.name, "subfolder": str(relative.parent),
                                  "type": "output"}], "text": [record]},
                "result": (torch.from_numpy(rgba), prefix, record)}


class BAMidFarReferenceGLB:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "rgba": ("IMAGE",), "model": ("MODEL",), "clip_vision": ("CLIP_VISION",),
            "vae": ("VAE",), "filename_prefix": ("STRING", {"forceInput": True}),
            "extraction_record": ("STRING", {"forceInput": True}),
            "seed": ("INT", {"default": 20261026, "min": 0, "max": 0xffffffffffffffff}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("glb", "export_record")
    FUNCTION = "run"
    CATEGORY = "MidFar Extract/Reference"
    OUTPUT_NODE = True
    DESCRIPTION = "Native Hunyuan geometry and embedded-texture GLB for one accepted image."

    def run(self, rgba, model, clip_vision, vae, filename_prefix, extraction_record, seed):
        extraction = json.loads(extraction_record)
        if extraction.get("status") != "success" or extraction.get("prefix") != filename_prefix:
            raise ValueError("GLB generation requires an accepted image with the matching asset prefix.")
        mesh, generation_record = BAMidFar3DGenerate().run(rgba, model, clip_vision, vae, seed)
        generation = json.loads(generation_record)
        generation["extraction"] = extraction
        return BAMidFarGLBExport().run(
            mesh, rgba, filename_prefix, json.dumps(generation, ensure_ascii=False))
