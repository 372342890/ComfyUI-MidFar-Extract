"""One upload -> three checked RGBA assets, using real installed carriers."""
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


class BAMidFarCompactExtract:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "reference": ("IMAGE",),
            "filename_prefix": ("STRING", {"default": "midfar-compact/asset"}),
        }, "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT"}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("rgba_1", "rgba_2", "rgba_3", "prefix_1", "prefix_2", "prefix_3", "audit")
    FUNCTION = "run"
    CATEGORY = "MidFar Extract/Compact"
    OUTPUT_NODE = True
    DESCRIPTION = "One image; native planning, localization, extraction and conditional repair. Stops on acceptance."

    def run(self, reference, filename_prefix, unique_id, prompt):
        import folder_paths
        import nodes
        reference = validate_image(reference.detach().cpu().numpy())
        if reference.shape[0] != 1:
            raise ValueError("Upload exactly one current reference image.")
        settings = json.loads(Path(__file__).with_name("compact_settings.json").read_text(encoding="utf8"))
        output_dir = folder_paths.get_output_directory()
        status_file = _diagnostic_path(filename_prefix, "_run.json", output_dir)
        record = {"schema_version": 1, "status": "running", "source_size":
                  [reference.shape[2], reference.shape[1]], "native_calls": [], "assets": []}

        def checkpoint():
            status_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf8")

        def native(stage, images, texts=()):
            files = [str(Path(folder_paths.get_input_directory())/
                         native_image_file(image, folder_paths.get_input_directory()))
                     for image in images]
            image_stage = stage in ("generate", "repair")

            def request(feedback=""):
                call = {"stage": stage, "status": "running", "started": time.time(),
                        "format_correction": bool(feedback)}
                record["native_calls"].append(call)
                checkpoint()
                result = invoke(
                    nodes.NODE_CLASS_MAPPINGS, "BALLMImage" if image_stage else "BALLMText",
                    getattr(self, "node_info", {}), unique_id, prompt, settings[stage],
                    files, list(texts) + ([feedback] if feedback else []))
                call.update(status="success", duration_seconds=round(time.time()-call["started"], 3))
                if not image_stage:
                    call["response"] = result[0]
                checkpoint()
                return result if image_stage else result[0]

            if image_stage:
                result = request()
                value = result[0]
                if isinstance(value, list):
                    if len(value) != 1:
                        raise ValueError("Native image generation must return exactly one image.")
                    value = value[0]
                return validate_image(value.detach().cpu().numpy())
            return request_json(request)

        checkpoint()
        previews, rgba_results, prefixes = [], [], []
        try:
            candidates = native("plan", [reference])
            record["candidate_plan"] = candidates
            sheet, manifest = candidate_review(candidates, reference)
            record["candidate_manifest"] = manifest
            checkpoint()
            selection = native("selection", [reference, sheet], [manifest])
            plans = split_three(selection, manifest)
            record["selection"] = plans[3]
            checkpoint()
            for index, plan in enumerate(plans[:3], 1):
                _, crop, key, prefix, source_record = asset_reference(
                    plan, reference, filename_prefix)
                record["current_asset"] = index
                record["current_prefix"] = prefix
                checkpoint()
                marked, localization_prompt, identity = localization_view(crop, source_record)
                localized = native("localization", [marked], [localization_prompt])
                guide, guide_prompt, guide_record = target_guide(
                    crop, plan, localized, key, identity)
                record.setdefault("localizations", []).append(
                    {"plan": plan, "source_record": source_record,
                     "localization": localized, "guide_record": guide_record})
                checkpoint()

                def generate(version, previous, feedback):
                    return native("generate" if version == 1 else "repair",
                                  [crop, guide] if version == 1 else [crop, guide, previous],
                                  [guide_prompt if version == 1 else feedback])

                def review(inspection, review_prompt, version):
                    return native("review", [inspection], [review_prompt])

                def observe(version, raw, inspection, rgba, numeric, review_prompt, numerics, visuals):
                    save_inspection(raw, inspection, rgba, numeric, review_prompt,
                                    prefix, version, output_dir)
                    save_review_records(prefix, numerics, visuals, output_dir)
                    record["current_version"] = version
                    checkpoint()

                rgba, audit = extract_until_accepted(
                    generate, review, marked, plan, key, guide, identity, observe)
                path = _diagnostic_path(prefix, ".png", output_dir)
                Image.fromarray(np.round(rgba[0]*255).astype(np.uint8), "RGBA").save(path)
                relative = path.relative_to(Path(output_dir).resolve())
                previews.append({"filename": path.name, "subfolder": str(relative.parent), "type": "output"})
                rgba_results.append(torch.from_numpy(rgba))
                prefixes.append(prefix)
                record["assets"].append({"index": index, "prefix": prefix,
                                         "png": str(relative), "audit": json.loads(audit)})
                checkpoint()
            record["status"] = "success"
            checkpoint()
        except Exception as error:
            record.update(status="failed", error=str(error))
            if record["native_calls"] and record["native_calls"][-1]["status"] == "running":
                record["native_calls"][-1].update(status="failed", error=str(error))
            checkpoint()
            raise
        audit = json.dumps(record, ensure_ascii=False)
        return {"ui": {"images": previews, "text": [audit]},
                "result": (*rgba_results, *prefixes, audit)}
