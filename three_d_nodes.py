"""Compact wrappers around installed native Hunyuan3D nodes."""
import hashlib
import json
from pathlib import Path
import time

from .model_assets import HUNYUAN, ensure_hunyuan_checkpoint
from .mesh_glb import prepare_rgb, encode_glb


def _native(name, **inputs):
    import nodes
    cls = nodes.NODE_CLASS_MAPPINGS[name]
    instance = cls()
    method = getattr(instance, getattr(cls, "FUNCTION", "execute"))
    result = method(**inputs)
    if hasattr(result, "result"):
        return result.result
    if isinstance(result, dict):
        return result["result"]
    return result


def _write_runtime_capabilities(destination):
    import inspect
    import comfy.model_detection
    import comfy.supported_models
    detection = inspect.getsource(comfy.model_detection).splitlines()
    keep = set()
    for i, line in enumerate(detection):
        if "hunyuan" in line.lower():
            keep.update(range(max(0, i-3), min(len(detection), i+12)))
    classes = [cls for cls in comfy.supported_models.models
               if "hunyuan" in cls.__name__.lower()]
    report = {"hunyuan_detection": "\n".join(detection[i] for i in sorted(keep)),
              "supported_hunyuan_classes": {cls.__name__: inspect.getsource(cls) for cls in classes}}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")


class BAMidFar3DModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}
    RETURN_TYPES = ("MODEL", "CLIP_VISION", "VAE")
    RETURN_NAMES = ("model", "clip_vision", "vae")
    FUNCTION = "run"
    CATEGORY = "MidFar Extract/3D"
    DESCRIPTION = "Verified official Hunyuan3D 2.0; compatible with native Hunyuan3Dv2 and shared by all assets."

    def run(self):
        import folder_paths
        _write_runtime_capabilities(
            Path(folder_paths.get_output_directory())/"midfar-3d"/"hunyuan-runtime.json")
        checkpoint = ensure_hunyuan_checkpoint(
            folder_paths.models_dir,
            Path(folder_paths.get_output_directory())/"midfar-3d"/"hunyuan-download.json")
        model, clip_vision, vae = _native("ImageOnlyCheckpointLoader", ckpt_name=checkpoint)
        model = _native("ModelSamplingAuraFlow", model=model, shift=1.0)[0]
        return model, clip_vision, vae


class BAMidFar3DGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "rgba": ("IMAGE",), "model": ("MODEL",), "clip_vision": ("CLIP_VISION",),
            "vae": ("VAE",), "seed": ("INT", {"default": 20261026, "min": 0,
                                             "max": 0xffffffffffffffff})}}
    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("mesh", "generation_record")
    FUNCTION = "run"
    CATEGORY = "MidFar Extract/3D"
    DESCRIPTION = "Native image-conditioned volumetric generation; no plane/extrusion fallback."

    def run(self, rgba, model, clip_vision, vae, seed):
        import torch
        started = time.monotonic()
        rgb = torch.from_numpy(prepare_rgb(rgba.detach().cpu().numpy()))
        encoded = _native("CLIPVisionEncode", clip_vision=clip_vision, image=rgb, crop="none")[0]
        positive, negative = _native("Hunyuan3Dv2Conditioning", clip_vision_output=encoded)
        latent = _native("EmptyLatentHunyuan3Dv2", resolution=3072, batch_size=1)[0]
        samples = _native("KSampler", model=model, seed=seed, steps=30, cfg=5.0,
                          sampler_name="euler", scheduler="normal", positive=positive,
                          negative=negative, latent_image=latent, denoise=1.0)[0]
        voxel = _native("VAEDecodeHunyuan3D", vae=vae, samples=samples,
                        num_chunks=8000, octree_resolution=256)[0]
        mesh = _native("VoxelToMesh", voxel=voxel, algorithm="surface net", threshold=.6)[0]
        record = {"model": {k: v for k, v in HUNYUAN.items() if k != "url"},
                  "native_nodes": ["CLIPVisionEncode", "Hunyuan3Dv2Conditioning",
                                   "EmptyLatentHunyuan3Dv2", "KSampler",
                                   "VAEDecodeHunyuan3D", "VoxelToMesh"],
                  "seed": seed, "steps": 30, "cfg": 5, "latent_resolution": 3072,
                  "octree_resolution": 256, "algorithm": "surface net",
                  "duration_seconds": round(time.monotonic()-started, 3)}
        return mesh, json.dumps(record)


class BAMidFarGLBExport:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mesh": ("MESH",), "rgba": ("IMAGE",),
            "filename_prefix": ("STRING", {"forceInput": True}),
            "generation_record": ("STRING", {"forceInput": True})}}
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("glb_file", "export_record")
    FUNCTION = "run"
    CATEGORY = "MidFar Extract/3D"
    OUTPUT_NODE = True
    DESCRIPTION = "Real mesh GLB + embedded matte texture; side/back colors extend front reference."

    def run(self, mesh, rgba, filename_prefix, generation_record):
        import folder_paths
        vertices, faces = mesh.vertices, mesh.faces
        if vertices.ndim == 3:
            if vertices.shape[0] != 1 or faces.shape[0] != 1:
                raise ValueError("Exactly one generated mesh is required")
            vertices, faces = vertices[0], faces[0]
        blob, report = encode_glb(vertices.detach().cpu().numpy(),
                                  faces.detach().cpu().numpy(), rgba.detach().cpu().numpy())
        folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory())
        Path(folder).mkdir(parents=True, exist_ok=True)
        filename = f"{name}_{counter:05d}.glb"
        destination = Path(folder)/filename
        destination.write_bytes(blob)
        report.update(glb_sha256=hashlib.sha256(blob).hexdigest(),
                      generation=json.loads(generation_record))
        record = json.dumps(report, ensure_ascii=False, indent=2)
        destination.with_suffix(".3d.json").write_text(record, encoding="utf-8")
        return {"ui": {"3d": [{"filename": filename, "subfolder": subfolder, "type": "output"}],
                       "text": [record]},
                "result": (str(Path(subfolder)/filename), record)}
