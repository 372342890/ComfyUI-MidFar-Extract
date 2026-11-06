"""Independent native Seedream 5.0 Pro requests with exact selected dimensions."""
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np
from PIL import Image
import torch

try:
    from .compact_native import invoke
    from .image_handoff import native_image_file
except ImportError:
    from compact_native import invoke
    from image_handoff import native_image_file


DEFAULT_INSTRUCTION = (
    "将唯一参考图增强为一张1024×1024的高清素材图。只以本次参考图为依据，"
    "保持同一个主体、当前视角、朝向、轮廓、比例、构图、颜色、背景和原有美术风格。"
    "提升清晰度、边缘质量与合理的材质细节；保持主体完整。"
    "不要旋转主体，不要添加其他物体，不要拼图，不要生成多视图，不要文字或水印。"
    "非方形参考图保持比例，以原有背景补齐方形画布，不要拉伸或裁掉主体。"
)
SIZE = 1024
RESOLUTIONS = {"960x960": (960, "960"), "1024x1024": (1024, "1k"),
               "2048x2048": (2048, "2k")}
MAX_SEED = 0x7fffffff
LOGGER = logging.getLogger(__name__)


def _settings(instruction, seed, resolution="1024x1024"):
    template = json.loads(
        Path(__file__).with_name("compact_settings.json").read_text(encoding="utf8"))
    model = template["generate"]["model"]
    if json.loads(model).get("model_name") != "Doubao-Seedream-5.0-pro":
        raise ValueError("The installed template must select Seedream 5.0 Pro.")
    # Use only the model descriptor; never reuse extraction prompts or credentials.
    return {"model": model, "user_prompt": instruction, "extra_inputs": {
        "size": resolution, "output_format": "png", "watermark": False,
        "layer_decomposition": False, "seed": seed,
    }}


def _rgb_tensor(value, exact_size=None):
    if not isinstance(value, torch.Tensor):
        raise ValueError("Expected an IMAGE tensor.")
    if (value.ndim != 4 or value.shape[-1] not in (3, 4)
            or min(value.shape[:3]) < 1):
        raise ValueError("Expected non-empty IMAGE [batch,height,width,3 or 4].")
    if exact_size and tuple(value.shape[:3]) != (1, exact_size, exact_size):
        raise ValueError(
            f"Seedream must return exactly one {exact_size}x{exact_size} image; "
            f"received shape {tuple(value.shape)}.")
    value = value.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(value).all() or value.min() < 0 or value.max() > 1:
        raise ValueError("IMAGE pixels must be finite and in the range 0..1.")
    if value.shape[-1] == 4:
        alpha = value[..., 3:4]
        value = value[..., :3] * alpha + (1 - alpha)
    return value


def _generated_image(result, size=SIZE):
    if not isinstance(result, (list, tuple)) or not result:
        raise ValueError("Seedream returned no IMAGE output.")
    value = result[0]
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("Seedream must return exactly one IMAGE output.")
        value = value[0]
    return _rgb_tensor(value, exact_size=size)


def _load_cache(path, size=SIZE):
    try:
        with Image.open(path) as image:
            if image.size != (size, size) or image.mode != "RGB":
                raise ValueError(f"Completed image must be {size}x{size} RGB.")
            pixels = np.asarray(image).astype(np.float32) / 255
        return torch.from_numpy(pixels)
    except Exception as error:
        raise ValueError(
            f"Invalid completed cache file {path.name}; remove this file or disable "
            f"reuse_completed to generate again: {error}") from error


def _save_cache(value, path):
    pixels = np.round(value[0].numpy() * 255).astype(np.uint8)
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, suffix=".png", delete=False) as file:
            temporary = Path(file.name)
        Image.fromarray(pixels, "RGB").save(temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def enhance_batch(images, mappings, node_info, unique_id, prompt,
                  input_directory, output_directory, seed=20261105, max_images=0,
                  instruction=DEFAULT_INSTRUCTION, reuse_completed=True,
                  check_interrupt=None, resolution="1024x1024", on_completed=None):
    """Return ordered RGB IMAGE batch and the actual generated file records."""
    source = _rgb_tensor(images)
    if not isinstance(resolution, str) or resolution not in RESOLUTIONS:
        raise ValueError("resolution must be 960x960, 1024x1024 or 2048x2048.")
    size, cache_label = RESOLUTIONS[resolution]
    if type(seed) is not int or not 0 <= seed <= MAX_SEED:
        raise ValueError("seed must be an integer between 0 and 2147483647.")
    if type(max_images) is not int or max_images < 0:
        raise ValueError("max_images must be 0 (all) or a positive integer.")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Enhancement instruction must not be empty.")
    # Saved workflows retain their old widget text when a plugin is updated.
    instruction = re.sub(r"(?<!\d)(512|768|960|1024|2048)\s*[xX×*]\s*\1(?!\d)",
                         f"{size}×{size}", instruction.strip())
    count = min(source.shape[0], max_images) if max_images else source.shape[0]
    input_directory = Path(input_directory).resolve()
    output_directory = Path(output_directory).resolve()
    cache_directory = output_directory / f"seedream-batch-{cache_label}"
    cache_directory.mkdir(parents=True, exist_ok=True)
    # Retain the original 1024 key; other sizes get independent namespaces.
    key_prefix = ("seedream-batch-1k-v1" if size == SIZE
                  else f"seedream-batch-{resolution}-v1").encode("ascii") + b"\0"
    # Allocate once; a 2048 image occupies about 48 MiB in this RGB tensor.
    output = torch.empty((count, size, size, 3), dtype=torch.float32, device="cpu")
    records = []
    for index in range(count):
        if check_interrupt:
            check_interrupt()
        try:
            settings = _settings(instruction, (seed + index) % (MAX_SEED + 1),
                                 resolution)
            reference = input_directory / native_image_file(
                source[index:index + 1].numpy(), input_directory)
            key = hashlib.sha256(
                key_prefix + reference.read_bytes()
                + json.dumps(settings, sort_keys=True, ensure_ascii=False).encode("utf8")
            ).hexdigest()
            destination = cache_directory / (key + ".png")
            cached = reuse_completed and destination.is_file()
            request_seconds = 0.0
            LOGGER.info("Seedream 5.0 Pro %s: image %s/%s (%s)",
                        resolution, index + 1, count,
                        "completed cache" if cached else "native request")
            if cached:
                pixels = _load_cache(destination, size)
            else:
                started = perf_counter()
                result = invoke(mappings, "BALLMImage", node_info, unique_id, prompt,
                                settings, [str(reference)], [])
                request_seconds = perf_counter() - started
                value = _generated_image(result, size)
                _save_cache(value, destination)
                # Cache and first-run output have identical 8-bit PNG pixels.
                pixels = _load_cache(destination, size)
                del result, value
            output[index].copy_(pixels)
            records.append({"index": index + 1, "cached": bool(cached),
                            "resolution": resolution, "request_seconds": request_seconds,
                            "relative_file": destination.relative_to(
                                output_directory).as_posix()})
            del pixels
            if on_completed:
                on_completed(index + 1, count)
        except Exception as error:
            raise RuntimeError(
                f"Seedream 5.0 Pro image {index + 1}/{count} failed: {error}. "
                "Completed images are retained; failed requests are not automatically retried."
            ) from error
    return output, records


class SeedreamBatch2K:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "seed": ("INT", {"default": 20261105, "min": 0, "max": MAX_SEED}),
            "max_images": ("INT", {"default": 0, "min": 0, "max": 10000,
                                   "tooltip": "0处理全部；1或2可先小批量验证。"}),
            "reuse_completed": ("BOOLEAN", {"default": True,
                                            "tooltip": "复用已生成图片，避免重复调用。"}),
            "instruction": ("STRING", {"default": DEFAULT_INSTRUCTION, "multiline": True}),
        }, "optional": {
            "resolution": (list(RESOLUTIONS), {
                "default": "1024x1024",
                "tooltip": "真实模型请求尺寸；960为921600像素，输出严格检查，不会缩放替代。"}),
        }, "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT"}}

    RETURN_TYPES = ("IMAGE",)
    # Keep saved class and socket identifiers for existing workflow connections.
    RETURN_NAMES = ("images_2048",)
    FUNCTION = "run"
    CATEGORY = "image/asset_sheet"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "多张小图逐张调用真实Seedream 5.0 Pro，按所选尺寸输出高清图像批次。"
        "每次仅使用当前一张参考图；成功图片缓存，失败显示序号。")

    def run(self, images, seed=20261105, max_images=0, reuse_completed=True,
            instruction=DEFAULT_INSTRUCTION, unique_id=None, prompt=None,
            resolution="1024x1024"):
        import folder_paths
        import nodes
        from comfy.model_management import throw_exception_if_processing_interrupted
        from comfy.utils import ProgressBar

        count = min(images.shape[0], max_images) if max_images else images.shape[0]
        progress = ProgressBar(count)
        result, records = enhance_batch(
            images, nodes.NODE_CLASS_MAPPINGS, getattr(self, "node_info", {}),
            unique_id, prompt, folder_paths.get_input_directory(),
            folder_paths.get_output_directory(), seed=seed, max_images=max_images,
            instruction=instruction, reuse_completed=reuse_completed,
            check_interrupt=throw_exception_if_processing_interrupted,
            resolution=resolution, on_completed=progress.update_absolute)
        previews = []
        for record in records:
            path = Path(record["relative_file"])
            previews.append({"filename": path.name, "subfolder": path.parent.as_posix(),
                             "type": "output"})
        return {"ui": {"images": previews, "seedream_timings": records},
                "result": (result,)}
