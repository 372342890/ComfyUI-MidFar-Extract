"""Independent native Seedream 5.0 Pro requests with durable 2048px results."""
import hashlib
import json
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

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
    "将唯一参考图增强为一张2048×2048的高清素材图。只以本次参考图为依据，"
    "保持同一个主体、当前视角、朝向、轮廓、比例、构图、颜色、背景和原有美术风格。"
    "提升清晰度、边缘质量与合理的材质细节；保持主体完整。"
    "不要旋转主体，不要添加其他物体，不要拼图，不要生成多视图，不要文字或水印。"
    "非方形参考图保持比例，以原有背景补齐方形画布，不要拉伸或裁掉主体。"
)
SIZE = 2048
MAX_SEED = 0x7fffffff
LOGGER = logging.getLogger(__name__)


def _settings(instruction, seed):
    template = json.loads(
        Path(__file__).with_name("compact_settings.json").read_text(encoding="utf8"))
    model = template["generate"]["model"]
    if json.loads(model).get("model_name") != "Doubao-Seedream-5.0-pro":
        raise ValueError("The installed template must select Seedream 5.0 Pro.")
    # Use only the model descriptor; never reuse extraction prompts or credentials.
    return {"model": model, "user_prompt": instruction, "extra_inputs": {
        "size": "2048x2048", "output_format": "png", "watermark": False,
        "layer_decomposition": False, "seed": seed,
    }}


def _rgb_tensor(value, exact_size=False):
    if not isinstance(value, torch.Tensor):
        raise ValueError("Expected an IMAGE tensor.")
    if (value.ndim != 4 or value.shape[-1] not in (3, 4)
            or min(value.shape[:3]) < 1):
        raise ValueError("Expected non-empty IMAGE [batch,height,width,3 or 4].")
    if exact_size and tuple(value.shape[:3]) != (1, SIZE, SIZE):
        raise ValueError("Seedream must return exactly one 2048x2048 image.")
    value = value.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(value).all() or value.min() < 0 or value.max() > 1:
        raise ValueError("IMAGE pixels must be finite and in the range 0..1.")
    if value.shape[-1] == 4:
        alpha = value[..., 3:4]
        value = value[..., :3] * alpha + (1 - alpha)
    return value


def _generated_image(result):
    if not isinstance(result, (list, tuple)) or not result:
        raise ValueError("Seedream returned no IMAGE output.")
    value = result[0]
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("Seedream must return exactly one IMAGE output.")
        value = value[0]
    return _rgb_tensor(value, exact_size=True)


def _load_cache(path):
    try:
        with Image.open(path) as image:
            if image.size != (SIZE, SIZE) or image.mode != "RGB":
                raise ValueError("Completed image must be 2048x2048 RGB.")
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
                  check_interrupt=None):
    """Return ordered RGB IMAGE batch and the actual generated file records."""
    source = _rgb_tensor(images)
    if type(seed) is not int or not 0 <= seed <= MAX_SEED:
        raise ValueError("seed must be an integer between 0 and 2147483647.")
    if type(max_images) is not int or max_images < 0:
        raise ValueError("max_images must be 0 (all) or a positive integer.")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Enhancement instruction must not be empty.")
    instruction = instruction.strip()
    count = min(source.shape[0], max_images) if max_images else source.shape[0]
    input_directory = Path(input_directory).resolve()
    output_directory = Path(output_directory).resolve()
    cache_directory = output_directory / "seedream-batch-2k"
    cache_directory.mkdir(parents=True, exist_ok=True)
    # A 16-image batch is ~768 MiB; do not hold a second concatenation copy.
    output = torch.empty((count, SIZE, SIZE, 3), dtype=torch.float32, device="cpu")
    records = []
    for index in range(count):
        if check_interrupt:
            check_interrupt()
        try:
            settings = _settings(instruction, (seed + index) % (MAX_SEED + 1))
            reference = input_directory / native_image_file(
                source[index:index + 1].numpy(), input_directory)
            key = hashlib.sha256(
                b"seedream-batch-2k-v1\0" + reference.read_bytes()
                + json.dumps(settings, sort_keys=True, ensure_ascii=False).encode("utf8")
            ).hexdigest()
            destination = cache_directory / (key + ".png")
            cached = reuse_completed and destination.is_file()
            LOGGER.info("Seedream 5.0 Pro 2K: image %s/%s (%s)",
                        index + 1, count, "completed cache" if cached else "native request")
            if cached:
                pixels = _load_cache(destination)
            else:
                result = invoke(mappings, "BALLMImage", node_info, unique_id, prompt,
                                settings, [str(reference)], [])
                value = _generated_image(result)
                _save_cache(value, destination)
                # Cache and first-run output have identical 8-bit PNG pixels.
                pixels = _load_cache(destination)
                del result, value
            output[index].copy_(pixels)
            records.append({"index": index + 1, "cached": bool(cached),
                            "relative_file": destination.relative_to(
                                output_directory).as_posix()})
            del pixels
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
        }, "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT"}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images_2048",)
    FUNCTION = "run"
    CATEGORY = "image/asset_sheet"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "多张小图逐张调用真实Seedream 5.0 Pro，输出2048×2048高清图像批次。"
        "每次仅使用当前一张参考图；成功图片缓存，失败显示序号。")

    def run(self, images, seed=20261105, max_images=0, reuse_completed=True,
            instruction=DEFAULT_INSTRUCTION, unique_id=None, prompt=None):
        import folder_paths
        import nodes
        from comfy.model_management import throw_exception_if_processing_interrupted

        result, records = enhance_batch(
            images, nodes.NODE_CLASS_MAPPINGS, getattr(self, "node_info", {}),
            unique_id, prompt, folder_paths.get_input_directory(),
            folder_paths.get_output_directory(), seed=seed, max_images=max_images,
            instruction=instruction, reuse_completed=reuse_completed,
            check_interrupt=throw_exception_if_processing_interrupted)
        previews = []
        for record in records:
            path = Path(record["relative_file"])
            previews.append({"filename": path.name, "subfolder": path.parent.as_posix(),
                             "type": "output"})
        return {"ui": {"images": previews}, "result": (result,)}
