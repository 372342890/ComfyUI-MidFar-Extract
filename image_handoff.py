"""Content-addressed PNG handoff to standard native image-loading nodes."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
try:
    from .core import validate_image
except ImportError:
    from core import validate_image


def native_image_file(image, input_directory):
    image = validate_image(image)
    if image.shape[0] != 1:
        raise ValueError("A native reference file must contain exactly one image.")
    pixels = np.round(image[0, ..., :3]*255).astype(np.uint8)
    digest = hashlib.sha256(str(pixels.shape).encode()+pixels.tobytes()).hexdigest()
    name = "midfar-native-images/"+digest+".png"
    destination = Path(input_directory)/name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        Image.fromarray(pixels, "RGB").save(destination)
    return name


def _diagnostic_path(prefix, suffix, output_directory):
    if (not isinstance(prefix, str) or not 0 < len(prefix) <= 500 or
            any(v in prefix for v in ("\0", "\\", ":")) or prefix.startswith("/") or
            any(part in ("", ".", "..") for part in prefix.split("/"))):
        raise ValueError("Diagnostic prefix must be a relative safe output path.")
    base = Path(output_directory).resolve()
    destination = (base / (prefix+suffix)).resolve()
    if not destination.is_relative_to(base):
        raise ValueError("Diagnostics cannot escape the output directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def save_inspection(raw, sheet, rgba, numeric, prompt, prefix, version, output_directory):
    """Retain observations even if the later selector raises; never edit tensors."""
    if type(version) is not int or not 1 <= version <= 3:
        raise ValueError("Diagnostic version must be 1, 2 or 3.")
    for suffix, image in (("raw.png", raw), ("sheet.png", sheet), ("rgba.png", rgba)):
        image = validate_image(image)
        if image.shape[0] != 1:
            raise ValueError("One diagnostic image is required.")
        file = _diagnostic_path(prefix, f"_v{version}_{suffix}", output_directory)
        Image.fromarray(np.round(image[0]*255).astype(np.uint8)).save(file)
    _diagnostic_path(prefix, f"_v{version}_numeric.json", output_directory).write_text(
        numeric, encoding="utf8")
    _diagnostic_path(prefix, f"_v{version}_review-prompt.txt", output_directory).write_text(
        prompt, encoding="utf8")


def save_review_records(prefix, numeric_reports, visual_reports, output_directory):
    """Preserve original model replies before either selection success or failure."""
    path = _diagnostic_path(prefix, "_reviews.json", output_directory)
    path.write_text(json.dumps({"schema_version": 1, "numeric_reports": numeric_reports,
                                "visual_reports": visual_reports}, ensure_ascii=True,
                               allow_nan=False), encoding="utf8")
