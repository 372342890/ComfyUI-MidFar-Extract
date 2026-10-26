"""Content-addressed PNG handoff to standard native image-loading nodes."""
import hashlib
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
