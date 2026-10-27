"""Invoke installed ByteArtist carriers using their real list-input contract."""
from copy import deepcopy
from pathlib import Path


def invoke(mappings, kind, node_info, unique_id, prompt, settings, image_files, texts):
    for file in image_files:
        if not isinstance(file, str) or not Path(file).is_absolute() or not Path(file).is_file():
            raise ValueError("A real absolute image file is required: " + str(file))
    cls = mappings.get(kind)
    if cls is None:
        raise RuntimeError("Required native ByteArtist node is unavailable: " + kind)
    instance = cls()
    instance.node_info = deepcopy(node_info or {})
    instance.node_info["node_id"] = str(unique_id)
    values = deepcopy(settings)
    values.update(unique_id=str(unique_id), prompt=prompt)
    values.update({f"image{i}": value for i, value in enumerate(image_files)})
    values.update({f"text{i}": value for i, value in enumerate(texts)})
    if getattr(cls, "INPUT_IS_LIST", False):
        values = {key: [value] for key, value in values.items()}
    result = getattr(instance, cls.FUNCTION)(**values)
    if isinstance(result, dict):
        return result["result"]
    if hasattr(result, "result"):
        return result.result
    return result
