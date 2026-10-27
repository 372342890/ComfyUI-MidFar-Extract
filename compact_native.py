"""Invoke installed ByteArtist carriers using their real list-input contract."""
from copy import deepcopy
from pathlib import Path
try:
    from .autonomous_audit import _json
except ImportError:
    from autonomous_audit import _json


def request_json(request):
    """Correct malformed JSON once with the same references and model settings."""
    feedback = ""
    for attempt in range(2):
        response = request(feedback)
        try:
            value = _json(response, "native response")
            if not isinstance(value, dict):
                raise ValueError("Native JSON response must be an object.")
        except ValueError as error:
            if attempt:
                raise
            feedback = (
                "上一条返回不是合法 JSON。请仅修复输出格式，重新观察同一参考图并遵循原来的"
                "目标和固定字段；所有坐标必须写出实际数字，不要使用省略号、注释或未加引号的文本。"
                "不要更换目标、编造部件或弱化任何检查。只返回完整 JSON 对象。\n"
                f"解析错误：{error}\n上一条实际返回：\n{response}"
            )
            continue
        return response


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
