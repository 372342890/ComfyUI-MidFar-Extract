"""Model-independent scene prompt compilation and chroma matting (BHWC, 0..1)."""
import json
import re

import cv2
import numpy as np

KEYS = {"green": "#00FF00", "magenta": "#FF00FF", "blue": "#0000FF"}


def parse_plan(text):
    if not isinstance(text, str) or len(text) > 60000:
        raise ValueError("Scene plan must be a JSON string under 60000 characters.")
    decoder = json.JSONDecoder()
    result = None
    for match in re.finditer(r"\{", text):
        try:
            candidate, _ = decoder.raw_decode(text[match.start():])
            if isinstance(candidate, dict) and "midground" in candidate:
                result = candidate
                break
        except ValueError:
            continue
    if result is None:
        raise ValueError("No scene-plan JSON object found.")
    for name in ("midground", "background"):
        values = result.get(name)
        if not isinstance(values, list) or len(values) > 40 or not all(
                isinstance(v, str) and 0 < len(v.strip()) < 1500 for v in values):
            raise ValueError(f"{name} must be an array of concise object descriptions.")
        result[name] = [v.strip() for v in values]
    for name in ("style", "occlusion", "foreground", "sky"):
        if name in result and not isinstance(result[name], (str, list)):
            raise ValueError(f"{name} must be text or a list of descriptions.")
    return result


def validate_image(image):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 4 or image.shape[-1] not in (3, 4) or min(image.shape[:3]) < 1:
        raise ValueError("IMAGE must have shape [batch, height, width, 3 or 4].")
    if not np.isfinite(image).all():
        raise ValueError("IMAGE contains non-finite values.")
    return np.clip(image, 0, 1)


def key_rgb(key):
    if key not in KEYS.values():
        raise ValueError("key_color must be #00FF00, #FF00FF or #0000FF.")
    return np.array([int(key[i:i+2], 16)/255 for i in (1, 3, 5)], dtype=np.float32)


def dominance(rgb, key):
    r, g, b = np.moveaxis(rgb, -1, 0)
    if key == "#00FF00":
        return g - np.maximum(r, b)
    if key == "#0000FF":
        return b - np.maximum(r, g)
    return np.minimum(r, b) - g


def choose_key(reference, mode):
    image = validate_image(reference)
    if mode in KEYS:
        return KEYS[mode]
    if mode != "auto":
        raise ValueError("Unknown key mode.")
    # Sampling bounds memory while considering all reference images in a batch.
    rgb = image[..., :3].reshape(-1, 3)[::max(1, image[..., 0].size // 200000)]
    scores = [(float(np.mean(np.maximum(dominance(rgb, k), 0)**2)), k) for k in KEYS.values()]
    return min(scores, key=lambda pair: pair[0])[1]


def describe(value):
    return "；".join(value) if isinstance(value, list) else str(value or "")


def compile_prompt(plan, layer, key):
    if layer not in ("midground", "background"):
        raise ValueError("layer must be midground or background.")
    key_rgb(key)
    name = "中景" if layer == "midground" else "远景"
    objects = "；".join(plan[layer])
    other = "background" if layer == "midground" else "midground"
    exclude = "；".join(filter(None, [
        describe(plan.get("foreground")), describe(plan.get("sky")), describe(plan[other])]))
    instruction = (
        f"严格依据这张参考图制作可叠加的{name}独立素材。保持同一画布、长宽比、相机、"
        "原物体位置和大小、原画风、色调与照明。输出一张图片，不加文字、不加边框、不画对比图。\n"
    )
    if not objects:
        instruction += f"原图没有可确认的{name}，不要发明景物，整张图输出纯色底。\n"
    else:
        instruction += f"本层仅保留：{objects}。\n移除其他层：{exclude or '全部不属于本层的内容'}。\n"
        instruction += (
            "这是完整图层重建，保留可见部分并连续补全被其他层遮挡的本层实体，"
            "遵守已有透视、轮廓与纹理，不把遮挡边界挖成孔洞，不留下遮挡者的轮廓；"
            "不扩展为全新的场景。远景的宽度和高度由原图实际层次决定。"
            "保留细枝、细线、栏杆和自然软边。\n")
    instruction += (
        f"分层关系：{describe(plan.get('occlusion'))}。风格：{describe(plan.get('style'))}。\n"
        f"被移除区域与所有空白一律填均匀纯色 {key}（RGB色键），包括天空和下方空白。"
        "不保留原天空、云、地面底色，不产生渐变、阴影、反光、绿色或紫色光晕。"
        "本层主体保留原本颜色，色键只属于背景。")
    return instruction


def matte(image, key, softness=0.04, despill=True):
    image = validate_image(image)
    if not 0.0 <= float(softness) <= 0.2:
        raise ValueError("softness must be between 0 and 0.2.")
    k = key_rgb(key)
    rgba_batch, previews, notes = [], [], []
    for source in image:
        rgb = source[..., :3].copy()
        excess = np.clip(dominance(rgb, key), 0, 1)
        alpha = 1 - excess
        # Small deviations from the requested key are screen, without eroding edges.
        screen = np.linalg.norm(rgb-k, axis=-1) <= float(softness)
        opaque = (excess <= 0.03) & ~screen
        alpha[screen] = 0
        alpha[opaque] = 1
        if opaque.any():
            # Every opaque pixel is a seed; DIST_LABEL_PIXEL preserves thin features.
            distance, labels = cv2.distanceTransformWithLabels(
                (~opaque).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
            colors = np.zeros((int(labels.max()) + 1, 3), np.float32)
            colors[labels[opaque]] = rgb[opaque]
            foreground = colors[labels]
            vector = foreground-k
            denominator = np.sum(vector*vector, axis=-1)
            solved = np.sum((rgb-k)*vector, axis=-1)/np.maximum(denominator, 1e-6)
            use = ~opaque & ~screen & (distance <= 32) & (denominator > 0.01)
            # Reject distant unrelated colors whose mixing line poorly matches.
            predicted = solved[..., None]*foreground + (1-solved[..., None])*k
            use &= np.linalg.norm(predicted-rgb, axis=-1) < 0.14
            alpha[use] = np.clip(solved[use], 0, 1)
        alpha[screen] = 0
        alpha[opaque] = 1
        if despill:
            rgb = np.clip((rgb-(1-alpha[..., None])*k)/np.maximum(alpha[..., None], 1e-5), 0, 1)
        if source.shape[-1] == 4:
            alpha *= source[..., 3]
        rgb[alpha < 1e-5] = 0
        rgba = np.concatenate([rgb, alpha[..., None]], axis=-1).astype(np.float32)
        yy, xx = np.indices(alpha.shape)
        check = np.where(((xx//24 + yy//24) % 2)[..., None] == 0, .72, .92)
        previews.append((rgb*alpha[..., None] + check*(1-alpha[..., None])).astype(np.float32))
        rgba_batch.append(rgba)
        coverage = float(np.mean(alpha > .05))
        notes.append({"foreground_coverage": round(coverage, 5),
                      "soft_edge_fraction": round(float(np.mean((alpha > .02) & (alpha < .98))), 5),
                      "warning": "Layer almost empty; review extraction." if coverage < .005 else
                                 "Screen not found; review key color." if float(screen.mean()) < .001 else "",
                      "mask_convention": "1=foreground opaque, 0=background transparent"})
    rgba_batch = np.stack(rgba_batch)
    return rgba_batch, rgba_batch[..., 3].copy(), np.stack(previews), json.dumps(notes, ensure_ascii=False)
