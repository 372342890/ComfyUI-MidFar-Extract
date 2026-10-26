"""Model-localized reference isolation; a guide never counts as a final asset."""
import json
import cv2
import numpy as np

try:
    from .core import key_rgb, validate_image
    from .autonomous_audit import _one_plan, _object
except ImportError:
    from core import key_rgb, validate_image
    from autonomous_audit import _one_plan, _object


def target_guide(reference_crop, plan_text, localization_text, key):
    reference = validate_image(reference_crop)
    if reference.shape[0] != 1:
        raise ValueError("Guide needs one current reference crop.")
    plan, target = _one_plan(plan_text)
    localization = _object(localization_text, "localization")
    if localization.get("asset_id") != target["id"]:
        raise ValueError("Localization asset_id must match the actual target.")
    polygon = localization.get("visible_polygon")
    if (not isinstance(polygon, list) or not 3 <= len(polygon) <= 80 or
            any(not isinstance(point, list) or len(point) != 2 or
                any(type(v) is not int or not 0 <= v <= 1000 for v in point)
                for point in polygon)):
        raise ValueError("visible_polygon requires 3..80 points on an integer 0..1000 grid.")
    for field in ("description", "hidden_base"):
        if not isinstance(localization.get(field), str) or not localization[field].strip():
            raise ValueError("Localization needs " + field)
    h, w = reference.shape[1:3]
    points = np.round(np.asarray(polygon)*[w-1, h-1]/1000).astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [points], 1)
    if not .015 <= float(mask.mean()) <= .94:
        raise ValueError("Localization polygon has implausible target coverage.")
    # Place only source pixels on a square screen; no texture is invented here.
    # Generative completion happens in the subsequent native Seedream node.
    ys, xs = np.where(mask)
    x0, x1, y0, y1 = xs.min(), xs.max()+1, ys.min(), ys.max()+1
    color = key_rgb(key)
    side = max(w, h)
    guide = np.broadcast_to(color, (side, side, 3)).copy().astype(np.float32)
    extent_w, extent_h = x1-x0, y1-y0
    scale = min(1., .65*side/max(extent_w, extent_h))
    cut = reference[0, y0:y1, x0:x1, :3]
    cut_mask = mask[y0:y1, x0:x1].astype(np.float32)
    dw, dh = max(1, round(extent_w*scale)), max(1, round(extent_h*scale))
    cut = cv2.resize(cut, (dw, dh), interpolation=cv2.INTER_AREA)
    cut_mask = cv2.resize(cut_mask, (dw, dh), interpolation=cv2.INTER_NEAREST)[..., None]
    left, top = (side-dw)//2, (side-dh)//2
    guide[top:top+dh, left:left+dw] = cut*cut_mask+color*(1-cut_mask)
    prompt = (
        f"编辑当前原图像素构成的粗轮廓参考，输出一座完整素材，背景均匀纯色{key}。"
        "优先保留原图目标自己的浅色、空气透视、笔触、峰形和相连岩肩，不重新设计。"
        "参考是自动粗定位，不是成品或不可改遮罩；可以删掉误纳入的天空/别的深度残片，"
        "恢复被粗定位漏掉的真实相连部分，并补全被挡住的短自然岩脚。"
        "下缘沿上方坡面自然收束成有厚度、略圆缓起伏的完整岩脚，不能平直切片。"
        "不增加山峰、草地托盘、浮空岛裙、树木、散落物。主体缩到画布约65%，四侧留白。"
        "不要加深颜色、增强对比、锐化成写实岩石；远景柔和雾色直接保留。只输出成图。\n"
        f"The input is a rough source-pixel localization GUIDE for {target['name']} "
        f"(asset_id {target['id']}), not a finished asset. This editable guide identifies "
        "the intended source unit; remove incorrect fragments from other depths and "
        "restore genuinely attached parts if localization missed them. Retain the correct "
        "visible target silhouette, peaks and attached shoulders. Do not reproduce the polygon mask's artificial lower "
        "edge: reconstruct the hidden short natural mountain foot and smooth natural outline "
        "from the target's own structure. Extend the lower visible slopes downward only "
        "slightly, let them curve into a gently rounded irregular foot with real volume. "
        "Preserve the original upper pixels' soft colors and painted texture; do not sharpen "
        "or saturate. No floating-island skirt, tabletop, slice, basin "
        "or added peak. Do not restore the removed foreground mountains or trees.\n"
        f"Local target observation: {localization['description']}\n"
        f"Hidden base to complete: {localization['hidden_base']}\n"
        f"Connected parts: {json.dumps(target['connected_parts'], ensure_ascii=False)}\n"
        f"Excluded objects: {json.dumps(target['excluded_objects'], ensure_ascii=False)}\n"
        f"Original style: {plan['style']}. Retain source light, colors, textures, viewpoint "
        "and atmospheric perspective. Finish as one complete natural terrain asset centered "
        f"with 15% empty margin on every side, on uniform pure {key}. No cast shadows, "
        "reflections, text, panels or environmental ground. Only output the completed image."
    )
    record = {**localization, "schema_version": 1, "guidance_only": True,
              "source_crop_size": [w, h], "key_color": key,
              "coverage": float(mask.mean()), "output_size": [side, side],
              "source_texture_preserved": True}
    return guide[None], prompt, json.dumps(record, ensure_ascii=True)
