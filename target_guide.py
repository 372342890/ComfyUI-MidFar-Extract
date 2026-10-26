"""Model-localized reference isolation; a guide never counts as a final asset."""
import json
import cv2
import numpy as np

try:
    from .core import key_rgb, validate_image
    from .autonomous_audit import _one_plan, _object, _identity_note
except ImportError:
    from core import key_rgb, validate_image
    from autonomous_audit import _one_plan, _object, _identity_note


def _polygon(value, label):
    if (not isinstance(value, list) or not 3 <= len(value) <= 256 or
            any(not isinstance(point, list) or len(point) != 2 or
                any(type(v) is not int or not 0 <= v <= 1000 for v in point)
                for point in value)):
        raise ValueError(label + " requires 3..256 points on an integer 0..1000 grid.")
    return value


def target_guide(reference_crop, plan_text, localization_text, key, identity_record=""):
    reference = validate_image(reference_crop)
    if reference.shape[0] != 1:
        raise ValueError("Guide needs one current reference crop.")
    plan, target = _one_plan(plan_text)
    localization = _object(localization_text, "localization")
    if localization.get("asset_id") != target["id"]:
        raise ValueError("Localization asset_id must match the actual target.")
    polygon = _polygon(localization.get("visible_polygon"), "visible_polygon")
    for field in ("description", "hidden_base"):
        if not isinstance(localization.get(field), str) or not localization[field].strip():
            raise ValueError("Localization needs " + field)
    h, w = reference.shape[1:3]
    points = np.round(np.asarray(polygon)*[w-1, h-1]/1000).astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [points], 1)
    if not .015 <= float(mask.mean()) <= .94:
        raise ValueError("Localization polygon has implausible target coverage.")
    occluders = localization.get("occluder_polygons", [])
    if not isinstance(occluders, list) or len(occluders) > 100:
        raise ValueError("occluder_polygons must be a list of at most 100 exclusions.")
    allowed = {"tree", "forest", "grass_tuft", "other_depth_terrain", "water", "sky", "cloud"}
    protected = {v.casefold() for v in target["connected_parts"] + [target["name"]]}
    excluded = np.zeros_like(mask)
    for item in occluders:
        if not isinstance(item, dict) or item.get("kind") not in allowed:
            raise ValueError("occluder kind cannot be target rock or mountain.")
        for field in ("name", "reason"):
            if not isinstance(item.get(field), str) or not 0 < len(item[field].strip()) <= 2000:
                raise ValueError("occluder requires concise " + field)
        if item["name"].strip().casefold() in protected:
            raise ValueError("occluder cannot exclude a connected target part.")
        shape = _polygon(item.get("polygon"), "occluder polygon")
        pts = np.round(np.asarray(shape)*[w-1, h-1]/1000).astype(np.int32)
        cv2.fillPoly(excluded, [pts], 1)
    removed = int(np.count_nonzero(mask & excluded))
    if removed > .45*int(mask.sum()):
        raise ValueError("Occluder removal exceeds 45% of target coverage.")
    mask[excluded > 0] = 0
    if float(mask.mean()) < .015:
        raise ValueError("Occluder removal leaves insufficient target coverage.")
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
    cut = cv2.resize(cut*cut_mask[..., None], (dw, dh), interpolation=cv2.INTER_AREA)
    cut_mask = cv2.resize(cut_mask, (dw, dh), interpolation=cv2.INTER_AREA)[..., None]
    left, top = (side-dw)//2, (side-dh)//2
    guide[top:top+dh, left:left+dw] = cut+color*(1-cut_mask)
    prompt = (
        f"图1是真实原图裁片，图2是自动粗定位参考。输出一座完整素材，背景均匀纯色{key}。"
        "图1中目标真实形体、峰数和颜色是权威依据；图2仅提示身份，"
        "可能漏掉整座峰或大块岩肩、有遮挡凹口，禁止照抄这些粗定位错误。"
        "优先保留原图目标自己的浅色、空气透视、笔触、峰形和相连岩肩，不重新设计。"
        "参考是自动粗定位，不是成品或不可改遮罩；可以删掉误纳入的天空/别的深度残片，"
        "恢复被粗定位漏掉的真实相连部分，并补全被挡住的短自然岩脚。"
        "先检查参考下缘：锯齿状树冠、蓝绿远林色带及前排山体都是遮挡物，"
        "即使与山体同色或粘在一起也必须完全移除。该处重建连续的山岩/积雪坡面。"
        "左右岩脚不能按多边形边界做竖直切面或纸片底，要自然收束并保留体积。"
        "下缘沿上方坡面自然收束成有厚度、略圆缓起伏的完整岩脚，不能平直切片。"
        "不增加山峰、草地托盘、浮空岛裙、树木、散落物。主体缩到画布约65%，四侧留白。"
        "不要加深颜色、增强对比、锐化成写实岩石；远景柔和雾色直接保留。只输出成图。\n"
        "Image 1 is the actual original reference crop and is authoritative for the "
        "target's visible geometry, peak count, proportions, material, light and style. "
        f"Image 2 is a rough source-pixel localization GUIDE for {target['name']} "
        f"(asset_id {target['id']}), not a finished asset. This editable guide identifies "
        "the intended source unit; remove incorrect fragments from other depths and "
        "restore genuinely attached parts from Image 1 if localization missed them. "
        "The guide may omit an entire real peak, remove large visible rock regions or "
        "contain occluder-shaped notches. Never copy those errors or let Image 2 override "
        "the actual original reference crop. Use the target name, appearance and attached "
        "parts to identify the same unit in Image 1, excluding its other-depth neighbors.\n"
        + _identity_note(identity_record, target) +
        "Retain the correct visible target silhouette, peaks and attached shoulders. "
        "Do not reproduce the polygon mask's artificial lower "
        "edge: reconstruct the hidden short natural mountain foot and smooth natural outline "
        "from the target's own structure. Extend the lower visible slopes downward only "
        "slightly, let them curve into a gently rounded irregular foot with real volume. "
        "Preserve the original upper pixels' soft colors and painted texture; do not sharpen "
        "or saturate. No floating-island skirt, tabletop, slice, basin "
        "or added peak. Do not restore the removed foreground mountains or trees.\n"
        "Inspect the lower guide carefully before editing: any forest silhouette, "
        "tiny jagged conifer crowns or distant blue tree ribbon is an occluder, not "
        "mountain texture, even if its color matches the rock. Remove the entire "
        "forest strip, including branches and trunks, and reconstruct continuous rock/snow slopes "
        "in its place. Erase tree silhouette geometry, not merely recolor it blue or turn "
        "its pointed crowns into artificial rock spikes. "
        "Complete both lower side shoulders and the base as short curved volumetric "
        "feet; never terminate them at straight side cut planes or a paper-thin polygon.\n"
        "The lower silhouette follows a softly undulating natural foot with thickness, "
        "not a straight slice, regular geometric ellipse or frozen guide boundary.\n"
        f"Local target observation: {localization['description']}\n"
        f"Hidden base to complete: {localization['hidden_base']}\n"
        f"Connected parts: {json.dumps(target['connected_parts'], ensure_ascii=False)}\n"
        f"Excluded objects: {json.dumps(target['excluded_objects'], ensure_ascii=False)}\n"
        f"Original style: {plan['style']}. Retain source light, colors, textures, viewpoint "
        "and atmospheric perspective. Finish as one complete natural terrain asset centered "
        f"with 15% empty margin on every side, on uniform pure {key}. "
        "The screen is a perfectly flat RGB color: no grain, painted texture, speckles, "
        "paper fibers, lighting, gradients or strokes anywhere in the background. "
        "Do not transfer the source image's paint texture to the empty screen. No cast shadows, "
        "reflections, text, panels or environmental ground. Only output the completed image."
    )
    record = {**localization, "occluder_polygons": occluders,
              "removed_occluder_pixels": removed, "schema_version": 1, "guidance_only": True,
              "source_crop_size": [w, h], "key_color": key,
              "coverage": float(mask.mean()), "output_size": [side, side],
              "source_texture_preserved": True}
    return guide[None], prompt, json.dumps(record, ensure_ascii=True)
