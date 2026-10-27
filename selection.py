"""Select connected mid/far units, crop current references, and export carriers."""
import copy
import json
import math
import re

import cv2
import numpy as np

try:
    from .core import dominance, key_rgb, nearest_colors, validate_image
    from .objects import crop_asset, object_filename
except ImportError:
    from core import dominance, key_rgb, nearest_colors, validate_image
    from objects import crop_asset, object_filename


def _box(values):
    if not isinstance(values, list) or len(values) != 4 or not all(
            type(v) in (float, int) and math.isfinite(v) and 0 <= v <= 1 for v in values):
        raise ValueError("bbox must be four finite normalized coordinates.")
    if values[0] >= values[2] or values[1] >= values[3]:
        raise ValueError("bbox must be [left, top, right, bottom].")
    return values


def parse_selection(text, max_assets=3):
    if type(max_assets) is not int or max_assets not in (2, 3):
        raise ValueError("Select at most 2 or 3 assets.")
    if not isinstance(text, str) or len(text) > 100000:
        raise ValueError("Selection plan must be text under 100000 characters.")
    plan = None
    for start in re.finditer(r"\{", text):
        try:
            candidate, _ = json.JSONDecoder().raw_decode(text[start.start():])
            if isinstance(candidate, dict) and isinstance(candidate.get("objects"), list):
                plan = candidate
                break
        except ValueError:
            continue
    if plan is None or not 1 <= len(plan["objects"]) <= 100:
        raise ValueError("No suitable asset objects found; revise reference or selection.")
    if not isinstance(plan.get("style"), str) or not 0 < len(plan["style"].strip()) < 6000:
        raise ValueError("style must describe this reference image.")
    ids, groups, rejected = set(), {}, []
    for source in plan["objects"]:
        if not isinstance(source, dict):
            raise ValueError("Each selection must be one complete asset unit.")
        obj = copy.deepcopy(source)
        for field in ("id", "name", "location", "appearance", "occlusion", "selection_reason"):
            if not isinstance(obj.get(field), str) or not 0 < len(obj[field].strip()) < 3000:
                raise ValueError(field + " must be non-empty concise text.")
            obj[field] = obj[field].strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", obj["id"]) or obj["id"] in ids:
            raise ValueError("Object identifiers must be unique safe ASCII identifiers.")
        ids.add(obj["id"])
        reason = None
        if obj.get("kind") not in ("rock", "hill", "mountain"):
            reason = "Only rock, hill and mountain assets are used by this workflow."
        elif re.search(r"草堆|草簇|草丛|枯草|碎石|整片风景|整个场景|"
                       r"\b(grass tuft|grass clump|gravel|entire scene)\b", obj["name"], re.I):
            reason = "Grass tufts, tiny scatter details and scenery layers are excluded."
        elif obj["name"] in ("地面", "天空", "云", "水面", "草地", "草坪", "ground", "sky"):
            reason = "A background surface is not a reusable complete asset."
        elif obj.get("suitable") is False or obj.get("boundary_cut") is True:
            reason = "Marked unsuitable or severely cut by the source image."
        if reason:
            rejected.append({"id": obj["id"], "name": obj["name"], "reason": reason})
            continue
        if obj.get("layer") not in ("midground", "background"):
            raise ValueError("layer must be midground or background.")
        if obj.get("asset_unit") not in ("connected_formation", "independent"):
            raise ValueError("asset_unit must indicate connected_formation or independent.")
        if obj.get("is_single") is not True:
            raise ValueError("is_single must identify one reusable complete unit.")
        parts = obj.get("connected_parts", [])
        if not isinstance(parts, list) or len(parts) > 40 or not all(
                isinstance(p, str) and 0 < len(p.strip()) < 500 for p in parts):
            raise ValueError("connected_parts must be a list of short part descriptions.")
        obj["connected_parts"] = list(dict.fromkeys(p.strip() for p in parts))
        obj["bbox"] = _box(obj.get("bbox"))
        score = obj.get("priority", 50)
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 100:
            raise ValueError("priority must be a finite suitability value from 0 to 100.")
        obj["priority"] = score
        unit = obj.get("unit_id", obj["id"])
        if not isinstance(unit, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", unit):
            raise ValueError("unit_id must be a safe connected-unit identifier.")
        obj["unit_id"] = unit
        if unit not in groups:
            obj["source_ids"] = [obj["id"]]
            groups[unit] = obj
            continue
        previous = groups[unit]
        if previous["layer"] != obj["layer"] or previous["kind"] != obj["kind"]:
            raise ValueError("Parts of the same connected unit must agree on layer and kind.")
        a, b = previous["bbox"], obj["bbox"]
        previous["bbox"] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
        previous["connected_parts"] = list(dict.fromkeys(previous["connected_parts"] + obj["connected_parts"]))
        previous["appearance"] += "；" + obj["appearance"]
        previous["occlusion"] += "；" + obj["occlusion"]
        previous["source_ids"].append(obj["id"])
        previous["asset_unit"] = "connected_formation"
        previous["priority"] = max(previous["priority"], obj["priority"])
    ranked = sorted(groups.values(), key=lambda obj: -obj["priority"])
    if not ranked:
        raise ValueError("No suitable complete terrain assets; do not invent targets.")
    # Cover available distance layers before taking the next highest-ranked unit.
    selected = []
    for layer in ("midground", "background"):
        candidate = next((o for o in ranked if o["layer"] == layer), None)
        if candidate is not None:
            selected.append(candidate)
    selected.extend(o for o in ranked if o not in selected)
    selected = selected[:max_assets]
    selected.sort(key=lambda obj: (obj["layer"] == "background", -obj["priority"]))
    warning = "合适目标不足2个，按实际数量提取，不凑数。" if len(selected) < 2 else ""
    return {"style": plan["style"].strip(), "objects": selected, "all_objects": ranked,
            "rejected_objects": rejected, "total": len(ranked),
            "truncated": max(0, len(ranked)-len(selected)), "warning": warning,
            "grouping_rule": "Directly joined rocks or peaks sharing a natural base are one asset."}


def crop_reference(image, bbox, context=.02):
    image = validate_image(image)
    if image.shape[0] != 1:
        raise ValueError("Each selected unit must use exactly one current reference image.")
    bbox = _box(bbox)
    if type(context) not in (float, int) or not math.isfinite(context) or not 0 <= context <= .15:
        raise ValueError("crop context must be from 0 to 0.15 of target dimensions.")
    _, h, w, _ = image.shape
    dx, dy = (bbox[2]-bbox[0])*w*context, (bbox[3]-bbox[1])*h*context
    x0, y0 = max(0, math.floor(bbox[0]*w-dx)), max(0, math.floor(bbox[1]*h-dy))
    x1, y1 = min(w, math.ceil(bbox[2]*w+dx)), min(h, math.ceil(bbox[3]*h+dy))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Selected crop is empty.")
    return image[:, y0:y1, x0:x1, :].copy(), {
        "source_size": [w, h], "source_bbox_pixels": [x0, y0, x1, y1],
        "crop_size": [x1-x0, y1-y0], "rescaled": False}


def selected_prompt(plan, target, key):
    key_rgb(key)
    parts = "；".join(target["connected_parts"]) or "目标自身完整结构"
    return (
        f"从这张原图局部提取整个「{target['name']}」，作为一个完整可复用资产。"
        "当前输入已经按目标bbox裁到局部，不能寻找整张旧参考图中的其他物体。\n"
        f"原图位置：{target['location']}。可见外观：{target['appearance']}。\n"
        f"必须整体保留的相接部位：{parts}。连体的别拆：相接岩身、附着岩柱、多个顶帽、"
        "多个峰与共享自然岩基/岩台属于同一个物体。内部裂纹不构成拆分理由。"
        "所有相接部分保持原位置、高低、尺寸和前后遮挡关系，不能只抽出最高柱。"
        "自身的阶梯岩台或覆草坡面属于目标，要保留，不能当成外部地面删掉。\n"
        f"遮挡情况：{target['occlusion']}。仅按自身结构补齐被遮挡的侧面和短底部，"
        "不能留遮挡者形状的凹洞，也不能把底部切成纸片。不得添加新峰或设计随机变体。\n"
        f"原图风格：{plan['style']}。保留可见轮廓、颜色、裂纹、材质、光照和视角，"
        "不强化为写实照片。远景保留空气透视。只移除与目标不相连的其他物体、"
        "外围平坦地面、天空、云、草堆、文字和投影。不移除目标的相连结构。\n"
        "完整居中，宽和高均不超过画布70%，四边至少15%空白安全区；"
        "全部峰顶、左右边缘、台基和岩脚必须完整。只输出一张整体目标图，不拆分或画四视图。\n"
        f"所有非目标区域严格使用均匀纯色{key}，无渐变、纹理、反光或棋盘格，"
        "色键只属于空白，不染色主体。"
    )


def expand_selection(plan, template, reference_filename):
    if not isinstance(reference_filename, str) or not reference_filename.strip():
        raise ValueError("Current reference filename must be non-empty.")
    roots = [n for n in template.get("nodes", []) if n["data"]["type"] == "LoadImage"]
    if len(roots) != 1:
        raise ValueError("Selection template needs exactly one current reference loader.")
    root = copy.deepcopy(roots[0])
    pin = next(i for i in root["data"]["inputs"] if i["name"] == "image")
    pin["value"] = reference_filename
    pin.pop("format", None)
    required = ("BAMidFarSelectedAsset", "BALLMImage", "BAMidFarSelectedMatte")
    for kind in required:
        if sum(n["data"]["type"] == kind for n in template["nodes"]) != 1:
            raise ValueError("Template must contain exactly one " + kind)
    branches = [n for n in template["nodes"] if n["data"]["type"] in required + ("SaveImage", "SaveText|pysssss")]
    branch_ids = {n["id"] for n in branches}
    edges = [e for e in template.get("edges", []) if e["target"] in branch_ids and
             e["source"] in branch_ids | {root["id"]}]
    result = {"name": "ba-clipboard", "version": 1, "nodes": [root], "edges": []}
    for j, target in enumerate(plan["objects"]):
        mapping = {n["id"]: f"selected_{j}_{k}" for k, n in enumerate(branches)}
        for k, source in enumerate(branches):
            n = copy.deepcopy(source)
            n["id"] = mapping[source["id"]]
            n["position"] = {"x": 420+min(k, 3)*440, "y": j*1050+(450*(k-3) if k >= 4 else 0)}
            n["selected"] = False
            n["data"]["title"] = target["layer"]+" · "+target["name"]+" · "+source["data"]["type"]
            if source["data"]["type"] == "BAMidFarSelectedAsset":
                pin = next(i for i in n["data"]["inputs"] if i["name"] == "plan")
                pin["value"] = json.dumps({"style": plan["style"], "objects": [target]}, ensure_ascii=False)
                pin.pop("format", None)
            if source["data"]["type"] == "SaveText|pysssss":
                for pin in n["data"]["inputs"]:
                    if pin["name"] == "file":
                        stem, sep, suffix = str(pin["value"]).rpartition(".")
                        pin["value"] = (stem if sep else suffix)+"_"+target["id"]+"."+(suffix if sep else "json")
            if source["data"]["type"] == "SaveImage":
                for pin in n["data"]["inputs"]:
                    if pin["name"] == "filename_prefix" and pin.get("value"):
                        pin["value"] += "/"+target["id"]
            result["nodes"].append(n)
        for source in edges:
            edge = copy.deepcopy(source)
            edge["id"] = "selection_edge_"+str(len(result["edges"]))
            edge["source"] = mapping.get(edge["source"], edge["source"])
            edge["target"] = mapping[edge["target"]]
            result["edges"].append(edge)
    return result


def selected_matte(image, key, softness=.06, padding=24, strict=True):
    image = validate_image(image)
    if image.shape[0] != 1:
        raise ValueError("Exactly one generated image is required per asset carrier.")
    intended = key_rgb(key)
    if not 0 <= float(softness) <= .2:
        raise ValueError("softness must be from 0 to 0.2.")
    rgb = image[0, ..., :3]
    h, w = rgb.shape[:2]
    band = max(1, min(32, h//10, w//10))
    border = np.concatenate((rgb[:band].reshape(-1, 3), rgb[-band:].reshape(-1, 3),
                             rgb[:, :band].reshape(-1, 3), rgb[:, -band:].reshape(-1, 3)))
    candidates = (dominance(border, key) > .35) & (border.max(axis=-1) > .4)
    if float(candidates.mean()) < .5:
        raise ValueError("Color key not found on the canvas border; review generated background.")
    observed = np.median(border[candidates], axis=0)
    if np.linalg.norm(observed/observed.max()-intended) > .4:
        raise ValueError("Observed background does not match the selected color key.")
    screen = (np.linalg.norm(rgb-observed, axis=-1) < float(softness)+.08) & (dominance(rgb, key) > .35)
    solid = (dominance(rgb, key) < .1) & ~screen
    stable = cv2.erode(solid.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if not screen.any() or not stable.any():
        raise ValueError("No stable object interior or chroma-screen region found.")
    _, screen_color = nearest_colors(screen, rgb)
    distance, foreground = nearest_colors(stable, rgb)
    direction = foreground-screen_color
    denominator = np.sum(direction**2, axis=-1)
    alpha = np.clip(np.sum((rgb-screen_color)*direction, axis=-1)/np.maximum(denominator, 1e-6), 0, 1)
    alpha[screen] = 0
    alpha[solid & (distance == 0)] = 1
    # Isolated, tiny key-colored islands without any stable foreground support
    # are compression/screen noise, not detached object parts. Keep every
    # non-key island for crop_asset's fail-closed detached-part audit.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (alpha > .01).astype(np.uint8), 8)
    screen_noise = np.zeros_like(screen)
    raw_key = dominance(rgb, key)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] > 16:
            continue
        island = labels == label
        ring = cv2.dilate(island.astype(np.uint8),
                          np.ones((3, 3), np.uint8)).astype(bool) & ~island
        key_supported = np.all(raw_key[island] > .35) or (
            np.all(raw_key[island] > .1) and ring.any() and np.all(screen[ring]))
        if (key_supported and
                np.all(distance[island] > max(12, min(h, w)*.03))):
            screen_noise |= island
    alpha[screen_noise] = 0
    color = np.clip((rgb-(1-alpha[..., None])*screen_color)/np.maximum(alpha[..., None], 1e-5), 0, 1)
    screen_distance = cv2.distanceTransform((~screen).astype(np.uint8), cv2.DIST_L2, 5)
    edge = (alpha > .01) & (screen_distance <= 5) & (distance <= 20)
    color[edge] = foreground[edge]
    alpha[(alpha < .02) | ((distance > 12) & (alpha < .25))] = 0
    if image.shape[-1] == 4:
        alpha *= image[0, ..., 3]
    color[alpha == 0] = 0
    rgba = np.concatenate((color, alpha[..., None]), axis=-1)[None].astype("float32")
    rgba, quality = crop_asset(rgba, padding)
    quality.update({"observed_screen_rgb": np.round(observed*255).astype(int).tolist(),
                    "key_color": key, "mask_convention": "1=foreground opaque, 0=transparent",
                    "suppressed_screen_noise_pixels": int(screen_noise.sum()),
                    "soft_edge_pixels": int(((rgba[..., 3] > 0) & (rgba[..., 3] < 1)).sum())})
    if strict and quality["warning"]:
        raise ValueError(quality["warning"])
    return rgba, quality
