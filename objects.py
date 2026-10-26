"""Instance planning and transparent asset cropping."""
import json
import re
import copy

import cv2
import numpy as np


def parse_objects(text, layer="both", max_objects=8):
    if layer not in ("both", "midground", "background"):
        raise ValueError("layer must be both, midground or background.")
    if not isinstance(max_objects, int) or not 1 <= max_objects <= 32:
        raise ValueError("max_objects must be between 1 and 32.")
    if not isinstance(text, str) or len(text) > 100000:
        raise ValueError("Object plan must be a JSON string under 100000 characters.")
    plan = None
    for match in re.finditer(r"\{", text):
        try:
            candidate, _ = json.JSONDecoder().raw_decode(text[match.start():])
            if isinstance(candidate, dict) and "objects" in candidate:
                plan = candidate
                break
        except ValueError:
            continue
    if plan is None or not isinstance(plan["objects"], list) or len(plan["objects"]) > 100:
        raise ValueError("Expected an objects array of individual mid/far instances.")
    if not isinstance(plan.get("style", ""), str):
        raise ValueError("style must be text.")
    ids, objects, rejected = set(), [], []
    for obj in plan["objects"]:
        if not isinstance(obj, dict) or obj.get("is_single", True) is not True:
            raise ValueError("Every entry must represent one individual object, not an entire layer.")
        for field in ("id", "name", "location", "appearance", "occlusion"):
            if not isinstance(obj.get(field), str) or not 0 < len(obj[field].strip()) < 2000:
                raise ValueError(f"Object {field} must be concise non-empty text.")
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", obj["id"]) or obj["id"] in ids:
            raise ValueError("Object ids must be unique safe ASCII identifiers.")
        ids.add(obj["id"])
        if obj.get("layer") not in ("midground", "background"):
            raise ValueError("Object layer must be midground or background.")
        if re.search(r"树丛|树群|树林|森林|天际线|建筑群|群山|山脉|绿化带|地面|草地|草坪|"
                     r"小岛|湖心岛|\b(forest|skyline|grove|mountain range|building cluster|ground plane|island)\b",
                     obj["name"], re.IGNORECASE) or obj["name"].strip() in (
                         "江面", "水面", "湖面", "湖泊", "河流", "天空", "地形", "water", "sky"):
            rejected.append({"id": obj["id"], "name": obj["name"],
                             "reason": "Environment surface or group, not an individual object."})
            continue
        box = obj.get("bbox")
        normalized = False
        if isinstance(box, list) and len(box) == 4 and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) and
                0 <= v <= 1000 for v in box) and max(box) > 10:
            box = [v/1000 if v > 1 else v for v in box]
            normalized = True
        if not isinstance(box, list) or len(box) != 4 or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and
                np.isfinite(v) and 0 <= v <= 1 for v in box) or not (
                box[0] < box[2] and box[1] < box[3]):
            raise ValueError("bbox must be normalized [left, top, right, bottom].")
        objects.append({**obj, "bbox": box, "bbox_normalized": normalized,
                        **{f: obj[f].strip() for f in
                                 ("id", "name", "location", "appearance", "occlusion")}})
    selected = [o for o in objects if layer == "both" or o["layer"] == layer]
    if layer == "both":
        mid = [o for o in selected if o["layer"] == "midground"]
        far = [o for o in selected if o["layer"] == "background"]
        selected = [o for i in range(max(len(mid), len(far))) for group in (mid, far)
                    for o in group[i:i+1]]
    if not selected:
        raise ValueError("No identifiable individual objects in the selected distance layer.")
    return {"style": plan.get("style", ""), "objects": selected[:max_objects],
            "all_objects": objects, "rejected_objects": rejected, "total": len(selected),
            "truncated": max(0, len(selected) - max_objects)}


def object_prompt(plan, target, key):
    if key not in ("#00FF00", "#FF00FF", "#0000FF"):
        raise ValueError("Unsupported chroma key.")
    others = "；".join(f"{o['name']}（{o['location']}）"
                      for o in plan.get("all_objects", plan["objects"]) if o["id"] != target["id"])
    box = ", ".join(f"{v:.3f}" for v in target["bbox"])
    return (
        "只输出一个独立完整物体的可复用美术素材，不输出任何整层景观或组合场景。\n"
        f"精确目标：{target['name']}；距离层：{target['layer']}；位置：{target['location']}；"
        f"归一化位置框（左上右下）：[{box}]。外观：{target['appearance']}。\n"
        "依据参考图定位这一个实例，保持其可见部分的形状、原颜色、纹理、材质、照明、"
        "相机朝向与透视。将这一个物体移动到画布中心并放大到合适的素材尺寸，"
        "主体宽高均不得超过画布70%，四边至少15%纯色键空白安全区，"
        "保证屋顶、树冠、树干脚部、建筑基础和侧面等完整且不被画布切断。\n"
        f"遮挡情况：{target['occlusion']}。只用本物体已有结构合理补全被遮挡部位；"
        "楼只补它自己的墙和基础，树只补自己的枝干树冠并露出完整短树干，"
        "山只保留该独立山体的轮廓并用山岩结构补齐原云雾遮挡处。"
        "保留细枝、建筑栏杆和自然软边，不能遗留遮挡者形状的缺口。\n"
        f"其他实例全部移除，包括同类相邻物体：{others or '所有其他物体'}。"
        "禁止把建筑群、树丛、整片树林、连续山脉、天际线或整片地面一起带入。"
        "不附带地面托盘、草地、道路、天空、云、雾、装饰、水面、投影、文字或边框。"
        "原图风格：" + plan.get("style", "") + "。\n"
        f"所有非目标区域一律均匀纯色 {key}。不画棋盘格、渐变、背景布、阴影或反光。"
        "色键仅属于空白，绝不能染色主体。只交付这一张单物体色键图。")


def object_filename(target, prefix):
    def safe(value):
        return re.sub(r"[\W_]+", "_", str(value), flags=re.UNICODE).strip("_")[:60] or "object"
    parts = [safe(p) for p in str(prefix).replace("\\", "/").split("/") if p not in ("", ".", "..")]
    return "/".join(parts + [target["layer"], safe(target["id"]) + "_" + safe(target["name"])])


def crop_asset(rgba, padding=24):
    rgba = np.asarray(rgba, dtype=np.float32)
    if rgba.ndim != 4 or rgba.shape[0] != 1 or rgba.shape[-1] != 4 or not np.isfinite(rgba).all():
        raise ValueError("Expected one finite RGBA image [1,H,W,4].")
    if not isinstance(padding, int) or not 0 <= padding <= 256:
        raise ValueError("padding must be an integer between 0 and 256.")
    rgba = rgba.copy()
    alpha = rgba[0, ..., 3]
    count, labels, stats, _ = cv2.connectedComponentsWithStats((alpha > .01).astype(np.uint8), 8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    removed = 0
    if len(areas) and areas.max() > 0:
        substantial = np.flatnonzero(areas >= max(16, float(areas.max())*.01)) + 1
        if len(substantial):
            distance = cv2.distanceTransform((~np.isin(labels, substantial)).astype(np.uint8), cv2.DIST_L2, 5)
            minimum = np.full(count, np.inf, np.float32)
            np.minimum.at(minimum, labels.ravel(), distance.ravel())
            candidates = ~np.isin(np.arange(count), substantial)
            candidates[0] = False
            remote_labels = np.flatnonzero(candidates & (minimum > max(12, min(alpha.shape)*.03)))
            remote = np.isin(labels, remote_labels)
            removed = int(remote.sum())
            rgba[0, remote] = 0
    ys, xs = np.where(alpha > .01)
    if not len(xs):
        raise ValueError("Single-object extraction is empty. Review plan or chroma key.")
    h, w = alpha.shape
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    solid = (alpha > .5).astype(np.uint8)
    radius = max(1, min(h, w) // 200)
    solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, np.ones((2*radius+1, 2*radius+1), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(solid, 8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    components = int(np.sum(areas >= max(16, float(areas.sum())*.025)))
    touches = x0 == 0 or y0 == 0 or x1 == w or y1 == h
    warnings = []
    if components > 1:
        warnings.append("Multiple substantial components: review for more than one object.")
    if touches:
        warnings.append("Object touches canvas border: review completeness or scene-strip leakage.")
    out = np.clip(rgba[:, max(0, y0-padding):min(h, y1+padding),
                       max(0, x0-padding):min(w, x1+padding), :], 0, 1).copy()
    return out, {"source_size": [w, h], "source_bbox": [x0, y0, x1, y1],
                 "output_size": [out.shape[2], out.shape[1]],
                 "meaningful_components": components, "touches_border": touches,
                 "removed_remote_pixels": removed,
                 "foreground_coverage": round(float((out[..., 3] > .05).mean()), 5),
                 "warning": " ".join(warnings),
                 "occlusion_completion": "AI inference; hidden texture is not ground truth."}


def expand_workflow(plan, template, reference_filename):
    """ByteArtist maps a generation carrier once; give each instance its own carrier."""
    nodes = template["nodes"]
    root = copy.deepcopy(next(n for n in nodes if n["data"]["type"] == "LoadImage"))
    root_id = root["id"]
    next(i for i in root["data"]["inputs"] if i["name"] == "image")["value"] = reference_filename
    branch_types = ("BAMidFarObjectPlan", "BALLMImage", "BAMidFarObjectMatte", "SaveImage")
    branches = [next(n for n in nodes if n["data"]["type"] == t) for t in branch_types]
    branch_ids = {n["id"] for n in branches}
    edges = [e for e in template["edges"] if e["target"] in branch_ids and
             e["source"] in (branch_ids | {root_id})]
    result = {"name": "ba-clipboard", "version": 1, "nodes": [root], "edges": []}
    for j, target in enumerate(plan["objects"]):
        mapping = {n["id"]: str(1000+j*10+k) for k, n in enumerate(branches)}
        for k, source in enumerate(branches):
            n = copy.deepcopy(source)
            n["id"] = mapping[source["id"]]
            n["position"] = {"x": 420+k*440, "y": j*550}
            n["selected"] = False
            n["data"]["title"] = f"{target['id']} · {target['name']} · {branch_types[k]}"
            if k == 0:
                inputs = n["data"]["inputs"]
                plan_input = next(i for i in inputs if i["name"] == "plan")
                plan_input["value"] = json.dumps({"style": plan.get("style", ""),
                                                  "objects": [target]}, ensure_ascii=False)
                plan_input.pop("format", None)
                next(i for i in inputs if i["name"] == "max_objects")["value"] = 1
                n["data"]["extra"] = {"output_preview_names": []}
            result["nodes"].append(n)
        for source in edges:
            e = copy.deepcopy(source)
            e["id"] = str(8000+len(result["edges"]))
            e["source"] = mapping.get(e["source"], e["source"])
            e["target"] = mapping[e["target"]]
            result["edges"].append(e)
    return result
