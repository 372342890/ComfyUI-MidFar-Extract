"""Deterministic planning for three scalar native-model asset branches.

Images are NumPy BHWC floats. All structured public outputs are JSON strings.
Only explicit unit_id membership joins candidates; screen overlap never does.
"""
import copy
import json
import math
import re

import cv2
import numpy as np

try:
    from .core import choose_key, validate_image
    from .objects import object_filename
    from .selection import _box, crop_reference, parse_selection
except ImportError:
    from core import choose_key, validate_image
    from objects import object_filename
    from selection import _box, crop_reference, parse_selection


_ID = re.compile(r"[A-Za-z0-9_-]{1,40}")
_DESCRIPTION_FIELDS = ("name", "location", "appearance", "occlusion", "selection_reason")
_SCENERY = re.compile(
    r"\b(?:grass\s+(?:tufts?|clumps?|piles?)|tufts?|entire\s+(?:scene|landscape)|"
    r"whole\s+(?:scene|landscape)|scenery|forest|ground\s+plane|sky|water\s+surface)\b|"
    r"\u8349\u5806|\u8349\u7c07|\u8349\u4e1b|\u6574\u7247\u98ce\u666f|"
    r"\u6574\u4e2a\u573a\u666f|\u68ee\u6797|\u6811\u6797", re.I)


def _number(value, name, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be a finite number from {low} to {high}.")
    return value


def _identifier(value, name="id"):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"{name} must be a safe ASCII identifier.")
    return value


def _json_safe(value):
    # Preserve invalid candidate evidence without emitting non-standard JSON numbers.
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _dump(value):
    return json.dumps(_json_safe(value), ensure_ascii=True, allow_nan=False)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field {key} is ambiguous.")
        result[key] = value
    return result


def _decode(text, required, limit=100000):
    if not isinstance(text, str) or len(text) > limit:
        raise ValueError(f"Plan must be JSON text under {limit} characters.")
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object)
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and required in value:
            return value
    raise ValueError(f"No JSON object containing {required} found.")


def _plan(text, require_style=True):
    value = _decode(text, "objects")
    if not isinstance(value["objects"], list) or len(value["objects"]) > 100:
        raise ValueError("objects must be an array of at most 100 candidates.")
    if require_style or "style" in value:
        style = value.get("style")
        if not isinstance(style, str) or not 0 < len(style.strip()) < 6000:
            raise ValueError("style must describe the current reference image.")
        value["style"] = style.strip()
    return value


def _list(value, name, exclusions=False):
    if not isinstance(value, list) or len(value) > 100:
        raise ValueError(f"{name} must be an array of concise descriptions.")
    result = []
    for item in value:
        if isinstance(item, str) and 0 < len(item.strip()) < 3000:
            item = item.strip()
        elif exclusions and isinstance(item, dict):
            if not isinstance(item.get("name"), str) or not 0 < len(item["name"].strip()) < 3000:
                raise ValueError("Excluded objects must have a concise name.")
            if "id" in item:
                _identifier(item["id"], "excluded id")
            for field in ("location", "appearance", "reason"):
                if field in item and (not isinstance(item[field], str) or
                                      not 0 < len(item[field].strip()) < 3000):
                    raise ValueError(f"Excluded {field} must be concise text.")
            item = copy.deepcopy(item)
        else:
            raise ValueError(f"{name} must contain concise descriptions.")
        if item not in result:
            result.append(item)
    return result


def _edge(box, margin):
    return box[0] <= margin or box[1] <= margin or box[2] >= 1-margin or box[3] >= 1-margin


def _id_list(value, name):
    if not isinstance(value, list) or len(value) > 100:
        raise ValueError(f"{name} must be an array of safe candidate identifiers.")
    for identifier in value:
        _identifier(identifier, name)
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not repeat identifiers.")
    return value


def _check_exclusions(obj):
    protected_ids = set(obj["source_ids"]) | {obj["id"], obj["unit_id"]}
    protected_names = {v.casefold() for v in obj["connected_parts"] + [obj["name"]]}
    for item in obj["excluded_objects"]:
        name = item if isinstance(item, str) else item["name"]
        identifier = item if isinstance(item, str) else item.get("id")
        if identifier in protected_ids or name.casefold() in protected_names:
            raise ValueError("excluded_objects cannot exclude this unit or its connected parts.")


def _candidate(source, style, margin):
    if not isinstance(source, dict):
        raise ValueError("Each candidate description must be an object.")
    obj = copy.deepcopy(source)
    _identifier(obj.get("id"))
    for field in ("boundary_cut", "suitable", "is_single"):
        if field in obj and type(obj[field]) is not bool:
            raise ValueError(f"{field} must be a real JSON boolean.")
    box = _box(obj.get("bbox"))
    if _edge(box, margin):
        raise ValueError("Actual bbox touches the source edge safety margin.")
    if obj.get("boundary_cut", False):
        raise ValueError("Source-cut formation: boundary_cut is true.")
    if obj.get("suitable", True) is False:
        raise ValueError("Candidate is explicitly unsuitable.")
    if isinstance(obj.get("name"), str) and _SCENERY.search(obj["name"]):
        raise ValueError("Grass tufts, scenery and unrelated environment layers are excluded.")
    obj["excluded_objects"] = _list(obj.get("excluded_objects", []), "excluded_objects", True)
    normalized = parse_selection(_dump({"style": style, "objects": [obj]}))["all_objects"][0]
    normalized["boundary_cut"] = False
    normalized["suitable"] = True
    _check_exclusions(normalized)
    return normalized


def _safe_units(plan, margin):
    originals = copy.deepcopy(plan["objects"])
    ids, valid, rejected, units = set(), [], {}, {}
    for source in originals:
        if not isinstance(source, dict):
            raise ValueError("Each candidate must be an object.")
        identifier = _identifier(source.get("id"))
        if identifier in ids:
            raise ValueError("Candidate ids must be unique.")
        ids.add(identifier)
        unit = source.get("unit_id", identifier)
        if isinstance(unit, str) and _ID.fullmatch(unit):
            units.setdefault(unit, []).append(identifier)
        try:
            valid.append(_candidate(source, plan["style"], margin))
        except ValueError as exc:
            rejected[identifier] = str(exc)
    # One clipped member makes its complete connected formation unsafe.
    for unit_ids in units.values():
        if any(identifier in rejected for identifier in unit_ids):
            for identifier in unit_ids:
                rejected.setdefault(identifier, "Connected unit includes an unsafe source candidate.")
    valid = [obj for obj in valid if obj["id"] not in rejected]
    grouped = {}
    for obj in valid:
        grouped.setdefault(obj["unit_id"], []).append(obj)
    accepted = []
    for members in grouped.values():
        try:
            whole = parse_selection(_dump({"style": plan["style"], "objects": members}))["all_objects"][0]
            whole["excluded_objects"] = _list(
                [item for obj in members for item in obj["excluded_objects"]], "excluded_objects", True)
            _check_exclusions(whole)
        except ValueError as exc:
            for obj in members:
                rejected[obj["id"]] = str(exc)
            continue
        source_objects = [copy.deepcopy(o) for o in originals if o["id"] in whole["source_ids"]]
        whole["original_candidates"] = source_objects
        whole["original_description"] = copy.deepcopy(source_objects[0])
        whole["source_bbox"] = whole["bbox"].copy()
        whole["boundary_cut"] = False
        whole["suitable"] = True
        accepted.append(whole)
    accepted.sort(key=lambda obj: -obj["priority"])
    rejected_objects = [
        {"id": obj["id"], "name": obj.get("name", ""), "reason": rejected[obj["id"]],
         "original_description": obj}
        for obj in originals if obj["id"] in rejected]
    return accepted, rejected_objects, originals


def _contact_sheet(image, originals, rejected_ids):
    columns = max(1, min(3, len(originals)))
    rows = max(1, math.ceil(len(originals)/columns))
    cw, ch, padding, label_height = 320, 256, 8, 32
    sheet = np.full((rows*ch, columns*cw, 3), .08, np.float32)
    tiles = []
    if not originals:
        header = np.full((label_height, cw, 3), 20, np.uint8)
        cv2.putText(header, "No candidates supplied", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    .6, (230, 230, 230), 1, cv2.LINE_AA)
        sheet[:label_height] = header.astype(np.float32)/255
    for index, obj in enumerate(originals):
        x, y = (index % columns)*cw, (index // columns)*ch
        label = obj["id"]
        rejected = label in rejected_ids
        # OpenCV 5 text requires uint8; do not quantize the actual float crops.
        header = np.full((label_height, cw, 3), 20, np.uint8)
        footer = np.full((32, cw, 3), 20, np.uint8)
        text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .6, 1)[0][0]
        font_scale = min(.6, .6*(cw-2*padding-2)/max(1, text_width))
        cv2.putText(header, label, (padding, 23), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (242, 242, 242), 1, cv2.LINE_AA)
        cv2.putText(footer, "REJECTED" if rejected else "SAFE", (padding, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, .45,
                    (242, 102, 89) if rejected else (102, 230, 128), 1, cv2.LINE_AA)
        sheet[y:y+label_height, x:x+cw] = header.astype(np.float32)/255
        sheet[y+ch-32:y+ch, x:x+cw] = footer.astype(np.float32)/255
        tile = {"id": label, "label": label, "accepted": not rejected,
                "sheet_label_pixels": [x, y, x+cw, y+label_height]}
        try:
            crop, info = crop_reference(image, obj.get("bbox"), 0)
            h, w = crop.shape[1:3]
            scale = min((cw-2*padding)/w, (ch-label_height-32)/h)
            dw, dh = max(1, round(w*scale)), max(1, round(h*scale))
            left = x+(cw-dw)//2
            top = y+label_height+(ch-label_height-32-dh)//2
            sheet[top:top+dh, left:left+dw] = cv2.resize(
                crop[0, ..., :3], (dw, dh), interpolation=cv2.INTER_AREA)
            tile.update(info)
            tile["sheet_crop_pixels"] = [left, top, left+dw, top+dh]
            tile["rescaled"] = [dw, dh] != [w, h]
        except ValueError:
            tile["sheet_crop_pixels"] = None
            tile["crop_error"] = "Invalid bbox; no crop fabricated."
        tiles.append(tile)
    return sheet[None], {"size": [sheet.shape[1], sheet.shape[0]], "tiles": tiles}


def candidate_review(plan_text, reference_numpy_BHWC, edge_margin=.01):
    """Return (RGB contact sheet float32 BHWC, candidate manifest JSON string).

    Invalid candidates are rejected with evidence. Fewer than three safe units
    yields status=insufficient_safe_candidates, including for an empty roster.
    The expected model roster is 6-9 candidates; no safe candidates are truncated.
    """
    margin = _number(edge_margin, "edge_margin", 0, .15)
    image = validate_image(reference_numpy_BHWC)
    if image.shape[0] != 1:
        raise ValueError("Exactly one current reference image is required.")
    plan = _plan(plan_text)
    accepted, rejected, originals = _safe_units(plan, margin)
    rejected_ids = [o["id"] for o in rejected]
    sheet, layout = _contact_sheet(image, originals, rejected_ids)
    count = len(accepted)
    report = {
        "schema_version": 1, "style": plan["style"],
        "status": "ready" if count >= 3 else "insufficient_safe_candidates",
        "source_size": [image.shape[2], image.shape[1]], "edge_margin": margin,
        "safe_unit_count": count, "accepted_candidates": accepted,
        "objects": accepted, "rejected_ids": rejected_ids, "rejected_objects": rejected,
        "original_objects": originals, "contact_sheet": layout,
        "diagnostics": [] if count >= 3 else [
            {"code": "insufficient_safe_candidates", "required": 3, "available": count,
             "message": "Review the current image; never invent units to fill branches."}],
        "grouping_rule": "Only explicit connected unit_id membership joins candidates; "
                         "2D overlap and occlusion alone never imply connection.",
    }
    return sheet.astype(np.float32, copy=False), _dump(report)


def _refined_box(box, source_box, margin):
    box = _box(box)
    if _edge(box, margin):
        raise ValueError("Refined bbox touches the source edge.")
    a, b = box, source_box
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(
        0, min(a[3], b[3])-max(a[1], b[1]))
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    iou = intersection/(area_a+area_b-intersection)
    if iou < .5 or any(abs(a[i]-b[i]) > .25*(b[i % 2+2]-b[i % 2])
                       for i in range(4)):
        raise ValueError("Refined bbox must remain on the actual safe candidate, not retarget it.")
    return box.copy()


def split_three(review_text, candidate_manifest_JSON, edge_margin=.01):
    """Return three single-object plan JSON strings, then a report JSON string.

    Reviews use style+objects, or objects with just safe id and optional refinements.
    Invalid count, unsafe identity, or insufficient safe units raises ValueError.
    """
    margin = _number(edge_margin, "edge_margin", 0, .15)
    manifest = _decode(candidate_manifest_JSON, "original_objects", 1000000)
    source_plan = {"style": manifest.get("style"), "objects": manifest["original_objects"]}
    # Revalidate original evidence, not reviewer-supplied safe flags or moved boxes.
    source_plan = _plan(_dump(source_plan))
    manifest_margin = _number(manifest.get("edge_margin"), "manifest edge_margin", 0, .15)
    margin = max(margin, manifest_margin)
    accepted, rejected, _ = _safe_units(source_plan, margin)
    if len(accepted) < 3:
        raise ValueError(_dump({"status": "insufficient_safe_candidates", "required": 3,
                               "available": len(accepted), "rejected_ids": [o["id"] for o in rejected]}))
    rejected_ids = set(_id_list(manifest.get("rejected_ids", []), "rejected_ids")) | {
        o["id"] for o in rejected}
    lookup = {identifier: obj for obj in accepted for identifier in obj["source_ids"]}
    review = _plan(review_text, require_style=False)
    if len(review["objects"]) != 3:
        raise ValueError("Visual review must select exactly three distinct safe candidate units.")
    selected, seen_ids, seen_units = [], set(), set()
    for choice in review["objects"]:
        if not isinstance(choice, dict):
            raise ValueError("Each reviewed selection must identify one actual safe candidate.")
        identifier = _identifier(choice.get("id"))
        if identifier in rejected_ids or identifier not in lookup:
            raise ValueError(f"Candidate {identifier} is rejected or not an actual safe source id.")
        source = lookup[identifier]
        if identifier in seen_ids or source["unit_id"] in seen_units:
            raise ValueError("Three distinct candidate ids AND connected units are required.")
        seen_ids.add(identifier)
        seen_units.add(source["unit_id"])
        for field in ("unit_id", "layer", "kind", "asset_unit"):
            if field in choice and choice[field] != source[field]:
                raise ValueError(f"Review cannot change the candidate's {field} identity.")
        if "source_ids" in choice:
            values = choice["source_ids"]
            if not isinstance(values, list) or not values or any(
                    not isinstance(v, str) or v not in source["source_ids"] or v in rejected_ids
                    for v in values):
                raise ValueError("Review source_ids must refer only to this actual safe unit.")
        target = copy.deepcopy(source)
        for field in _DESCRIPTION_FIELDS + ("boundary_cut", "suitable", "is_single", "priority"):
            if field in choice:
                target[field] = copy.deepcopy(choice[field])
        target["id"] = identifier
        target["bbox"] = _refined_box(choice.get("bbox", source["bbox"]), source["bbox"], margin)
        target["connected_parts"] = list(dict.fromkeys(
            source["connected_parts"] + _list(choice.get("connected_parts", []), "connected_parts")))
        target["excluded_objects"] = _list(
            source["excluded_objects"] + _list(choice.get("excluded_objects", []),
                                               "excluded_objects", True),
            "excluded_objects", True)
        _check_exclusions(target)
        _candidate(target, manifest["style"], margin)
        selected.append(target)
    outputs = [_dump({"schema_version": 1, "style": manifest["style"],
                      "source_size": manifest.get("source_size"), "edge_margin": margin,
                      "objects": [target]}) for target in selected]
    report = {"schema_version": 1, "status": "ready",
              "selected_ids": [o["id"] for o in selected],
              "selected_unit_ids": [o["unit_id"] for o in selected],
              "objects": selected, "rejected_ids": sorted(rejected_ids),
              "diagnostics": [], "bbox_refinement_rule": "IoU >= 0.5; each side moves at "
              "most 25% of its original axis span; source-edge rejection is irreversible."}
    return (*outputs, _dump(report))


def asset_reference(one_plan_JSON, reference_BHWC, filename_prefix, crop_context=.025):
    """Return (prompt str, real crop BHWC float32, key str, prefix str, record JSON).

    The crop keeps the reference's RGB/RGBA channels and is never rescaled.
    Original full-image descriptions remain prompt context after visual refinement.
    """
    context = _number(crop_context, "crop_context", 0, .15)
    if not isinstance(filename_prefix, str) or not filename_prefix.strip():
        raise ValueError("filename_prefix must be non-empty text.")
    plan = _plan(one_plan_JSON)
    if len(plan["objects"]) != 1:
        raise ValueError("Each asset branch requires exactly one complete target.")
    image = validate_image(reference_BHWC)
    if image.shape[0] != 1:
        raise ValueError("Exactly one current reference image is required.")
    size = [image.shape[2], image.shape[1]]
    if plan.get("source_size") is not None and plan["source_size"] != size:
        raise ValueError("Reference dimensions do not match the current-image plan.")
    margin = _number(plan.get("edge_margin", .01), "plan edge_margin", 0, .15)
    raw = plan["objects"][0]
    if not isinstance(raw, dict):
        raise ValueError("Asset plan requires one object.")
    target = _candidate(raw, plan["style"], margin)
    originals = raw.get("original_candidates", [raw.get("original_description", raw)])
    if not isinstance(originals, list) or not originals:
        raise ValueError("Original candidate descriptions must be a non-empty array.")
    original_targets = [_candidate(o, plan["style"], margin) for o in originals]
    original_ids = _id_list([o["id"] for o in original_targets], "original candidate ids")
    source_ids = _id_list(raw.get("source_ids", original_ids), "source_ids")
    if set(source_ids) != set(original_ids) or target["id"] not in original_ids or any(
            o["unit_id"] != target["unit_id"] or o["kind"] != target["kind"] or
            o["layer"] != target["layer"] for o in original_targets):
        raise ValueError("Original descriptions must belong to this actual connected source unit.")
    source_box = [min(o["bbox"][0] for o in original_targets),
                  min(o["bbox"][1] for o in original_targets),
                  max(o["bbox"][2] for o in original_targets),
                  max(o["bbox"][3] for o in original_targets)]
    if "source_bbox" in raw and _box(raw["source_bbox"]) != source_box:
        raise ValueError("source_bbox must match the original connected source unit.")
    target["bbox"] = _refined_box(target["bbox"], source_box, margin)
    target["source_ids"] = source_ids
    original_parts = [part for o in original_targets for part in o["connected_parts"]]
    target["connected_parts"] = list(dict.fromkeys(original_parts + target["connected_parts"]))
    target["excluded_objects"] = _list(
        [item for o in original_targets for item in o["excluded_objects"]] +
        target["excluded_objects"], "excluded_objects", True)
    _check_exclusions(target)
    crop, info = crop_reference(image, target["bbox"], context)
    key = choose_key(crop, "auto")
    prefix = object_filename(target, filename_prefix)
    descriptions = "\n".join(
        f"Source candidate {o['id']}: " + "; ".join(
            f"{field}: {o[field]}" for field in _DESCRIPTION_FIELDS)
        for o in originals)
    exclusions = "; ".join(
        item if isinstance(item, str) else _dump(item) for item in target["excluded_objects"])
    parts = "; ".join(target["connected_parts"]) or "all of this target's connected structure"
    prompt = (
        f"Extract exactly one COMPLETE connected terrain asset: {target['name']} "
        f"({target['layer']}, source id {target['id']}). The supplied image is the actual "
        "current-reference crop, not a new scene or an old full-image reference.\n"
        f"Original full-image context and descriptions:\n{descriptions}\n"
        f"Current reviewed location: {target['location']}. Appearance: {target['appearance']}. "
        f"Occlusion: {target['occlusion']}. Selection reason: {target['selection_reason']}.\n"
        f"Keep ALL connected parts together: {parts}. Attached caps, pillars, peaks, terraces "
        "and their shared natural stone base are ONE complete unit. Preserve their relative "
        "positions, proportions, height, depth and overlap; never extract only the tallest peak. "
        "Internal cracks do not divide an asset. Preserve the target's own grass-covered slope "
        "and stone shelf. 2D overlap with a separate depth object does not join it to the target.\n"
        f"Explicit excluded_objects: {exclusions or 'none specifically identified'}. Remove only "
        "unconnected occluders and surrounding scenery, including separate ridges, trees, grass "
        "tufts, water, sky, clouds, flat ground, shadows and text. Never remove attached parts.\n"
        "Complete hidden sides and a short natural complete base using only this formation's "
        "visible structure. Keep every peak, lateral edge, shelf and foot intact; fill occluder-shaped "
        "notches without inventing peaks, a long cliff skirt, a ground tray or a random variant.\n"
        f"Match the original reference style: {plan['style']}. Preserve color, silhouette, texture, "
        "lighting, perspective and aerial depth; do not turn stylized art into a photograph.\n"
        "Center the complete unit on a generous canvas. Subject width AND height must be at most "
        "70% of the canvas, leaving at least 15% empty safety border on EACH of all four sides. "
        "No clipped tips or base, no separate panels, no multiple subjects.\n"
        f"All non-target pixels must be uniform pure chroma key {key}, with no gradients, texture, "
        "checkerboard, reflections or cast shadows. The key belongs only to empty background; "
        "never tint the subject. Output a single complete asset image."
    )
    record = {**info, "schema_version": 1, "id": target["id"], "unit_id": target["unit_id"],
              "source_ids": source_ids,
              "bbox": target["bbox"], "source_bbox": source_box,
              "crop_context": context, "style": plan["style"],
              "original_description": raw.get("original_description", originals[0]),
              "original_candidates": originals, "connected_parts": target["connected_parts"],
              "excluded_objects": target["excluded_objects"], "key_color": key,
              "output_prefix": prefix, "canvas_safety_margin": .15}
    return prompt, crop, key, prefix, _dump(record)
