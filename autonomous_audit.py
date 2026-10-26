"""Fail-closed auditing for one connected terrain asset (NumPy BHWC, 0..1).

Plans are {"style": "...", "objects": [one selection.py target]}; optional
target.excluded_objects is a list of strings or objects with name and optional
id/location/appearance/reason. connected_parts is a list of part descriptions.
Reports accept JSON objects
or JSON strings, optionally wrapped in one JSON code fence, without prose,
duplicate keys, non-finite values or aliases. Visual reports have exactly the
seven VISUAL_GATES, score (number 0..100), and issues (blocking defect strings).
Numeric and selection records use schema_version=1. Candidate indices are
zero-based; report lists may also be JSON arrays of objects or JSON strings.
Selection records retain all numeric_reports/visual_reports in input order:
strings verbatim, JSON objects unchanged, non-finite objects as diagnostic text.
No model calls, file access or workflow construction occur here.
"""

import json
import math
import re

import cv2
import numpy as np

try:
    from .core import key_rgb, validate_image
    from .objects import crop_asset
    from .selection import parse_selection, selected_matte, selected_prompt
    from .foot_completion import foot_profile_problem
except ImportError:
    from core import key_rgb, validate_image
    from objects import crop_asset
    from selection import parse_selection, selected_matte, selected_prompt
    from foot_completion import foot_profile_problem


VISUAL_GATES = ("target_match", "complete_outline", "connected_parts_preserved",
                "natural_base", "no_unrelated_objects", "style_match", "clean_background")
NUMERIC_FIELDS = (
    "schema_version", "asset_id", "key_color", "valid", "diagnostic_only",
    "problems", "matte_error", "source_size", "source_bbox", "output_size",
    "meaningful_components", "touches_border", "removed_remote_pixels",
    "foreground_coverage", "warning", "observed_screen_rgb", "soft_edge_pixels",
    "mask_convention", "comparison_panels",
)


def _dump(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field: " + key)
        result[key] = value
    return result


def _finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("JSON numbers must be finite.")
    return value


def _bad_constant(text):
    raise ValueError("JSON numbers must be finite: " + text)


def _json(value, label):
    try:
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("```"):
                lines = text.splitlines()
                if len(lines) < 3 or lines[0] not in ("```json", "```") or lines[-1] != "```":
                    raise ValueError("Expected one JSON code fence without prose.")
                text = "\n".join(lines[1:-1])
            value = json.loads(text, object_pairs_hook=_unique_pairs,
                               parse_float=_finite_float, parse_constant=_bad_constant)
        # Also validate dict/list inputs; NaN and unsupported values cannot bypass parsing.
        _dump(value)
    except (ValueError, TypeError, OverflowError, RecursionError) as error:
        raise ValueError(label + " JSON: " + str(error)) from None
    return value


def _object(value, label):
    value = _json(value, label)
    if not isinstance(value, dict):
        raise ValueError(label + " must be a JSON object.")
    return value


def _one_plan(value):
    source = _object(value, "one_plan")
    if not isinstance(source.get("objects"), list) or len(source["objects"]) != 1:
        raise ValueError("one_plan must contain exactly one target in objects.")
    result = parse_selection(_dump(source))
    target = result["objects"][0]
    if "exclude_objects" in target:
        raise ValueError("Use excluded_objects, not the unsupported exclude_objects alias.")
    exclusions = target.get("excluded_objects", [])
    if not isinstance(exclusions, list) or len(exclusions) > 100:
        raise ValueError("excluded_objects must be a list of concise descriptions.")
    protected_names = {name.casefold() for name in target["connected_parts"] + [target["name"]]}
    protected_ids = set(target["source_ids"]) | {target["id"], target["unit_id"]}
    for item in exclusions:
        name = item.get("name") if isinstance(item, dict) else item
        if not isinstance(name, str) or not 0 < len(name.strip()) < 3000:
            raise ValueError("excluded_objects entries require a concise name or description.")
        identifier = item.get("id") if isinstance(item, dict) else item
        if isinstance(item, dict):
            if "id" in item and (not isinstance(identifier, str) or not re.fullmatch(
                    r"[A-Za-z0-9_-]{1,40}", identifier)):
                raise ValueError("excluded_objects id must be a safe identifier.")
            for field in ("location", "appearance", "reason"):
                if field in item and (not isinstance(item[field], str) or
                                      not 0 < len(item[field].strip()) < 3000):
                    raise ValueError("excluded_objects " + field + " must be concise text.")
        if identifier in protected_ids or name.strip().casefold() in protected_names:
            raise ValueError("excluded_objects cannot exclude the target or its connected_parts.")
    target["excluded_objects"] = exclusions
    return result, target


def _strings(value):
    return isinstance(value, list) and all(
        isinstance(item, str) and 0 < len(item.strip()) <= 2000 for item in value)


def _number(value, low=0, high=100):
    return type(value) in (int, float) and low <= value <= high and math.isfinite(value)


def _visual(value):
    report = _object(value, "visual score/schema")
    required = set(VISUAL_GATES) | {"score", "issues"}
    missing, extra = required - report.keys(), report.keys() - required
    if missing or extra:
        raise ValueError("visual schema missing fields: " + ", ".join(sorted(missing)) +
                         "; unexpected fields: " + ", ".join(sorted(extra)))
    for gate in VISUAL_GATES:
        if type(report[gate]) is not bool:
            raise ValueError("visual " + gate + " must be a true boolean, not text or a number.")
    if not _number(report["score"]):
        raise ValueError("visual score must be a finite number in 0..100.")
    if not _strings(report["issues"]):
        raise ValueError("visual issues must be a list of concrete nonempty defect strings.")
    problems = ["visual " + gate + " is false" for gate in VISUAL_GATES if not report[gate]]
    problems.extend("visual blocking issue: " + issue.strip() for issue in report["issues"])
    return report, problems


def _size(value, length=2, positive=True):
    return isinstance(value, list) and len(value) == length and all(
        type(v) is int and v >= (1 if positive else 0) for v in value)


def _numeric(value, rgba=None):
    report = _object(value, "numeric")
    missing = set(NUMERIC_FIELDS) - report.keys()
    if missing:
        raise ValueError("numeric schema missing fields: " + ", ".join(sorted(missing)))
    problems = []
    if type(report["schema_version"]) is not int or report["schema_version"] != 1:
        problems.append("numeric schema_version must be 1")
    if not isinstance(report["asset_id"], str) or not report["asset_id"].strip():
        problems.append("numeric asset_id is missing")
    try:
        key_rgb(report["key_color"])
    except (ValueError, TypeError):
        problems.append("numeric key_color is invalid")
    if report["valid"] is not True:
        problems.append("numeric valid must be true")
    if report["diagnostic_only"] is not False:
        problems.append("numeric diagnostic_only must be false")
    if not _strings(report["problems"]):
        problems.append("numeric problems must be a list of defect strings")
    else:
        problems.extend("numeric " + item for item in report["problems"])
    for field in ("matte_error", "warning"):
        if not isinstance(report[field], str) or report[field]:
            problems.append("numeric " + field + ": " + str(report[field]))
    if report["touches_border"] is not False:
        problems.append("numeric object touches canvas border or border status is unknown")
    if type(report["meaningful_components"]) is not int or report["meaningful_components"] != 1:
        problems.append("numeric meaningful_components must be exactly 1")
    if type(report["removed_remote_pixels"]) is not int or report["removed_remote_pixels"] != 0:
        problems.append("numeric recovery removed detached pixels; review the raw image")
    if not _number(report["foreground_coverage"], 0, 1) or report["foreground_coverage"] <= 0:
        problems.append("numeric foreground_coverage must be finite and positive")
    if type(report["soft_edge_pixels"]) is not int or report["soft_edge_pixels"] < 0:
        problems.append("numeric soft_edge_pixels must be a nonnegative integer")
    observed = report["observed_screen_rgb"]
    if not _size(observed, 3, False) or any(v > 255 for v in observed):
        problems.append("numeric observed_screen_rgb must be three RGB bytes")
    if report["mask_convention"] != "1=foreground opaque, 0=transparent":
        problems.append("numeric mask_convention must describe foreground alpha")
    source, output, box = report["source_size"], report["output_size"], report["source_bbox"]
    if not _size(source) or not _size(output):
        problems.append("numeric source_size and output_size must be positive pixel sizes")
    elif not _size(box, 4, False) or not (
            0 < box[0] < box[2] < source[0] and 0 < box[1] < box[3] < source[1]):
        problems.append("numeric source_bbox is invalid or touches canvas border")
    elif not (box[2]-box[0] <= output[0] <= source[0] and
              box[3]-box[1] <= output[1] <= source[1]):
        problems.append("numeric output_size cannot cut away the recovered outline")
    if rgba is not None and output != [rgba.shape[2], rgba.shape[1]]:
        problems.append("numeric output_size does not match candidate RGBA")
    return report, list(dict.fromkeys(problems))


def _comparison(source, raw, rgba, guide=None):
    alpha = rgba[0, ..., 3:4]
    yy, xx = np.indices(alpha.shape[:2])
    checker = np.where(((xx//16 + yy//16) % 2)[..., None], .88, .65)
    recovered = rgba[0, ..., :3]*alpha + checker*(1-alpha)
    pictures = (source[0, ..., :3], raw[0, ..., :3], recovered)
    labels = ("SOURCE CROP", "RAW GENERATED", "RECOVERED ALPHA")
    if guide is not None:
        pictures = (source[0, ..., :3], guide[0, ..., :3], raw[0, ..., :3], recovered)
        labels = ("SOURCE CROP", "EDITABLE TARGET GUIDE", "RAW GENERATED", "RECOVERED ALPHA")
    # Two columns avoid shrinking a 3168px strip at the vision carrier limit.
    # The extra full-width foot view magnifies defects; candidate pixels are untouched.
    width = max(256, min(768, max(p.shape[1] for p in pictures)))
    height = max(128, min(560, max(p.shape[0] for p in pictures)))
    margin, header = 12, 36
    cell = width + 2*margin
    row_height = height + header + 2*margin
    rows = (len(pictures)+1)//2
    ys, xs = np.where(alpha[..., 0] > .05)
    if len(ys):
        x0, x1, y1 = max(0, int(xs.min())-8), min(alpha.shape[1], int(xs.max())+9), min(
            alpha.shape[0], int(ys.max())+9)
        y0 = max(0, int(ys.min()) + int(.58*(int(ys.max())-int(ys.min())+1)))
    else:
        x0, x1, y0, y1 = 0, alpha.shape[1], alpha.shape[0]//2, alpha.shape[0]
    detail = recovered[y0:y1, x0:x1]
    detail_width = 2*cell-2*margin
    scale = min(1., detail_width/detail.shape[1], 460/detail.shape[0])
    dw, dh = max(1, round(detail.shape[1]*scale)), max(1, round(detail.shape[0]*scale))
    sheet = np.full((rows*row_height+header+2*margin+dh, 2*cell, 3), .16, np.float32)
    panels = []
    for index, (picture, label) in enumerate(zip(pictures, labels)):
        h, w = picture.shape[:2]
        scale = min(1., width/w, height/h)
        dw, dh = max(1, round(w*scale)), max(1, round(h*scale))
        col, row = index % 2, index // 2
        x = col*cell + margin + (width-dw)//2
        y = row*row_height + header + margin + (height-dh)//2
        resized = picture if (dw, dh) == (w, h) else cv2.resize(
            picture, (dw, dh), interpolation=cv2.INTER_AREA)
        sheet[y:y+dh, x:x+dw] = resized
        strip = np.full((header, cell, 3), 41, np.uint8)
        cv2.putText(strip, label, (margin, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (242, 242, 242), 1, cv2.LINE_AA)
        sheet[row*row_height:row*row_height+header, col*cell:(col+1)*cell] = strip/255
        panels.append({"label": label, "input_size": [w, h],
                       "display_bbox": [x, y, x+dw, y+dh]})
    scale = min(1., detail_width/detail.shape[1], 460/detail.shape[0])
    dw, dh = max(1, round(detail.shape[1]*scale)), max(1, round(detail.shape[0]*scale))
    x, y = (2*cell-dw)//2, rows*row_height+header+margin
    sheet[y:y+dh, x:x+dw] = cv2.resize(detail, (dw, dh), interpolation=cv2.INTER_AREA)
    strip = np.full((header, 2*cell, 3), 41, np.uint8)
    cv2.putText(strip, "LOWER FOOT DETAIL - check trees, cut planes and base volume",
                (margin, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (242, 242, 242), 1, cv2.LINE_AA)
    sheet[rows*row_height:rows*row_height+header] = strip/255
    panels.append({"label": "LOWER FOOT DETAIL", "source_panel": "RECOVERED ALPHA",
                   "source_bbox": [x0, y0, x1, y1], "input_size": [x1-x0, y1-y0],
                   "display_bbox": [x, y, x+dw, y+dh]})
    return sheet[None], panels


def _review_prompt(plan, target, report):
    schema = {**dict.fromkeys(VISUAL_GATES, False), "score": 0,
              "issues": ["Concrete blocking defect, with location and missing/extraneous part"]}
    contract = {field: target[field] for field in (
        "name", "location", "appearance", "occlusion", "connected_parts")}
    contract["excluded_objects"] = target["excluded_objects"]
    return (
        "Review one connected terrain asset against its actual reference crop. "
        "Inspect LOWER FOOT DETAIL first, then compare the two-column full views. "
        "The detail is an unchanged enlarged crop of recovered alpha, not a separate asset. "
        "Look for a forest silhouette: rows of tiny conifer crowns, jagged tree silhouettes "
        "or a blue/green forest ribbon across the lower slopes. These are unrelated trees "
        "even when pale, fog-colored, touching the mountain or copied from the source. "
        "If actually visible, no_unrelated_objects=false; do not label them rock texture. "
        "Inspect left and right feet for side cut planes, a paper-thin polygon slice, "
        "straight chopped endpoints or a foreground occlusion border. Such actual cuts "
        "require natural_base=false and complete_outline=false even when the object has "
        "transparent margins and is not touching the canvas. Reconstructing a foot means "
        "continuous sloping rock/snow volume, not adding margin around a cutout. "
        "A visible horizontal pasted color strip or seam requires style_match=false. "
        "Panels: SOURCE CROP, RAW GENERATED, RECOVERED ALPHA (checker composite). "
        "An EDITABLE TARGET GUIDE panel, when present, identifies the selected source "
        "mountain; the source crop includes other-depth neighbors. Do not substitute "
        "a neighboring upper-left mountain for this identified target. The guide is rough, "
        "so judge correct visible source geometry but allow stray fragments to be removed "
        "and hidden bases to be completed. Compare every panel; a diagnostic tensor is not an asset.\n"
        "Target location descriptions refer to the original full scene, not positions "
        "within this cropped panel. Identify the target using the actual source crop and "
        "its appearance and connected parts; do not reinterpret global location as a "
        "different object at the crop's upper-left corner.\n"
        "Expected transformations: isolation on a screen/transparent background, centering and "
        "uniform rescaling are intended. A hidden base must be plausibly completed: AI inference "
        "is required and is not a defect by itself. Do not require original foreground occluders, "
        "trees, water, sky, clouds or distant formations to remain. Judge the visible target's "
        "own silhouette, peak count, slopes, material and light; judge completed areas for "
        "plausibility, smooth natural feet and no new peaks, sharp cut plane or ground tray. "
        "A separate localized guide may accompany the sheet; its polygon lower boundary is "
        "not a finished outline and must not be copied as a flat cut. "
        "Report only a defect actually visible in these images, not a hypothetical risk. "
        "A chroma background alone is not spill; compare the recovered alpha edges. "
        "The panels contain nonblack image pixels when numeric source/output dimensions are "
        "positive; look closely before claiming that they are entirely black.\n"
        "Return only this fixed JSON schema (no aliases, extra fields or prose; a single "
        "JSON code fence is allowed). Use true booleans, never strings or integers; "
        "score is a finite numeric value in 0..100, and issues is a concrete list:\n" +
        _dump(schema) + "\n"
        "target_match: exactly the planned reference unit, not a nearby or invented formation. "
        "complete_outline: all peaks, sides and feet intact with no canvas clipping. "
        "connected_parts_preserved: every attached peak, pillar and shared ledge retained; "
        "overlapping objects at a different depth are not connected parts. "
        "natural_base: a short natural base/rock foot inferred only from the target, "
        "without rectangular notches, holes shaped like occluders or flat sliced bases. "
        "no_unrelated_objects: no trees, separate terrain, ground trays or other-depth objects. "
        "style_match: preserve reference silhouette, relative proportions, materials, colors, "
        "lighting, viewpoint and atmospheric perspective. "
        "clean_background: uniform screen in the raw image, clean soft alpha without key spill.\n"
        "List only blocking defects in issues; use [] if none. Any non-empty issues rejects "
        "the candidate even if all flags are true. A high score cannot override a false gate "
        "or a numeric failure. Never change numeric valid:false into acceptance.\n"
        "Target: " + _dump(contract) + "\nReference style: " + plan["style"] +
        "\nIndependent numeric report: " + _dump(report)
    )


def inspect_asset(image, reference_crop, plan_text, key, softness=.06, padding=24,
                  target_guide=None):
    """Return comparison RGB, visual-review prompt, candidate RGBA and numeric JSON.

    Numeric recovery failures return diagnostic_only:true and valid:false. Invalid
    input shapes, plans and parameters remain explicit programming errors.
    """
    plan, target = _one_plan(plan_text)
    image, reference_crop = validate_image(image), validate_image(reference_crop)
    if image.shape[0] != 1 or reference_crop.shape[0] != 1:
        raise ValueError("Exactly one generated image and one reference crop are required.")
    key_rgb(key)
    if not _number(softness, 0, .2):
        raise ValueError("softness must be a finite number from 0 to 0.2.")
    if type(padding) is not int or not 0 <= padding <= 256:
        raise ValueError("padding must be an integer between 0 and 256.")
    matte_error, problems = "", []
    try:
        rgba, quality = selected_matte(image, key, softness, padding, strict=False)
        if quality["warning"]:
            problems.append(quality["warning"])
        if quality["removed_remote_pixels"]:
            problems.append("Matte recovery removed detached pixels; inspect raw image and regenerate.")
        outline_problem = foot_profile_problem(rgba)
        if outline_problem:
            problems.append(outline_problem)
    except (ValueError, cv2.error) as error:
        matte_error = str(error)
        problems.append(matte_error)
        # Preserve raw RGB for diagnosis, without claiming a successful recovered mask.
        alpha = image[..., 3:4] if image.shape[-1] == 4 else np.ones(image.shape[:-1]+(1,), np.float32)
        rgba = np.concatenate((image[..., :3], alpha), axis=-1).astype(np.float32)
        quality = {
            "source_size": [image.shape[2], image.shape[1]], "source_bbox": None,
            "output_size": [image.shape[2], image.shape[1]], "meaningful_components": 0,
            "touches_border": None, "removed_remote_pixels": 0,
            "foreground_coverage": float((alpha > .05).mean()),
            "warning": matte_error, "observed_screen_rgb": None,
            "soft_edge_pixels": int(((alpha > 0) & (alpha < 1)).sum()),
            "mask_convention": "1=foreground opaque, 0=transparent",
        }
    if target_guide is not None:
        target_guide = validate_image(target_guide)
        if target_guide.shape[0] != 1:
            raise ValueError("Exactly one current editable target guide is required.")
    sheet, panels = _comparison(reference_crop, image, rgba, target_guide)
    report = {
        **quality, "schema_version": 1, "asset_id": target["id"], "key_color": key,
        "valid": not problems, "diagnostic_only": bool(problems),
        "problems": problems, "matte_error": matte_error, "comparison_panels": panels,
        "connected_parts": target["connected_parts"], "excluded_objects": target["excluded_objects"],
    }
    # Apply the same numeric gates used by selection; warnings never become success.
    if report["valid"]:
        _, numeric_problems = _numeric(report, rgba)
        if numeric_problems:
            report.update(valid=False, diagnostic_only=True, problems=numeric_problems)
    return sheet, _review_prompt(plan, target, report), rgba, _dump(report)


def repair_prompt(plan_text, key, numeric_report, visual_report, attempt):
    """Compile generic feedback, retaining malformed audits as blocking failures."""
    plan, target = _one_plan(plan_text)
    key_rgb(key)
    if type(attempt) is not int or attempt < 1:
        raise ValueError("attempt must be a positive integer.")
    try:
        _, numeric_problems = _numeric(numeric_report)
    except ValueError as error:
        numeric_problems = [str(error)]
    try:
        visual, visual_problems = _visual(visual_report)
    except ValueError as error:
        visual, visual_problems = None, [str(error)]
    feedback = {"numeric_problems": numeric_problems, "visual_problems": visual_problems,
                "visual_report": visual}
    contract = {field: target[field] for field in (
        "name", "appearance", "occlusion", "connected_parts", "excluded_objects")}
    return (
        f"重新从本次原始局部参考图提取完整目标，背景输出均匀纯色{key}。"
        "输入是本次原图产生的可编辑目标参考，不是上一版成图，也没有必须保留的错误遮罩。"
        "根据下方目标结构与审查反馈纠正整个目标轮廓：删除混入的其他深度残片，"
        "恢复真实相连的侧峰岩肩，并补全被树木、前排地形遮挡的自然山脚。"
        "目标自己的峰形、积雪、岩层、颜色和空气透视遵照原图；不增加新峰、"
        "树木、地面托盘或浮空岩基。底部是有体积的短自然岩脚，不是直线平切。\n"
        "Regenerate exactly this reference target from the editable source-pixel guide "
        "derived from the actual original reference crop, "
        "with the same "
        "visible peak count, slopes, proportions, viewpoint, color and soft aerial perspective. "
        "Correct the entire target silhouette when the previous localization or upper "
        "geometry was wrong. Do not freeze an unrelated fragment or omit a connected "
        "side peak because it appeared in a previous guide. Global location descriptions "
        "refer to the full original scene; identify the same unit inside the supplied crop. "
        "Complete the hidden lower region with a short natural sloping foot, not a "
        "straight or paper-thin cut. Preserve the correct visible target geometry. "
        "Do not sharpen, darken, increase contrast, add vertical columns, new peaks or redesign "
        "the formation. Preserve painted reference texture. Target contract:\n" +
        _dump(contract) + "\nOriginal style: " + plan["style"] + "\n"
        f"Repair attempt {attempt}. Previous candidate is not accepted as final until "
        "every numeric and semantic gate passes. Treat the following as structured "
        "blocking feedback, not permission to weaken checks:\n" + _dump(feedback) + "\n"
        "If clipping or a border cut is reported, regenerate the complete silhouette and "
        "scale down/recenter the whole connected unit. Keep width and height at most 70% "
        "of the canvas, with at least 15% pure screen margin on all four sides. "
        "Do not crop off defects or pad a cut fragment to pretend it is complete.\n"
        "Restore any missing connected peak or shared natural ledge ONLY when listed in "
        "the target contract, in their original relative positions, proportions and depth relationships. "
        "Complete only the target's own short natural base and rock foot using its structure. "
        "Avoid rectangular notches, occluder-shaped holes, flat sliced bases, thin paper bases "
        "and invented wide ground platforms. Remove any forest silhouette or tree-crown "
        "ribbon across the lower slope, including pale blue distant trees touching the "
        "mountain. Reconstruct the area beneath them as continuous target rock/snow. "
        "Remove artificial side cut planes and reconstruct both lower shoulders as "
        "short curved natural rock feet with thickness; no straight chopped endpoints.\n"
        "Remove trees, depth-mismatched terrain, separate overlapping formations, external "
        "ground and unrelated objects. Do not remove attached rock parts or the target's own "
        "natural surface. Explicit excluded_objects: " + _dump(target["excluded_objects"]) + "\n"
        "Preserve reference fidelity: silhouette, connected parts, proportions, color, "
        "texture, stylization, lighting, viewpoint and atmospheric perspective. "
        "Do not invent new peaks, random variants or a new scene. Retain already-correct "
        "features; a later repair is not automatically better.\n"
        f"Output one complete centered unit on uniform pure {key}, no text, gradients, "
        "shadows, checkerboard or key contamination. It must pass fresh numeric and visual "
        "review; malformed reports require a fresh fixed-schema review, not automatic approval."
    )


def _candidate(value):
    try:
        rgba = np.asarray(value, dtype=np.float32)
    except (ValueError, TypeError, OverflowError):
        raise ValueError("candidate RGBA must be a numeric BHWC tensor.") from None
    if (rgba.ndim != 4 or rgba.shape[0] != 1 or rgba.shape[-1] != 4 or
            min(rgba.shape[1:3]) < 1 or not np.isfinite(rgba).all() or
            np.any(rgba < 0) or np.any(rgba > 1)):
        raise ValueError("candidate RGBA must be finite [1,H,W,4] in 0..1.")
    alpha = rgba[..., 3]
    if not np.any(alpha > .5) or not np.any(alpha < .01):
        raise ValueError("candidate RGBA must contain visible foreground and transparent background.")
    # Validate the tensor independently; do not deliver the helper's cropped/cleaned copy.
    _, quality = crop_asset(rgba, padding=0)
    if quality["meaningful_components"] != 1 or quality["removed_remote_pixels"]:
        raise ValueError("candidate RGBA must retain exactly one connected unit without removed parts.")
    return rgba


def _report_list(value, label):
    if isinstance(value, str):
        value = _json(value, label)
    if not isinstance(value, list):
        raise ValueError(label + " must be a JSON array.")
    return value


def _raw_record(value):
    if isinstance(value, str):
        return value
    try:
        _dump(value)
    except (ValueError, TypeError, OverflowError, RecursionError):
        # Invalid JSON numbers remain evidence as text, never accepted JSON numbers.
        return json.dumps(value, ensure_ascii=True, allow_nan=True, default=str)
    return value


def choose_asset(rgba_candidates, numeric_reports, visual_reports, min_score=85):
    """Select the highest-scored eligible version, retaining the earliest score tie.

    Every candidate must pass numeric and semantic gates independently. Malformed
    per-candidate reports reject only that candidate. Misaligned inputs are errors.
    Returns an unmodified RGBA copy and a JSON selection record; never a fallback.
    The record includes every original numeric/visual report, including rejections.
    """
    if not _number(min_score):
        raise ValueError("min_score must be a finite number in 0..100.")
    if not isinstance(rgba_candidates, (list, tuple)) or not rgba_candidates:
        raise ValueError("At least one candidate RGBA is required.")
    numeric_reports = _report_list(numeric_reports, "numeric_reports")
    visual_reports = _report_list(visual_reports, "visual_reports")
    if len(rgba_candidates) != len(numeric_reports) or len(rgba_candidates) != len(visual_reports):
        raise ValueError("Candidate RGBA, numeric and visual lists must have equal lengths.")
    eligible, rejected = [], []
    for index, (candidate, numeric_value, visual_value) in enumerate(zip(
            rgba_candidates, numeric_reports, visual_reports)):
        problems, rgba, numeric, visual = [], None, None, None
        try:
            rgba = _candidate(candidate)
        except ValueError as error:
            problems.append(str(error))
        try:
            numeric, numeric_problems = _numeric(numeric_value, rgba)
            problems.extend(numeric_problems)
        except ValueError as error:
            problems.append(str(error))
        try:
            visual, visual_problems = _visual(visual_value)
            problems.extend(visual_problems)
            if visual["score"] < min_score:
                problems.append(f"visual score {visual['score']} is below min_score {min_score}")
        except ValueError as error:
            problems.append(str(error))
        if problems:
            rejected.append({"index": index, "reasons": list(dict.fromkeys(problems))})
        else:
            eligible.append((index, rgba, numeric, visual))
    if not eligible:
        summary = "; ".join(f"candidate {r['index']}: " + ", ".join(r["reasons"])[:240]
                            for r in rejected[:3])
        raise ValueError("No eligible asset: " + summary)
    index, rgba, numeric, visual = max(eligible, key=lambda entry: entry[3]["score"])
    record = {
        "schema_version": 1, "selected_index": index, "asset_id": numeric["asset_id"],
        "score": visual["score"], "min_score": min_score,
        "eligible_indices": [entry[0] for entry in eligible], "rejected_candidates": rejected,
        "numeric_reports": [_raw_record(value) for value in numeric_reports],
        "visual_reports": [_raw_record(value) for value in visual_reports],
    }
    return rgba.copy(), _dump(record)
