"""Identify the chosen source unit visually, without altering the real crop."""
import json
import cv2
import numpy as np

try:
    from .core import validate_image
except ImportError:
    from core import validate_image


def localization_view(reference_crop, source_record):
    image = validate_image(reference_crop)
    record = json.loads(source_record)
    if image.shape[0] != 1:
        raise ValueError("Localization needs one current crop.")
    sw, sh = record["source_size"]
    x0, y0, x1, y1 = record["source_bbox_pixels"]
    box = record["bbox"]
    point = record.get("target_point")
    if point is None:
        point = [(box[0]+box[2])*500, (box[1]+box[3])*500]
    if (not isinstance(point, list) or len(point) != 2 or
            any(type(v) not in (int, float) or not np.isfinite(v) for v in point) or
            not box[0]*1000 <= point[0] <= box[2]*1000 or
            not box[1]*1000 <= point[1] <= box[3]*1000):
        raise ValueError("target_point must lie inside the actual chosen target bbox.")
    h, w = image.shape[1:3]
    if [w, h] != [x1-x0, y1-y0]:
        raise ValueError("Current crop dimensions do not match source record.")
    cx, cy = round(point[0]*sw/1000-x0), round(point[1]*sh/1000-y0)
    if not 0 <= cx < w or not 0 <= cy < h:
        raise ValueError("target_point is outside current crop.")
    marked = np.round(image[0, ..., :3]*255).astype(np.uint8).copy()
    radius = max(4, min(w, h)//35)
    cv2.circle(marked, (cx, cy), radius, (255, 48, 35), 2, cv2.LINE_AA)
    cv2.line(marked, (cx-radius-5, cy), (cx+radius+5, cy), (255, 48, 35), 1)
    cv2.line(marked, (cx, cy-radius-5), (cx, cy+radius+5), (255, 48, 35), 1)
    note = {**record, "guidance_only": True,
            "target_point_crop_pixels": [cx, cy],
            "target_point_crop_1000": [round(cx/w*1000), round(cy/h*1000)],
            "marker_is_not_subject": True}
    prompt = (
        f"Localize asset_id {record['id']} in this CURRENT source crop. The red circle/cross "
        f"is a target identity marker, not part of the rock; its crop position is "
        f"{note['target_point_crop_1000']} on the 0..1000 grid. Select the mountain/rock "
        "containing this point and ALL its genuinely attached parts, not a different "
        "peak at the crop's upper-left or another depth. Ignore/remove the marker itself. "
        "Trace the visible outline tightly with 24..60 clockwise polygon points; include "
        "all true side shoulders and peaks, exclude sky above slopes, trees, water and "
        "foreground occluders. Do not use a loose convex hull across sky gaps or "
        "another mountain. Stop at the occlusion boundary; the hidden foot is completed later. "
        "Return only JSON with asset_id, visible_polygon, description, hidden_base. "
        "All polygon coordinates use the whole supplied crop's 0..1000 integer grid. "
        f"Connected parts: {json.dumps(record['connected_parts'], ensure_ascii=False)}. "
        f"Excluded objects: {json.dumps(record['excluded_objects'], ensure_ascii=False)}."
    )
    return marked[None].astype(np.float32)/255, prompt, json.dumps(note, ensure_ascii=True)
