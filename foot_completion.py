"""Scoped native-image completion with immutable visible reference geometry."""
import cv2
import numpy as np
try:
    from .core import key_rgb, validate_image
except ImportError:
    from core import key_rgb, validate_image


def completion_context(guide_image, key):
    image = validate_image(guide_image)
    if image.shape[0] != 1:
        raise ValueError("One guide is required.")
    color = key_rgb(key)
    h, w = image.shape[1:3]
    foreground = np.linalg.norm(image[0, ..., :3]-color, axis=-1) > .08
    ys, xs = np.where(foreground)
    if not len(xs):
        raise ValueError("Completion guide is empty.")
    x0, x1, y0, y1 = xs.min(), xs.max()+1, ys.min(), ys.max()+1
    start = int(y0+.62*(y1-y0))
    center = range(x0+round((x1-x0)*.2), x1-round((x1-x0)*.2))
    bottoms = [np.where(foreground[:, x])[0][-1] for x in center if foreground[:, x].any()]
    if bottoms:
        # Occluder notches can rise above a fixed edit rectangle. Start above
        # the shallow central lower contour while preserving the upper mass.
        start = min(start, int(np.quantile(bottoms, .10)-.08*(y1-y0)))
    start = max(y0+round((y1-y0)*.35), start)
    end = min(h-2, y1+max(8, round((y1-y0)*.16)))
    left = max(2, x0-round((x1-x0)*.04))
    right = min(w-2, x1+round((x1-x0)*.04))
    mask = np.zeros((h, w), np.float32)
    mask[start:end, left:right] = 1
    feather = max(3, min(32, round((y1-y0)*.10)))
    mask[start:start+feather, left:right] *= np.linspace(0, 1, feather)[:, None]
    marked = image.copy()
    marker = np.array([1., .22, .04], np.float32)
    marked[0, ..., :3] = image[0, ..., :3]*(1-mask[..., None])+marker*mask[..., None]
    return marked, mask[None]


def complete_foot(generated_image, source_guide, foot_mask, key):
    generated, source = validate_image(generated_image), validate_image(source_guide)
    if generated.shape[0] != 1 or source.shape[0] != 1:
        raise ValueError("One generated image and one source guide are required.")
    color = key_rgb(key)
    mask = np.asarray(foot_mask, np.float32)
    if (mask.shape != source.shape[:3] or not np.isfinite(mask).all() or
            np.any(mask < 0) or np.any(mask > 1) or not np.any(mask > .5) or
            np.any(mask[:, 0]) or np.any(mask[:, -1]) or
            np.any(mask[:, :, 0]) or np.any(mask[:, :, -1])):
        raise ValueError("Foot mask must be finite BHW, scoped inside the canvas.")
    h, w = generated.shape[1:3]
    resized = cv2.resize(source[0, ..., :3], (w, h), interpolation=cv2.INTER_CUBIC)
    resized = np.clip(resized, 0, 1)
    weight = cv2.resize(mask[0], (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
    pixels = generated[0, ..., :3].copy()
    source_fg = np.linalg.norm(resized-color, axis=-1) > .2
    generated_fg = np.linalg.norm(pixels-color, axis=-1) > .2
    overlap = ((weight[..., 0] > .05) & (weight[..., 0] < .8) &
               source_fg & generated_fg)
    offset = np.zeros(3, np.float32)
    if int(overlap.sum()) >= 16:
        offset = np.clip(np.median((resized-pixels)[overlap], axis=0), -.12, .12)
        pixels[generated_fg] = np.clip(pixels[generated_fg]+offset, 0, 1)
    completed = pixels*weight+resized*(1-weight)
    report = {"schema_version": 1, "original_upper_geometry_preserved": True,
              "generated_foot_pixels": int((weight > .01).sum()),
              "source_size": [source.shape[2], source.shape[1]], "output_size": [w, h],
              "foot_transition_color_offset": offset.tolist(),
              "key_color": key, "requires_fresh_numeric_and_visual_audit": True,
              "operation": "Only the fixed workflow's lower-foot edit mask uses generated pixels."}
    return np.clip(completed[None], 0, 1).astype(np.float32), report


def foot_profile_problem(rgba):
    """Reject deep lower concavities characteristic of an unfilled occluder."""
    alpha = np.asarray(rgba)[0, ..., 3] > .5
    ys, xs = np.where(alpha)
    if not len(xs):
        return ""
    x0, x1, y0, y1 = xs.min(), xs.max()+1, ys.min(), ys.max()+1
    crop = alpha[y0:y1, x0:x1].astype(np.uint8)
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ""
    hull = np.zeros_like(crop)
    cv2.fillConvexPoly(hull, cv2.convexHull(np.concatenate(contours)), 1)
    width, height = x1-x0, y1-y0
    defect = np.zeros(width, np.uint8)
    for x in range(round(width*.05), round(width*.95)):
        actual, envelope = np.where(crop[:, x])[0], np.where(hull[:, x])[0]
        if len(actual) and len(envelope) and envelope[-1]-actual[-1] > max(4, height*.13):
            defect[x] = 1
    runs = np.diff(np.r_[0, defect, 0].astype(np.int8))
    lengths = np.where(runs == -1)[0]-np.where(runs == 1)[0]
    if len(lengths) and lengths.max() > max(4, width*.03):
        return ("Lower silhouette has a deep inward notch consistent with an uncompleted "
                "occluder. Fill the target's own continuous natural rock foot, without "
                "adding a tray or cutting off the slope.")
    return ""
