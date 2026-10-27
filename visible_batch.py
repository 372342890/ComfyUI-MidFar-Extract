"""Shared transparent processing with bounded noise tolerance and duplicate checks."""
import math

import cv2
import numpy as np

try:
    from .selection import selected_matte
    from .core import dominance, validate_image
except ImportError:
    from selection import selected_matte
    from core import dominance, validate_image


def _normalized_asset(rgba):
    pixels = rgba[0]
    alpha = pixels[..., 3] > .5
    ys, xs = np.where(alpha)
    if not len(xs):
        raise ValueError("透明主体为空。")
    pixels = pixels[ys.min():ys.max()+1, xs.min():xs.max()+1]
    height, width = pixels.shape[:2]
    scale = 128 / max(height, width)
    target = (max(1, round(width*scale)), max(1, round(height*scale)))
    asset = cv2.resize(pixels, target, interpolation=cv2.INTER_AREA)
    resized = np.zeros((128, 128, 4), np.float32)
    left, top = (128-target[0])//2, (128-target[1])//2
    resized[top:top+target[1], left:left+target[0]] = asset
    return resized[..., :3] * resized[..., 3:4], resized[..., 3] > .5


def _variant_signature(rgba, key):
    pixels = rgba[0]
    core = pixels[..., 3] > .99
    if key != "#FFFFFF":
        core &= dominance(pixels[..., :3], key) < .1
    ys, xs = np.where(core)
    if not len(xs):
        return None
    # A changed support/base or key-colored mist must not turn the same main
    # formation into a new asset. Compare its upper structure separately.
    bottom = max(int(ys.min())+1, round(ys.min()+(ys.max()+1-ys.min())*.7))
    core[bottom:] = False
    ys, xs = np.where(core)
    crop = pixels[ys.min():ys.max()+1, xs.min():xs.max()+1, :3]
    mask = core[ys.min():ys.max()+1, xs.min():xs.max()+1]
    height, width = mask.shape
    dims = (max(1, round(width*128/max(height, width))),
            max(1, round(height*128/max(height, width))))
    color = np.zeros((128, 128, 3), np.float32)
    silhouette = np.zeros((128, 128), np.uint8)
    left, top = (128-dims[0])//2, (128-dims[1])//2
    color[top:top+dims[1], left:left+dims[0]] = cv2.resize(crop*mask[..., None], dims)
    silhouette[top:top+dims[1], left:left+dims[0]] = cv2.resize(
        mask.astype(np.uint8), dims, interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(cv2.resize(color, (32, 32)), cv2.COLOR_RGB2GRAY)
    low_frequency = cv2.dct(gray)[:8, :8].ravel()[1:]
    perceptual_hash = low_frequency > np.median(low_frequency)
    hsv = cv2.cvtColor(np.round(np.clip(crop, 0, 1)*255).astype(np.uint8),
                       cv2.COLOR_RGB2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], mask.astype(np.uint8),
                             [18, 8], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram)
    return silhouette.astype(bool), perceptual_hash, histogram


def _white_asset(source):
    pixels = validate_image(source)
    if pixels.shape[0] != 1:
        raise ValueError("每路只接受一张实际提取图。")
    rgb = pixels[..., :3].copy()
    if pixels.shape[-1] == 4:
        alpha = pixels[..., 3:4]
        rgb = rgb * alpha + (1 - alpha)
    # This mask is only for content checks. Pale rock and snow pixels in the
    # delivered RGB image remain intact, including pure-white interior pixels.
    mask = np.max(1 - rgb[0], axis=-1) > .08
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > 16]
    mask = np.isin(labels, keep)
    if not mask.any():
        raise ValueError("白底提取图没有可识别的主体。")
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        raise ValueError("主体触及画布边界，可能被裁断或背景不是白底。")
    comparison = np.concatenate((rgb, mask[None, ..., None].astype(np.float32)), axis=-1)
    quality = {
        "background": "white", "alpha_required": False,
        "source_pixels_preserved": pixels.shape[-1] == 3,
        "comparison_mask_only": True, "foreground_pixels": int(mask.sum()),
        "removed_remote_pixels": 0, "warning": "",
    }
    return rgb, comparison, quality


def process_three(images, key):
    if not isinstance(images, (list, tuple)) or len(images) != 3:
        raise ValueError("需要三张实际生成的图片。")
    outputs, records, signatures, variants = [], [], [], []
    for index, source in enumerate(images, 1):
        try:
            if key == "#FFFFFF":
                output, rgba, quality = _white_asset(source)
            else:
                rgba, quality = selected_matte(source, key, .06, 24, strict=False)
                output = rgba
        except ValueError as error:
            stage = "白底内容检查" if key == "#FFFFFF" else "透明检查"
            raise ValueError(f"第{index}路{stage}失败：{error}") from error
        limit = min(16, math.floor(np.asarray(source).shape[1] *
                                   np.asarray(source).shape[2] * .00005))
        removed = quality["removed_remote_pixels"]
        if quality["warning"] or removed > limit:
            reason = quality["warning"] or (
                f"远离主体的碎片有{removed}像素，超过{limit}像素的小噪点容错范围。")
            raise ValueError(f"第{index}路透明检查失败：{reason}")
        quality.update(
            branch=index,
            cleaned_remote_pixels=removed,
            cleanup_limit_pixels=limit,
            status="warning" if removed else "success",
        )
        if removed:
            quality["warning"] = f"已清理{removed}个远离主体的小噪点像素。"
        signatures.append(_normalized_asset(rgba))
        variants.append(_variant_signature(rgba, key))
        outputs.append(output)
        records.append(quality)

    for first in range(3):
        color_a, mask_a = signatures[first]
        for second in range(first+1, 3):
            color_b, mask_b = signatures[second]
            union = mask_a | mask_b
            overlap = float((mask_a & mask_b).sum()) / max(1, int(union.sum()))
            color_error = float(np.abs(color_a[union]-color_b[union]).mean())
            if overlap >= .995 and color_error <= .008:
                raise ValueError(
                    f"第{first+1}路与第{second+1}路成图重复或近乎相同，"
                    "请查看识别与PE是否指向同一目标。")
            a, b = variants[first], variants[second]
            if a is None or b is None:
                continue
            upper_union = a[0] | b[0]
            upper_overlap = float((a[0] & b[0]).sum()) / max(1, int(upper_union.sum()))
            hash_distance = int(np.count_nonzero(a[1] != b[1]))
            color_distance = cv2.compareHist(a[2], b[2], cv2.HISTCMP_BHATTACHARYYA)
            if upper_overlap >= .75 and hash_distance <= 16 and color_distance <= .45:
                raise ValueError(
                    f"第{first+1}路与第{second+1}路疑似重复：主体结构相近，"
                    f"即使光照或底部不同也可能是同一岩体变体"
                    f"（结构重合{upper_overlap:.3f}，结构差异{hash_distance}/63，"
                    f"色调距离{color_distance:.3f}）。")
    return outputs, records
