"""Export genuine model mesh with embedded front-reference color projection."""
import hashlib
import io
import json
import struct
import numpy as np
from PIL import Image


def _rgba(rgba):
    rgba = np.asarray(rgba, dtype=np.float32)
    if rgba.ndim != 4 or rgba.shape[0] != 1 or rgba.shape[-1] != 4:
        raise ValueError("Exactly one RGBA image is required")
    if not np.isfinite(rgba).all() or not np.any(rgba[0, ..., 3] > .5):
        raise ValueError("RGBA must contain finite visible foreground")
    return np.clip(rgba[0], 0, 1)


def prepare_rgb(rgba):
    image = _rgba(rgba)
    rgb = image[..., :3]*image[..., 3:4] + (1-image[..., 3:4])
    h, w = rgb.shape[:2]
    side = max(h, w)
    result = np.ones((1, side, side, 3), dtype=np.float32)
    y, x = (side-h)//2, (side-w)//2
    result[0, y:y+h, x:x+w] = rgb
    return result


def extended_texture(rgba):
    import cv2
    image = _rgba(rgba)
    opaque = image[..., 3] > .5
    _, labels = cv2.distanceTransformWithLabels(
        (~opaque).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    colors = np.zeros((int(labels.max())+1, 3), dtype=np.float32)
    colors[labels[opaque]] = image[..., :3][opaque]
    texture = np.rint(colors[labels]*255).astype(np.uint8)
    # Keep original source colors exactly wherever foreground is visible.
    texture[opaque] = np.rint(image[..., :3][opaque]*255).astype(np.uint8)
    return texture


def encode_glb(vertices, faces, rgba):
    vertices = np.asarray(vertices, dtype="<f4")
    original_faces = np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4 or
            original_faces.ndim != 2 or original_faces.shape[1] != 3 or
            len(original_faces) == 0 or not np.isfinite(vertices).all() or
            not np.isfinite(original_faces).all()):
        raise ValueError("Invalid or empty triangle mesh")
    if (not np.equal(original_faces, np.floor(original_faces)).all() or
            original_faces.min() < 0 or original_faces.max() >= len(vertices)):
        raise ValueError("Mesh face index is invalid")
    extent = np.ptp(vertices, axis=0)
    singular = np.linalg.svd(vertices-vertices.mean(axis=0), compute_uv=False)
    if extent.min() < 1e-5 or singular[-1] < singular[0]*1e-5:
        raise ValueError("Mesh is planar; a real volumetric model is required")
    faces = np.asarray(original_faces, dtype="<u4")
    image = _rgba(rgba)
    yy, xx = np.where(image[..., 3] > .5)
    h, w = image.shape[:2]
    uv = np.column_stack((
        (xx.min() + (vertices[:, 0]-vertices[:, 0].min())/extent[0] *
         (xx.max()-xx.min()) + .5)/w,
        (yy.min() + (vertices[:, 1].max()-vertices[:, 1])/extent[1] *
         (yy.max()-yy.min()) + .5)/h)).astype("<f4")
    triangle = vertices[faces]
    face_normals = np.cross(triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0])
    normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)
    length = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(length, 1e-20)[:, None]
    normals[length < 1e-20] = [0, 1, 0]
    png = io.BytesIO()
    Image.fromarray(extended_texture(rgba)).save(png, format="PNG")
    chunks, views, accessors = [], [], []

    def buffer(data, target=None):
        offset = sum(len(x) for x in chunks)
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target:
            view["target"] = target
        views.append(view)
        chunks.append(data + b"\x00"*((-len(data)) % 4))
        return len(views)-1

    def accessor(array, component, kind, target, bounds=False):
        record = {"bufferView": buffer(array.tobytes(), target), "byteOffset": 0,
                  "componentType": component, "count": len(array), "type": kind}
        if bounds:
            record.update(min=array.min(axis=0).tolist(), max=array.max(axis=0).tolist())
        accessors.append(record)
        return len(accessors)-1

    pos = accessor(vertices, 5126, "VEC3", 34962, True)
    normal = accessor(normals, 5126, "VEC3", 34962)
    texcoord = accessor(uv, 5126, "VEC2", 34962)
    indices = accessor(faces.reshape(-1), 5125, "SCALAR", 34963)
    texture_view = buffer(png.getvalue())
    report = {
        "vertices": len(vertices), "triangles": len(faces),
        "bounds_min": vertices.min(axis=0).tolist(),
        "bounds_max": vertices.max(axis=0).tolist(), "extent": extent.tolist(),
        "signed_volume": float(np.einsum("ij,ij->i", triangle[:, 0],
                                        np.cross(triangle[:, 1], triangle[:, 2])).sum()/6),
        "source_rgba_sha256": hashlib.sha256(np.asarray(rgba, "<f4").tobytes()).hexdigest(),
        "texture_method": "single-front reference projection with nearest foreground color extension",
        "texture_limitation": "Side/back color is extrapolated from the front PNG, not independently AI painted",
        "geometry_method": "Input model mesh preserved; no procedural replacement or flattening",
    }
    binary = b"".join(chunks)
    doc = {
        "asset": {"version": "2.0", "generator": "ComfyUI-MidFar-Extract", "extras": report},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {
            "POSITION": pos, "NORMAL": normal, "TEXCOORD_0": texcoord},
            "indices": indices, "material": 0, "mode": 4}]}],
        "materials": [{"name": "MidFar Matte", "doubleSided": True,
                       "pbrMetallicRoughness": {"baseColorTexture": {"index": 0},
                                               "metallicFactor": 0, "roughnessFactor": 1}}],
        "samplers": [{"magFilter": 9729, "minFilter": 9729, "wrapS": 33071, "wrapT": 33071}],
        "textures": [{"sampler": 0, "source": 0}],
        "images": [{"bufferView": texture_view, "mimeType": "image/png"}],
        "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(binary)}],
    }
    encoded = json.dumps(doc, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    encoded += b" "*((-len(encoded)) % 4)
    blob = (struct.pack("<4sII", b"glTF", 2, 28+len(encoded)+len(binary)) +
            struct.pack("<I4s", len(encoded), b"JSON") + encoded +
            struct.pack("<I4s", len(binary), b"BIN\x00") + binary)
    return blob, report
