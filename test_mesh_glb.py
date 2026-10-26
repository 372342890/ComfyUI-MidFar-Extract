import io
import json
import struct
import unittest
import numpy as np
from PIL import Image


def fixture():
    vertices = np.array([[-1, -1, -1], [1, -1, -1],
                         [0, 1, -1], [0, 0, 1]], dtype=np.float32)
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    rgba = np.zeros((1, 12, 20, 4), dtype=np.float32)
    rgba[0, 2:10, 2:18] = [0.2, 0.4, 0.8, 1]
    return vertices, faces, rgba


def parse_glb(blob):
    magic, version, length = struct.unpack_from("<4sII", blob)
    assert (magic, version, length) == (b"glTF", 2, len(blob))
    size, typ = struct.unpack_from("<I4s", blob, 12)
    assert typ == b"JSON"
    doc = json.loads(blob[20:20+size])
    bsize, btype = struct.unpack_from("<I4s", blob, 20+size)
    assert btype == b"BIN\x00"
    binary = blob[28+size:28+size+bsize]
    assert len(binary) == bsize
    return doc, binary


class GLBTests(unittest.TestCase):
    def test_preparation_retains_aspect_and_every_foreground_pixel(self):
        from mesh_glb import prepare_rgb
        rgba = fixture()[2]
        rgb = prepare_rgb(rgba)
        self.assertEqual(rgb.shape, (1, 20, 20, 3))
        np.testing.assert_allclose(rgb[0, 6:14, 2:18], rgba[0, 2:10, 2:18, :3])
        np.testing.assert_allclose(rgb[0, :4], 1)

    def test_preparation_rejects_empty_batch_and_empty_alpha(self):
        from mesh_glb import prepare_rgb
        for a in [np.zeros((1, 10, 10, 4)), np.zeros((2, 10, 10, 4)),
                  np.zeros((0, 10, 10, 4))]:
            with self.assertRaises(ValueError):
                prepare_rgb(a)

    def test_glb_preserves_real_geometry_and_embeds_matte_texture(self):
        from mesh_glb import encode_glb
        vertices, faces, rgba = fixture()
        blob, report = encode_glb(vertices, faces, rgba)
        doc, binary = parse_glb(blob)
        primitive = doc["meshes"][0]["primitives"][0]
        for name, expected, dtype in [
                ("POSITION", vertices, "<f4"), ("indices", faces.flatten(), "<u4")]:
            aid = primitive["indices"] if name == "indices" else primitive["attributes"][name]
            accessor = doc["accessors"][aid]
            view = doc["bufferViews"][accessor["bufferView"]]
            actual = np.frombuffer(binary[view["byteOffset"]:view["byteOffset"]+
                                          view["byteLength"]], dtype=dtype)
            np.testing.assert_array_equal(actual, expected.flatten())
        pbr = doc["materials"][0]["pbrMetallicRoughness"]
        self.assertEqual((pbr["metallicFactor"], pbr["roughnessFactor"]), (0, 1))
        self.assertTrue(all("uri" not in x for x in doc["images"] + doc["buffers"]))
        view = doc["bufferViews"][doc["images"][0]["bufferView"]]
        image = Image.open(io.BytesIO(binary[view["byteOffset"]:view["byteOffset"]+
                                             view["byteLength"]]))
        self.assertEqual(image.format, "PNG")
        self.assertEqual(report["vertices"], 4)
        self.assertIn("single-front", report["texture_method"])

    def test_projection_puts_upper_vertices_at_upper_source_pixels(self):
        from mesh_glb import encode_glb
        blob, _ = encode_glb(*fixture())
        doc, binary = parse_glb(blob)
        primitive = doc["meshes"][0]["primitives"][0]
        view = doc["bufferViews"][doc["accessors"][
            primitive["attributes"]["TEXCOORD_0"]]["bufferView"]]
        uv = np.frombuffer(binary[view["byteOffset"]:view["byteOffset"]+
                                   view["byteLength"]], "<f4").reshape(-1, 2)
        self.assertLess(uv[2, 1], uv[0, 1])
        self.assertTrue(np.isfinite(uv).all())

    def test_texture_extends_foreground_instead_of_empty_pixel_rgb(self):
        from mesh_glb import extended_texture
        rgba = fixture()[2]
        tex = extended_texture(rgba)
        np.testing.assert_array_equal(tex[0, 0], [51, 102, 204])

    def test_export_rejects_invalid_and_flat_geometry(self):
        from mesh_glb import encode_glb
        vertices, faces, rgba = fixture()
        bad = vertices.copy()
        bad[0, 0] = np.nan
        flat = vertices.copy()
        flat[:, 2] = 0
        for v, f in [(bad, faces), (flat, faces), (vertices, np.array([[0, 1, 99]])),
                     (vertices, np.empty((0, 3))), (np.empty((0, 3)), faces)]:
            with self.assertRaises(ValueError):
                encode_glb(v, f, rgba)


if __name__ == "__main__":
    unittest.main()
