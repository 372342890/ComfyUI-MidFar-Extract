"""Build the single-reference, three-GLB ByteArtist clipboard graph."""


def _node(identifier, kind, title, x, y, inputs, outputs, width=400, height=460):
    return {
        "id": str(identifier),
        "type": "baNode",
        "position": {"x": x, "y": y},
        "width": width,
        "initialWidth": width,
        "initialHeight": height,
        "selected": False,
        "data": {
            "type": kind,
            "title": title,
            "inputs": inputs,
            "outputSlots": [
                {"name": name, "label": name, "type": kind}
                for name, kind in outputs
            ],
            "extra": {"output_preview_names": []},
        },
    }


def _slot(name, kind):
    return {"name": name, "label": name, "type": kind, "format": "slot"}


def build_workflow(input_filename="reference.png",
                   filename_prefix="midfar-reference/asset"):
    """Return nine real nodes and 28 edges, without loading or writing files."""
    model_outputs = [
        ("model", "MODEL"), ("clip_vision", "CLIP_VISION"), ("vae", "VAE")
    ]
    graph = {
        "name": "ba-clipboard",
        "version": 1,
        "nodes": [
            _node(
                1, "LoadImage", "① 只上传一张参考图", 0, 560,
                [{"name": "image", "label": "image", "type": "COMBO",
                  "value": input_filename}],
                [("IMAGE", "IMAGE"), ("MASK", "MASK")], width=360, height=420,
            ),
            _node(
                2, "BAMidFarReferencePlan", "② 统一识别 · 准备三份目标",
                440, 560,
                [_slot("reference", "IMAGE"),
                 {"name": "filename_prefix", "label": "filename_prefix",
                  "type": "STRING", "value": filename_prefix}],
                [("plan_1", "STRING"), ("plan_2", "STRING"),
                 ("plan_3", "STRING"), ("filename_prefix", "STRING"),
                 ("audit", "STRING")],
            ),
            _node(
                3, "BAMidFar3DModel", "共享混元3D模型 · 加载一次",
                440, 0, [], model_outputs, width=360, height=420,
            ),
        ],
        "edges": [],
    }

    def edge(source, source_handle, target, target_handle, kind):
        graph["edges"].append({
            "id": f"reference-{len(graph['edges'])}",
            "type": "baEdge",
            "source": str(source),
            "target": str(target),
            "sourceHandle": source_handle,
            "targetHandle": target_handle,
            "data": {
                "type": kind,
                "sourceType": kind,
                "targetType": kind,
                "extra": {"type": kind},
            },
        })

    edge(1, "IMAGE", 2, "reference", "IMAGE")
    for branch in range(1, 4):
        extract_id = 2 * branch + 2
        glb_id = extract_id + 1
        y = (branch - 1) * 560
        graph["nodes"].append(_node(
            extract_id, "BAMidFarReferenceExtract",
            f"③ 自动提取与检查 · 素材{branch}", 940, y,
            [_slot("reference", "IMAGE"), _slot("plan", "STRING"),
             _slot("filename_prefix", "STRING")],
            [("rgba", "IMAGE"), ("filename_prefix", "STRING"),
             ("extraction_record", "STRING")],
        ))
        graph["nodes"].append(_node(
            glb_id, "BAMidFarReferenceGLB",
            f"④ 生成并导出 · 模型{branch}.glb", 1440, y,
            [_slot("rgba", "IMAGE")] +
            [_slot(name, kind) for name, kind in model_outputs] +
            [_slot("filename_prefix", "STRING"),
             _slot("extraction_record", "STRING"),
             {"name": "seed", "label": "seed", "type": "INT",
              "value": 20261026}],
            [("glb", "STRING"), ("export_record", "STRING")],
        ))
        edge(1, "IMAGE", extract_id, "reference", "IMAGE")
        edge(2, f"plan_{branch}", extract_id, "plan", "STRING")
        edge(2, "filename_prefix", extract_id, "filename_prefix", "STRING")
        edge(extract_id, "rgba", glb_id, "rgba", "IMAGE")
        for name, kind in model_outputs:
            edge(3, name, glb_id, name, kind)
        edge(extract_id, "filename_prefix", glb_id, "filename_prefix", "STRING")
        edge(extract_id, "extraction_record", glb_id,
             "extraction_record", "STRING")

    return graph
