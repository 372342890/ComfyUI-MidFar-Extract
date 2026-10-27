"""One upload, two explicit native text stages, and three visible image branches."""
import copy
import json
from pathlib import Path


RECOGNITION = """你是游戏中远景素材识别助手。只观察当前这一张场景原图，识别并选出恰好三个不同的、能作为完整单体复用的中景或远景山石、山丘或山体。
先判断实际距离、真实岩体连接和遮挡。相接的双峰、多峰、岩肩与共享天然岩基算同一个目标，不能强行拆分凑三个；前后不同深度的重叠山体不算连接。只选择原图边界内的完整单元，原图左右裁断的大山脉、天空、水面、树林、前景草堆、整层风景不作为目标。保留目标自身覆草覆雪。被环境遮挡的短山脚可在图片提取时补齐。
描述每份实际目标的原图唯一位置、峰数、相连岩肩、轮廓、材质颜色、风格、遮挡物以及需要排除的其他对象。三个目标优先覆盖可用的中景和远景，按形态差异与完整性选择，不固定取某个位置。
输出一个完整JSON对象，字段为style和objects。objects恰好三项，按素材1、2、3顺序。每项字段为id、unit_id、name、kind、layer、location、appearance、occlusion、selection_reason、asset_unit、connected_parts、excluded_objects、is_single、boundary_cut、suitable。
id与unit_id使用不同的安全英文编号，kind只能rock/hill/mountain，layer只能midground/background，asset_unit只能independent/connected_formation。用自然语言描述目标在完整原图中的唯一位置和外观，不输出bbox或坐标；后续直接用完整原图提取。
connected_parts与excluded_objects是字符串数组；priority是0到100数值；is_single、boundary_cut、suitable为真正JSON布尔。选定完整单元is_single和suitable为true、boundary_cut为false。所有描述必须来自本张图。若不足三个，输出实际可用数量并写明不足，不虚构。只输出合法JSON，不写代码围栏、省略号或注释。"""

PE = """你是中远景素材提取的Prompt Engineer（PE）。图像输入是当前完整原图，文本输入是上一节点的实际识别结果。
逐项对照这张原图，为识别节点的三个目标分别写一份完整、具体、可直接交给图像编辑模型的提取提示词。所有描述取自当前目标及原图，三个提示词分别描述各自形体，不能复用泛化内容替代目标细节。
每份提示词明确：只提取该id对应的一个完整单元；保留原画风、原色调、可见轮廓、峰数、相连岩肩及天然岩基、覆草覆雪和空气透视；移除天空、树林、水面、不同深度邻山、地面托盘及全部无关物体。遵守识别中的connected_parts与excluded_objects。仅用目标自身结构补全被环境挡住的短自然山脚和侧面，不能复制挡物造成的凹口，不新添峰柱，不重新设计。
这份提示词会与当前完整原图一起输入图片提取节点。准确写出这个目标在原图中的位置、峰数、外观与相连部分，避免提取同类邻山。输出一个完整居中的独立素材，宽高均不超过画布70%，四侧各留至少15%空白；全部峰尖、侧边和自然山脚完整。
提示词中所有非目标区域写为“均匀纯色{KEY_COLOR}背景”，此占位符由提示词分路节点根据原图颜色自动替换，不让用户手工填写。不画棋盘格、渐变、阴影、文字、水印，也不将色键染上主体。
只输出合法JSON：{"objects":[{"id":"识别中的第一个实际id","prompt":"第一份完整中文提取提示词"},{"id":"识别中的第二个实际id","prompt":"第二份完整中文提取提示词"},{"id":"识别中的第三个实际id","prompt":"第三份完整中文提取提示词"}]}。
恰好三份，id与识别结果逐项一致。不得替换目标id、缩框或虚构目标，不输出Markdown、省略号和注释。"""

IMAGE_INSTRUCTION = """严格遵循连接的本路PE提示词，从输入的完整原图中提取指定的一个真实目标，输出一张完整独立的中远景提取图。
保留目标真实外观与原画风，补齐自身短自然山脚；只移除无关环境，不重设计。使用指定纯色键背景，四侧留足空白，不加文字或其他物体。"""


def _slot(name, kind, label=None):
    return {"name": name, "label": label or name, "type": kind, "format": "slot"}


def _node(identifier, kind, title, x, y, inputs, outputs, previews, width=390):
    return {"id": str(identifier), "type": "baNode", "position": {"x": x, "y": y},
            "width": width, "initialWidth": width, "initialHeight": 500, "selected": False,
            "data": {"type": kind, "title": title, "inputs": inputs,
                     "outputSlots": [{"name": name, "label": name, "type": typ}
                                     for name, typ in outputs],
                     "extra": {"output_preview_names": previews}}}


def build_workflow(input_filename="reference.png", filename_prefix="midfar-visible/asset",
                   background_mode="key"):
    if background_mode not in ("key", "white"):
        raise ValueError("Unknown output background mode.")
    pe_instruction = PE
    image_instruction = IMAGE_INSTRUCTION
    if background_mode == "white":
        pe_instruction = PE.replace("均匀纯色{KEY_COLOR}背景", "纯白底{KEY_COLOR}背景")
        pe_instruction += "\n本次允许白底，默认输出白底提取图，不要求透明通道。主体外围不能保留雾气。"
        image_instruction = IMAGE_INSTRUCTION.replace(
            "指定纯色键背景", "纯白底背景")
        image_instruction += "\n不要求透明通道，保留主体白色岩面与积雪，移除外围雾气。"
    templates = json.loads(Path(__file__).with_name("visible_native_templates.json").read_text(
        encoding="utf8"))
    nodes = [_node(1, "LoadImage", "① 上传一张原图", 0, 650,
                   [{"name": "image", "label": "image", "type": "COMBO", "value": input_filename}],
                   [("IMAGE", "IMAGE"), ("MASK", "MASK")], ["IMAGE"], 350)]

    def carrier(template, identifier, title, x, y, instruction):
        node = copy.deepcopy(templates[template])
        node.update(id=str(identifier), position={"x": x, "y": y}, width=410,
                    initialWidth=410, initialHeight=600, selected=False)
        node["data"]["title"] = title
        for slot in node["data"]["inputs"]:
            if slot["name"] == "user_prompt":
                slot["value"] = instruction
        node["data"]["extra"]["user_prompt"] = [
            {"type": "paragraph", "children": [{"text": line}]}
            for line in instruction.splitlines()]
        nodes.append(node)

    carrier("recognition", 2, "② 识别 · 查看三个目标与原图分析", 440, 650, RECOGNITION)
    carrier("pe", 3, "③ PE · 查看三份提取提示词", 940, 650, pe_instruction)
    edges = []

    def connect(source, output, target, input_name):
        a = next(n for n in nodes if n["id"] == str(source))
        b = next(n for n in nodes if n["id"] == str(target))
        out = next(s for s in a["data"]["outputSlots"] if s["name"] == output)
        inp = next(s for s in b["data"]["inputs"] if s["name"] == input_name)
        assert out["type"] == inp["type"], (source, output, target, input_name)
        inp.pop("value", None)
        inp["format"] = "slot"
        edges.append({"id": "visible-"+str(len(edges)), "type": "baEdge",
                      "source": str(source), "sourceHandle": output,
                      "target": str(target), "targetHandle": input_name,
                      "data": {"type": out["type"], "sourceType": out["type"],
                               "targetType": inp["type"], "extra": {"type": out["type"]}}})

    connect(1, "IMAGE", 2, "image")
    connect(1, "IMAGE", 3, "image")
    connect(2, "text", 3, "text")
    nodes.append(_node(
        4, "BAMidFarVisiblePE", "④ 三份PE自动分路 · 查看实际提示词", 1460, 650,
        [_slot("reference", "IMAGE"), _slot("recognition", "STRING"), _slot("pe", "STRING"),
         {"name": "filename_prefix", "label": "输出目录", "type": "STRING", "value": filename_prefix}],
        [("prompt_1", "STRING"), ("prompt_2", "STRING"), ("prompt_3", "STRING"),
         ("key_color", "STRING"), ("prefix_1", "STRING"), ("prefix_2", "STRING"),
         ("prefix_3", "STRING"), ("record", "STRING")],
        ["prompt_1", "prompt_2", "prompt_3", "record"]))
    if background_mode == "white":
        nodes[-1]["data"]["inputs"].append(
            {"name": "background_mode", "label": "输出背景", "type": "STRING", "value": "white"})
    connect(1, "IMAGE", 4, "reference")
    connect(2, "text", 4, "recognition")
    connect(3, "text", 4, "pe")
    for index in range(1, 4):
        generate = index+4
        y = (index-1)*760
        carrier("image", generate, f"⑤ 图片提取{index} · 查看原始成图", 1960, y, image_instruction)
        connect(1, "IMAGE", generate, "image")
        connect(4, f"prompt_{index}", generate, "prompt")
    nodes.append(_node(
        8, "BAMidFarVisibleThreePNG", "⑥ 三图统一去重 · 输出三个透明PNG", 2500, 650,
        [_slot(f"image_{i}", "IMAGE") for i in range(1, 4)] +
        [_slot("key_color", "STRING")] +
        [_slot(f"prefix_{i}", "STRING") for i in range(1, 4)],
        [("rgba_1", "IMAGE"), ("rgba_2", "IMAGE"), ("rgba_3", "IMAGE"),
         ("quality_record", "STRING")], ["rgba_1", "rgba_2", "rgba_3", "quality_record"], 490))
    for index in range(1, 4):
        connect(index+4, "image", 8, f"image_{index}")
        connect(4, f"prefix_{index}", 8, f"prefix_{index}")
    connect(4, "key_color", 8, "key_color")
    if background_mode == "white":
        nodes[-1]["data"]["title"] = "⑥ 三图统一去重 · 输出三张白底提取PNG"
    return {"name": "ba-clipboard", "version": 1, "nodes": nodes, "edges": edges}
