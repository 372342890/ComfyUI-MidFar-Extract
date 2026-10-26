"""Reference-guided individual natural terrain asset generation."""
import copy
import json
import re

KINDS = ("rock", "hill", "grass_tuft", "mountain")
KEYS = ("#00FF00", "#FF00FF", "#0000FF")


def parse_assets(text, max_assets=8):
    if type(max_assets) is not int or not 1 <= max_assets <= 32:
        raise ValueError("max_assets must be an integer between 1 and 32.")
    if not isinstance(text, str) or len(text) > 100000:
        raise ValueError("Asset plan must be a JSON string under 100000 characters.")
    plan = None
    for match in re.finditer(r"\{", text):
        try:
            candidate, _ = json.JSONDecoder().raw_decode(text[match.start():])
            if isinstance(candidate, dict) and "assets" in candidate:
                plan = candidate
                break
        except ValueError:
            continue
    if plan is None or not isinstance(plan["assets"], list) or not 1 <= len(plan["assets"]) <= 100:
        raise ValueError("Expected a non-empty assets array.")
    style = plan.get("style")
    if not isinstance(style, str) or not 0 < len(style.strip()) < 6000:
        raise ValueError("style must describe the current reference.")
    assets, ids = [], set()
    for item in plan["assets"]:
        if not isinstance(item, dict) or item.get("is_single") is not True:
            raise ValueError("Each target must explicitly be one single natural object.")
        for field in ("id", "name", "shape", "surface", "variation"):
            if not isinstance(item.get(field), str) or not 0 < len(item[field].strip()) < 3000:
                raise ValueError(f"Asset {field} must be concise non-empty text.")
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", item["id"]) or item["id"] in ids:
            raise ValueError("Asset ids must be safe unique ASCII identifiers.")
        ids.add(item["id"])
        if item.get("kind") not in KINDS:
            raise ValueError("kind must be rock, hill, grass_tuft or mountain.")
        if re.search(r"连续山脉|群山|山脉|整片|连成一片|岩石群|山石群|地面托盘|"
                     r"\b(mountain range|landscape|ground plane|rock cluster|skyline)\b",
                     item["name"], re.IGNORECASE):
            raise ValueError("A terrain asset cannot be an entire landscape, group or ground tray.")
        assets.append({**item, **{f: item[f].strip() for f in
                                 ("id", "name", "shape", "surface", "variation")}})
    return {"style": style.strip(), "assets": assets[:max_assets],
            "total": len(assets), "truncated": max(0, len(assets) - max_assets)}


def asset_prompt(plan, target, key):
    if key not in KEYS:
        raise ValueError("Unsupported chroma key.")
    own_volume = {
        "hill": "草丘是有厚度的低矮圆丘：保留它自身的草面、土壤或岩基厚度，"
                "起伏圆润、矮而宽；底边自然收拢，不能把草面全部去掉，也不做薄纸片或悬空平板。",
        "grass_tuft": "草堆是一簇相连的完整草本实体：所有主要草叶从同一个紧凑根部生长，"
                      "保留完整根部、叶尖和自然疏密，允许内部镂空，不附带大片草坪或泥土托盘。",
        "rock": "山石是一个完整相连的岩石体块：保留它自身的岩基，块面概括、层理清楚、"
                "棱角适度圆磨；不要拼成石堆、多个分离岩石或连续崖壁。",
        "mountain": "山体是一个完整独立山丘或山峰：保留自身岩层和覆草坡面，"
                    "轮廓两侧完整收拢，不能变成连续山脉或整层风景。",
    }[target["kind"]]
    return (
        "生成一个可单独复用的游戏自然地形美术素材，每张图片只画一个完整物体。\n"
        "图1是画风参考：仅从图1学习颜色、材质、纹理概括、照明与美术风格，"
        "不要复制其天空、地平线、道路、建筑、树木或整片场景。\n"
        "图2是单体形体参考：优先学习它的体积比例、轮廓和结构；"
        "若图2是四视图或多视图，只综合理解同一个物体，最终不输出四视图、拼图、对比图。"
        "只输出一个三分之四视角的完整单体；若图2是场景图，只借鉴与目标相关的局部形态。\n"
        f"本次目标：{target['name']}。形体要求：{target['shape']}。"
        f"表面要求：{target['surface']}。个体变化：{target['variation']}。\n"
        f"画风严格贴近参考：{plan['style']}。\n{own_volume}\n"
        "不带天空、云、雾、地平线、水面、旁边其他物体、文字、装饰、投影或额外地面托盘。"
        "自身的底部结构属于物体，要完整保留。造型必须自然相连，不能是横向连成一片的素材层。"
        "主体居中，宽高都不超过画布70%，四边至少15%空白安全区，"
        "叶尖、山顶、左右边缘和底部都不能裁断。柔和顶侧光、哑光表面，不用写实照片质感。\n"
        f"所有空白必须是均匀纯色{key}，色键只属于背景，不能染到主体上。"
        "不画棋盘格、渐变、背景阴影、反光或色键光晕。只交付一张单物体彩色键图。")


def asset_filename(target, prefix):
    def safe(value):
        return re.sub(r"[\W_]+", "_", str(value), flags=re.UNICODE).strip("_")[:60] or "asset"
    parts = [safe(p) for p in str(prefix).replace("\\", "/").split("/") if p not in ("", ".", "..")]
    return "/".join(parts + [target["kind"], safe(target["id"]) + "_" + safe(target["name"])])


def expand_terrain_workflow(plan, template, style_filename, shape_filename):
    """Each asset has a distinct image carrier, with two independent image inputs."""
    nodes = template.get("nodes", [])
    roots = [copy.deepcopy(n) for n in nodes if n["data"]["type"] == "LoadImage"]
    if len(roots) != 2:
        raise ValueError("Terrain template must contain style and shape LoadImage nodes, in that order.")
    for root, filename in zip(roots, (style_filename, shape_filename)):
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("Reference filenames must be non-empty.")
        pin = next(i for i in root["data"]["inputs"] if i["name"] == "image")
        pin["value"] = filename
        pin.pop("format", None)
    branch_types = ("BATerrainAssetPrompt", "BALLMImage", "BAMidFarObjectMatte", "SaveImage")
    branches = []
    for kind in branch_types:
        matches = [n for n in nodes if n["data"]["type"] == kind]
        if len(matches) != 1:
            raise ValueError(f"Terrain template must contain exactly one {kind}.")
        branches.append(matches[0])
    # Extra SaveText nodes preserve manifest and matte QA, if present.
    branches += [n for n in nodes if n["data"]["type"] == "SaveText|pysssss"]
    branch_ids, root_ids = {n["id"] for n in branches}, {n["id"] for n in roots}
    edges = [e for e in template.get("edges", []) if e["target"] in branch_ids and
             e["source"] in (branch_ids | root_ids)]
    result = {"name": "ba-clipboard", "version": 1, "nodes": roots, "edges": []}
    for j, target in enumerate(plan["assets"]):
        mapping = {n["id"]: f"terrain_{j}_{k}" for k, n in enumerate(branches)}
        for k, source in enumerate(branches):
            node = copy.deepcopy(source)
            node["id"] = mapping[source["id"]]
            node["position"] = {"x": 440 + min(k, 3)*440, "y": j*920 + (450 if k >= 4 else 0)}
            node["selected"] = False
            node["data"]["title"] = f"{target['id']} · {target['name']} · {source['data']['type']}"
            if source["data"]["type"] == "BATerrainAssetPrompt":
                pin = next(i for i in node["data"]["inputs"] if i["name"] == "plan")
                pin["value"] = json.dumps({"style": plan["style"], "assets": [target]}, ensure_ascii=False)
                pin.pop("format", None)
                node["data"]["extra"] = {"output_preview_names": []}
            if source["data"]["type"] == "SaveText|pysssss":
                for pin in node["data"]["inputs"]:
                    if pin["name"] == "file" and isinstance(pin.get("value"), str):
                        stem, _, suffix = pin["value"].rpartition(".")
                        pin["value"] = (stem or suffix) + "_" + target["id"] + "." + (suffix if stem else "json")
            result["nodes"].append(node)
        for source in edges:
            edge = copy.deepcopy(source)
            edge["id"] = f"terrain_edge_{len(result['edges'])}"
            edge["source"] = mapping.get(edge["source"], edge["source"])
            edge["target"] = mapping[edge["target"]]
            result["edges"].append(edge)
    return result
