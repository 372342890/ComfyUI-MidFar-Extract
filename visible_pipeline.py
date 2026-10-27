"""Visible PE handoff only: validate identities and prepare current-image crops."""
import json
import re
import unicodedata
from uuid import uuid4

try:
    from .autonomous_plan import candidate_review, split_three, asset_reference, _decode
    from .localization_view import localization_view
    from .core import choose_key, validate_image
    from .objects import object_filename
except ImportError:
    from autonomous_plan import candidate_review, split_three, asset_reference, _decode
    from localization_view import localization_view
    from core import choose_key, validate_image
    from objects import object_filename


def split_prompts(recognition, pe, reference, prefix, background_mode="key"):
    source = validate_image(reference)
    if source.shape[0] != 1:
        raise ValueError("Only one current reference image is accepted.")
    plan = _decode(recognition, "objects")
    targets = plan["objects"]
    if not isinstance(targets, list) or len(targets) != 3:
        raise ValueError("识别节点需要三个实际完整目标，不能虚构补足。")
    ids, units, identities = [], set(), set()
    for target in targets:
        if not isinstance(target, dict):
            raise ValueError("Each recognized target must be an object.")
        identifier, unit = target.get("id"), target.get("unit_id")
        if (not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", identifier)
                or identifier in ids or not isinstance(unit, str) or
                not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", unit.strip())):
            raise ValueError("Three distinct actual target ids and connected units are required.")
        normalized_unit = unit.strip().casefold()
        if normalized_unit in units:
            raise ValueError("识别中存在重复的相连单元，不能拆分凑三个目标。")
        if (target.get("kind") not in ("rock", "hill", "mountain") or
                target.get("layer") not in ("midground", "background") or
                target.get("is_single") is not True or target.get("suitable") is not True or
                target.get("boundary_cut") is not False):
            raise ValueError("识别目标不是可提取的完整中远景单元。")
        for field in ("name", "location", "appearance"):
            if not isinstance(target.get(field), str) or not target[field].strip():
                raise ValueError("识别描述缺少字段：" + field)
        identity = tuple(
            re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", target[field]).casefold())
            for field in ("layer", "location", "appearance"))
        if identity in identities:
            raise ValueError("识别中有位置与外观重复的目标，换编号不能作为新素材。")
        identities.add(identity)
        occlusion = target.get("occlusion")
        if not (isinstance(occlusion, str) and occlusion.strip() or
                isinstance(occlusion, list) and
                all(isinstance(item, str) and item.strip() for item in occlusion)):
            raise ValueError("识别遮挡描述必须是文本或文本列表。")
        ids.append(identifier)
        units.add(normalized_unit)
    entries = _decode(pe, "objects")["objects"]
    if not isinstance(entries, list) or len(entries) != 3:
        raise ValueError("PE节点必须输出三份独立提示词。")
    lookup = {}
    for entry in entries:
        if (not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or
                entry["id"] in lookup or not isinstance(entry.get("prompt"), str) or
                not 0 < len(entry["prompt"].strip()) < 15000):
            raise ValueError("PE目标id或提示词无效，不能静默补写。")
        lookup[entry["id"]] = entry["prompt"].strip()
    if set(lookup) != set(ids):
        raise ValueError("PE目标必须与识别结果的三个id一致。")
    if background_mode not in ("key", "white"):
        raise ValueError("Unknown output background mode.")
    key = "#FFFFFF" if background_mode == "white" else choose_key(source, "auto")
    prompts = []
    for target in targets:
        excluded = "；".join(
            f"「{other['name']}」（id={other['id']}，位置：{other['location']}）"
            for other in targets if other["id"] != target["id"])
        identity = (
            f"本路唯一目标：{target['name']}（id={target['id']}，距离层={target['layer']}）。"
            f"原图位置：{target['location']}。外观：{target['appearance']}。\n"
            f"本路排除的其他目标：{excluded}。不得用其他目标的同形变体替代本路目标。\n"
            "当前输入是完整原图，先按以上身份定位，再执行以下PE。\n")
        prompts.append(identity + lookup[target["id"]].replace("{KEY_COLOR}", key) +
                       f"\n全部空白区域必须使用均匀纯色{key}，不要让键色染上主体。"
                       "完整居中，宽高均不超过画布70%，四侧至少15%空白，"
                       "只输出一张完整单元。" +
                       ("本次输出白底提取图，背景为纯白；保留主体自身白色岩面与积雪。"
                        "移除主体外围雾气和环境，不要求透明通道。"
                        if background_mode == "white" else ""))
    run_id = uuid4().hex
    prefixes = [object_filename(target, prefix + "/run_" + run_id) for target in targets]
    record = {"schema_version": 1, "status": "ready", "source_image_modified": False,
              "source_size": [source.shape[2], source.shape[1]], "key_color": key,
              "background_mode": background_mode,
              "run_id": run_id,
              "recognized_targets": targets, "original_pe_prompts": [lookup[i] for i in ids],
              "execution_prompts": prompts, "prefixes": prefixes, "hidden_model_calls": 0}
    return (*prompts, key, *prefixes, json.dumps(record, ensure_ascii=False))


def prepare_asset(recognition, pe, reference, index, prefix):
    if type(index) is not int or index not in (1, 2, 3):
        raise ValueError("asset_index must be 1, 2 or 3.")
    _, manifest_text = candidate_review(recognition, reference)
    manifest = json.loads(manifest_text)
    if manifest["safe_unit_count"] != 3 or len(manifest["original_objects"]) != 3:
        raise ValueError("识别节点必须提供三个实际可安全提取的完整目标。")
    value = _decode(pe, "objects")
    entries = value["objects"]
    if not isinstance(entries, list) or len(entries) != 3:
        raise ValueError("PE节点必须提供三份独立提示词。")
    prompts = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each PE item must contain an actual id and prompt.")
        identifier, prompt = entry.get("id"), entry.get("prompt")
        if not isinstance(identifier, str) or identifier in prompts:
            raise ValueError("PE ids must be unique actual recognized ids.")
        if not isinstance(prompt, str) or not 0 < len(prompt.strip()) < 15000:
            raise ValueError("PE提示词为空或过长，不能静默补写。")
        prompts[identifier] = prompt.strip()
    ids = [obj["id"] for obj in manifest["original_objects"]]
    if set(ids) != set(prompts):
        raise ValueError("PE改变了识别目标id，不能换目标或补虚构目标。")
    plans = split_three(json.dumps({"objects": [{"id": identifier} for identifier in ids]}),
                        manifest_text)
    plan = plans[index-1]
    target = json.loads(plan)["objects"][0]
    _, crop, key, filename, source = asset_reference(plan, reference, prefix)
    marked, _, identity = localization_view(crop, source)
    pe_prompt = prompts[target["id"]]
    prompt = pe_prompt.replace("{KEY_COLOR}", key) + (
        f"\n\n当前路目标：{target['name']}（id={target['id']}）。"
        "输入是原图中这一目标的局部裁图，红色圆圈十字只用于目标身份定位，"
        "不是山岩纹理，成图必须去掉此标注。保留包含标注位置的目标及其真实相连结构。"
        f"所有非目标背景必须使用均匀纯色{key}；主体不能染上键色。"
        "完整居中，宽高均不超过画布70%，四侧至少15%空白。"
        "只输出一张完整独立资产图。\n" + identity)
    record = json.loads(source)
    record.update(asset_id=target["id"], asset_index=index, pe_prompt=pe_prompt,
                  execution_prompt=prompt, source_plan=json.loads(plan),
                  preview="current crop with target identity marker",
                  hidden_model_calls=0)
    return marked, prompt, key, filename, json.dumps(record, ensure_ascii=False)
