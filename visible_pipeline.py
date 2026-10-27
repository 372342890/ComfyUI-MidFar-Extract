"""Visible PE handoff only: validate identities and prepare current-image crops."""
import json

try:
    from .autonomous_plan import candidate_review, split_three, asset_reference, _decode
    from .localization_view import localization_view
except ImportError:
    from autonomous_plan import candidate_review, split_three, asset_reference, _decode
    from localization_view import localization_view


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
