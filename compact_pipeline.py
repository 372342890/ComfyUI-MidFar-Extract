"""Conditional extraction: stop on acceptance; at most two quality repairs."""
import json
try:
    from .autonomous_audit import inspect_asset, choose_asset, repair_prompt
except ImportError:
    from autonomous_audit import inspect_asset, choose_asset, repair_prompt


def extract_until_accepted(generate, review, reference, plan, key, guide=None,
                           identity_record="", observer=None):
    candidates, numerics, visuals = [], [], []
    previous, feedback = None, ""
    for version in range(1, 4):
        # Service errors deliberately propagate without automatic paid retries.
        raw = generate(version, previous, feedback)
        sheet, prompt, rgba, numeric = inspect_asset(
            raw, reference, plan, key, .06, 24, guide, identity_record)
        visual = review(sheet, prompt, version)
        candidates.append(rgba)
        numerics.append(numeric)
        visuals.append(visual)
        if observer is not None:
            observer(version, raw, sheet, rgba, numeric, prompt, numerics, visuals)
        try:
            selected, record = choose_asset(candidates, numerics, visuals, 85)
        except ValueError:
            if version == 3:
                raise
            feedback = repair_prompt(plan, key, numeric, visual, version, identity_record)
            previous = raw
            continue
        record = json.loads(record)
        record["attempts"] = version
        record["stopped_on_acceptance"] = True
        return selected, json.dumps(record, ensure_ascii=True, allow_nan=False)
