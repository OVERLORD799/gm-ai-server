#!/usr/bin/env python3
"""HTTP smoke test for the deployed canonical VLM contract."""
from __future__ import annotations

import argparse
import base64
import io
import json
from urllib.request import Request, urlopen

from PIL import Image


def _post(url: str, payload: dict) -> dict:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = Request(
        url,
        data=raw,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=180.0) as response:
        value = json.loads(response.read().decode("utf-8"))
    if type(value) is not dict:
        raise RuntimeError("VLM response is not an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    args = parser.parse_args()
    buffer = io.BytesIO()
    Image.new("RGB", (96, 64), color=(35, 80, 140)).save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    prompt = (
        "Return ONLY this JSON object, with no Markdown or additional keys: "
        '{"scene_summary":"synthetic blue test image","keywords":["blue image"],'
        '"risk_type":"none","risk_confidence":0.9,"affected_entities":[],'
        '"predicted_consequence":"none","prediction_horizon_s":2.0,'
        '"time_to_risk_s":null,"explanation":"no hazard in synthetic image",'
        '"suggested_action":"continue","spatial_hint":"none"}'
    )
    value = _post(
        args.base_url.rstrip("/") + "/analyze",
        {
            "image_b64": image_b64,
            "prompt": prompt,
            "request_id": "smoke-request",
            "frame_id": "smoke-frame",
            "model_id": "Qwen2.5-VL-7B-Instruct-awq",
            "prompt_version": "five_stage_safety_v1",
            "schema_version": "five_stage_vlm_v1",
            "max_output_tokens": 256,
        },
    )
    required = {
        "ok",
        "request_id",
        "frame_id",
        "scene_summary",
        "keywords",
        "risk_type",
        "risk_confidence",
        "affected_entities",
        "predicted_consequence",
        "prediction_horizon_s",
        "time_to_risk_s",
        "explanation",
        "suggested_action",
        "spatial_hint",
        "prompt_version",
        "schema_version",
        "model_id",
        "latency_ms",
    }
    if value.get("ok") is not True or not required.issubset(value):
        raise RuntimeError("VLM response does not satisfy canonical V0-A")
    if value.get("model_id") != "Qwen2.5-VL-7B-Instruct-awq":
        raise RuntimeError("VLM model identity drift")
    print(json.dumps(value, sort_keys=True, ensure_ascii=True))
    print("smoke_ok")


if __name__ == "__main__":
    main()
