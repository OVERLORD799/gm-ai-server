#!/usr/bin/env python3
"""HTTP smoke test for GDINO ground and SAM2 track contracts."""
from __future__ import annotations

import argparse
import base64
import io
import json
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw


def _post(url: str, payload: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=180.0) as response:
        value = json.loads(response.read().decode("utf-8"))
    if type(value) is not dict:
        raise RuntimeError("perception response is not an object")
    return value


def _image() -> str:
    image = Image.new("RGB", (320, 240), color=(210, 210, 210))
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 60, 230, 190), fill=(220, 30, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18082")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    image_b64 = _image()

    ground = _post(
        base + "/ground",
        {
            "text_prompt": "red rectangle . object",
            "keywords": ["red rectangle", "object"],
            "image_b64": image_b64,
            "box_threshold": 0.2,
            "text_threshold": 0.25,
            "run_sam2": True,
            "request_id": "smoke-request",
            "frame_id": "smoke-frame",
        },
    )
    if ground.get("ok") is not True or not {
        "gdino_model_id",
        "sam2_model_id",
        "sam2_checkpoint",
    }.issubset(ground.get("model_versions") or {}):
        raise RuntimeError("ground response does not satisfy canonical V0-A")

    init = _post(
        base + "/track",
        {
            "action": "init",
            "frame_index": 0,
            "image_b64": image_b64,
            "init": {
                "target_label": "red rectangle",
                "box_xyxy": [80.0, 60.0, 230.0, 190.0],
                "re_detect_every_n": 100,
            },
            "return_mask_rle": True,
        },
    )
    session_id = init.get("session_id")
    if init.get("ok") is not True or type(session_id) is not str or not session_id:
        raise RuntimeError("SAM2 init failed")
    tracks = init.get("tracks") or []
    if not tracks or type(tracks[0].get("mask_rle")) is not dict:
        raise RuntimeError("SAM2 init did not return the requested mask RLE")
    step = _post(
        base + "/track",
        {
            "action": "step",
            "frame_index": 1,
            "image_b64": image_b64,
            "session_id": session_id,
        },
    )
    if step.get("ok") is not True or step.get("session_id") != session_id:
        raise RuntimeError("SAM2 step failed")
    print(json.dumps({"ground": ground, "track_init": init, "track_step": step}, sort_keys=True))
    print("smoke_ok")


if __name__ == "__main__":
    main()
