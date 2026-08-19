#!/usr/bin/env python3
"""Bounded dependency-free health check used by Compose and operators."""
from __future__ import annotations

import argparse
import json
import sys
from urllib.request import Request, urlopen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--service", choices=("vlm", "perception"), required=True)
    parser.add_argument("--require-loaded", action="store_true")
    args = parser.parse_args()

    request = Request(args.url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=4.0) as response:
        if response.status != 200:
            raise RuntimeError(f"health HTTP status {response.status}")
        raw = response.read(65537)
    if len(raw) > 65536:
        raise RuntimeError("health response exceeds 64 KiB")
    value = json.loads(raw.decode("utf-8"))
    if type(value) is not dict:
        raise RuntimeError("health response is not an object")

    if args.service == "vlm":
        if (
            value.get("status") != "ok"
            or value.get("model_id") != "Qwen2.5-VL-7B-Instruct-awq"
            or value.get("backend_model_id") != "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
            or value.get("quantization") != "awq"
            or value.get("awq_backend") != "torch_fallback"
            or value.get("model_loaded") is not True
        ):
            raise RuntimeError("VLM health identity/status mismatch")
    else:
        if value.get("status") not in ("ok", "warming"):
            raise RuntimeError("perception health status mismatch")
        expected = {
            "gdino_model_id": "IDEA-Research/grounding-dino-base",
            "sam2_model_id": "sam2.1_hiera_small.pt",
            "sam2_checkpoint_sha256": "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38",
        }
        for key, identity in expected.items():
            if value.get(key) != identity:
                raise RuntimeError(f"perception health identity mismatch: {key}")
        if args.require_loaded and value.get("models_loaded") is not True:
            raise RuntimeError("perception models are not loaded")
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # healthcheck must fail closed
        print(f"healthcheck failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
