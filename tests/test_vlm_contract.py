#!/usr/bin/env python3
from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi import HTTPException
from PIL import Image

os.environ["GM_AI_TEST_MODE"] = "1"

APP_PATH = Path(__file__).resolve().parents[1] / "vlm-service" / "app.py"
SPEC = importlib.util.spec_from_file_location("gm_vlm_app", APP_PATH)
assert SPEC is not None and SPEC.loader is not None
APP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = APP
SPEC.loader.exec_module(APP)


def _png() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 12), color=(20, 30, 40)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class VLMContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.original_generate = APP._generate
        self.semantic = {
            "scene_summary": "robot and workcell",
            "keywords": ["robot arm", "container"],
            "risk_type": "none",
            "risk_confidence": 0.92,
            "affected_entities": ["ur10e"],
            "predicted_consequence": "normal operation",
            "prediction_horizon_s": 2.0,
            "time_to_risk_s": None,
            "explanation": "no semantic hazard",
            "suggested_action": "continue",
            "spatial_hint": "none",
        }
        APP._generate = lambda image, prompt, tokens: (
            json.dumps(self.semantic, separators=(",", ":")),
            12.5,
        )

    def tearDown(self) -> None:
        APP._generate = self.original_generate

    def _request(self, **overrides):
        values = {
            "image_b64": _png(),
            "request_id": "request-1",
            "frame_id": "frame-1",
            "model_id": APP.API_MODEL_ID,
            "prompt_version": APP.PROMPT_VERSION,
            "schema_version": APP.SCHEMA_VERSION,
        }
        values.update(overrides)
        return APP.AnalyzeRequest(**values)

    def test_health_pins_actual_awq_identity(self) -> None:
        health = APP.health()
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["model_id"], "Qwen2.5-VL-7B-Instruct-awq")
        self.assertEqual(health["backend_model_id"], "Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
        self.assertEqual(health["quantization"], "awq")
        self.assertEqual(health["awq_backend"], "torch_fallback")

    def test_awq_loader_preserves_unquantized_output_head(self) -> None:
        config = SimpleNamespace(
            quantization_config={"modules_to_not_convert": ["visual"]}
        )
        self.assertIs(APP._prepare_awq_config(config), config)
        self.assertEqual(
            config.quantization_config["modules_to_not_convert"],
            ["visual", "lm_head"],
        )

    def test_analyze_returns_canonical_v0a_and_legacy_aliases(self) -> None:
        response = APP.analyze(self._request()).model_dump()
        for key, value in self.semantic.items():
            self.assertEqual(response[key], value)
        self.assertTrue(response["ok"])
        self.assertEqual(response["request_id"], "request-1")
        self.assertEqual(response["frame_id"], "frame-1")
        self.assertEqual(response["model_id"], APP.API_MODEL_ID)
        self.assertEqual(response["vlm_keywords"], self.semantic["keywords"])
        self.assertEqual(response["remote_contract"], "canonical_v0a")

    def test_wrong_model_identity_fails_closed(self) -> None:
        with self.assertRaises(HTTPException) as caught:
            APP.analyze(self._request(model_id="wrong"))
        self.assertEqual(caught.exception.status_code, 409)

    def test_invalid_model_enum_fails_closed(self) -> None:
        self.semantic["risk_type"] = "unknown"
        with self.assertRaises(HTTPException) as caught:
            APP.analyze(self._request())
        self.assertEqual(caught.exception.status_code, 502)

    def test_unexpected_model_output_field_fails_closed(self) -> None:
        self.semantic["unbound_field"] = "must not pass through"
        with self.assertRaises(HTTPException) as caught:
            APP.analyze(self._request())
        self.assertEqual(caught.exception.status_code, 502)

    def test_json_extractor_handles_braces_inside_strings(self) -> None:
        value = APP._extract_json('prefix {"scene_summary":"brace } stays","keywords":[]} tail')
        self.assertEqual(value["scene_summary"], "brace } stays")

    def test_image_pixel_limit_fails_closed(self) -> None:
        original_limit = APP.MAX_IMAGE_PIXELS
        APP.MAX_IMAGE_PIXELS = 16
        try:
            with self.assertRaises(HTTPException) as caught:
                APP._decode_image(self._request())
            self.assertEqual(caught.exception.status_code, 413)
        finally:
            APP.MAX_IMAGE_PIXELS = original_limit

    def test_enabled_missing_image_path_is_a_client_error(self) -> None:
        original = APP.ALLOW_IMAGE_PATH
        APP.ALLOW_IMAGE_PATH = True
        try:
            with self.assertRaises(HTTPException) as caught:
                APP._decode_image(self._request(image_b64=None, image_path="/missing/image.png"))
            self.assertEqual(caught.exception.status_code, 400)
        finally:
            APP.ALLOW_IMAGE_PATH = original


if __name__ == "__main__":
    unittest.main()
