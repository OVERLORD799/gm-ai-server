#!/usr/bin/env python3
from __future__ import annotations

import base64
import importlib.util
import io
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from fastapi import HTTPException
from PIL import Image

os.environ["GM_AI_TEST_MODE"] = "1"

APP_PATH = Path(__file__).resolve().parents[1] / "perception-service" / "app.py"
SPEC = importlib.util.spec_from_file_location("gm_perception_app", APP_PATH)
assert SPEC is not None and SPEC.loader is not None
APP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = APP
SPEC.loader.exec_module(APP)


def _png() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), color=(100, 40, 20)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class _Inputs(dict):
    def __init__(self):
        super().__init__(input_ids=torch.tensor([[1]]))

    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, device):
        return self


class _Processor:
    def __call__(self, **kwargs):
        return _Inputs()

    def post_process_grounded_object_detection(self, *args, **kwargs):
        return [
            {
                "boxes": torch.tensor([[2.0, 3.0, 20.0, 18.0]]),
                "scores": torch.tensor([0.9]),
                "text_labels": ["hand"],
            }
        ]


class _Model:
    def __call__(self, **kwargs):
        return object()


class _Predictor:
    def set_image(self, image):
        self.image = image

    def predict(self, *, box, multimask_output):
        mask = np.zeros((1, 24, 32), dtype=bool)
        mask[:, 3:18, 2:20] = True
        return mask, np.array([0.95]), None


class PerceptionContractTest(unittest.TestCase):
    def setUp(self) -> None:
        APP._gdino_processor = _Processor()
        APP._gdino_model = _Model()
        APP._sam2_predictor = _Predictor()
        APP._sessions.clear()

    def test_health_pins_both_model_identities(self) -> None:
        health = APP.health()
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["gdino_model_id"], "IDEA-Research/grounding-dino-base")
        self.assertEqual(health["sam2_model_id"], "sam2.1_hiera_small.pt")

    def test_ground_returns_canonical_model_versions(self) -> None:
        response = APP.ground(
            APP.GroundRequest(
                text_prompt="hand . robot",
                keywords=["hand"],
                image_b64=_png(),
                request_id="request-1",
                frame_id="frame-1",
            )
        ).model_dump()
        self.assertTrue(response["ok"])
        self.assertEqual(response["request_id"], "request-1")
        self.assertEqual(response["frame_id"], "frame-1")
        self.assertEqual(response["model_versions"], APP._model_versions())
        self.assertEqual(len(response["detections"]), 1)
        self.assertEqual(response["keyword_detection_map"]["hand"], [response["detections"][0]["detection_id"]])

    def test_requested_mask_rle_round_trips(self) -> None:
        response = APP.ground(
            APP.GroundRequest(
                text_prompt="hand",
                image_b64=_png(),
                return_mask_rle=True,
            )
        ).model_dump()
        rle = response["detections"][0]["mask_rle"]
        self.assertEqual(rle["size"], [24, 32])
        self.assertEqual(rle["order"], "C")
        flat = np.zeros(24 * 32, dtype=bool)
        offset = 0
        on = False
        for count in rle["counts"]:
            if on:
                flat[offset : offset + count] = True
            offset += count
            on = not on
        self.assertEqual(offset, flat.size)
        expected = np.zeros((24, 32), dtype=bool)
        expected[3:18, 2:20] = True
        np.testing.assert_array_equal(flat.reshape(24, 32), expected)

    def test_track_is_ordered_and_model_bound(self) -> None:
        first = APP.track(
            APP.TrackRequest(
                action="init",
                frame_index=0,
                image_b64=_png(),
                init=APP.TrackInitParams(box_xyxy=[2.0, 3.0, 20.0, 18.0]),
            )
        ).model_dump()
        second = APP.track(
            APP.TrackRequest(
                action="step",
                frame_index=1,
                image_b64=_png(),
                session_id=first["session_id"],
            )
        ).model_dump()
        self.assertEqual(second["model_versions"], APP._model_versions())
        with self.assertRaises(HTTPException) as caught:
            APP.track(
                APP.TrackRequest(
                    action="step",
                    frame_index=1,
                    image_b64=_png(),
                    session_id=first["session_id"],
                )
            )
        self.assertEqual(caught.exception.status_code, 409)

    def test_noncanonical_base64_fails_closed(self) -> None:
        with self.assertRaises(HTTPException) as caught:
            APP._load_image("not base64", None)
        self.assertEqual(caught.exception.status_code, 400)

    def test_image_pixel_limit_fails_closed(self) -> None:
        original_limit = APP._MAX_IMAGE_PIXELS
        APP._MAX_IMAGE_PIXELS = 16
        try:
            with self.assertRaises(HTTPException) as caught:
                APP._load_image(_png(), None)
            self.assertEqual(caught.exception.status_code, 413)
        finally:
            APP._MAX_IMAGE_PIXELS = original_limit


if __name__ == "__main__":
    unittest.main()
