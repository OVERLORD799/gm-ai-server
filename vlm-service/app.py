#!/usr/bin/env python3
"""Qwen2.5-VL AWQ service with the GMrobot canonical V0-A contract."""
from __future__ import annotations

import base64
import binascii
import io
import json
import math
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)


MODEL_ID = os.environ.get("VLM_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
MODEL_REVISION = os.environ.get(
    "VLM_MODEL_REVISION", "536a35794df8831aa814970ee8f89eff577e7718"
)
API_MODEL_ID = os.environ.get("VLM_API_MODEL_ID", "Qwen2.5-VL-7B-Instruct-awq")
QUANTIZATION = os.environ.get("VLM_QUANTIZATION", "awq").strip().lower()
AWQ_BACKEND = os.environ.get("VLM_AWQ_BACKEND", "torch_fallback").strip().lower()
HOST = os.environ.get("VLM_HOST", "0.0.0.0")
PORT = int(os.environ.get("VLM_PORT", "8080"))
MAX_IMAGE_BYTES = int(os.environ.get("VLM_MAX_IMAGE_BYTES", str(16 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.environ.get("VLM_MAX_IMAGE_PIXELS", str(4096 * 4096)))
ALLOW_IMAGE_PATH = os.environ.get("VLM_ALLOW_IMAGE_PATH", "0") == "1"
TEST_MODE = os.environ.get("GM_AI_TEST_MODE", "0") == "1"

PROMPT_VERSION = "five_stage_safety_v1"
SCHEMA_VERSION = "five_stage_vlm_v1"
_RISK_TYPES = frozenset({"static", "dynamic", "functional", "none"})
_ACTIONS = frozenset({"continue", "slow_down", "stop", "replan", "alert"})
_SPATIAL_HINTS = frozenset({"left", "right", "above", "retreat", "none"})

_model: Qwen2_5_VLForConditionalGeneration | None = None
_processor: Any = None
_model_lock = threading.Lock()


DEFAULT_PROMPT = """You are the advisory semantic safety stage for a simulated UR10e pick-and-place cell.
Inspect the supplied image and return ONLY one JSON object with these exact keys:
{"scene_summary":"...","keywords":["..."],"risk_type":"static|dynamic|functional|none","risk_confidence":0.0,"affected_entities":["..."],"predicted_consequence":"...","prediction_horizon_s":2.0,"time_to_risk_s":null,"explanation":"...","suggested_action":"continue|slow_down|replan|stop","spatial_hint":"left|right|above|retreat|none"}
Do not wrap the JSON in Markdown. Semantics are advisory and may only preserve or tighten safety."""


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: str = Field(default=DEFAULT_PROMPT, min_length=1, max_length=32768)
    image_b64: str | None = None
    image_path: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    request_id: str = Field(default="", max_length=256)
    frame_id: str = Field(default="", max_length=256)
    model_id: str = Field(default=API_MODEL_ID, max_length=256)
    prompt_version: str = Field(default=PROMPT_VERSION, max_length=128)
    schema_version: str = Field(default=SCHEMA_VERSION, max_length=128)
    max_output_tokens: int = Field(default=256, ge=1, le=256)


class AnalyzeResponse(BaseModel):
    ok: bool = True
    request_id: str
    frame_id: str
    scene_summary: str
    keywords: list[str]
    risk_type: str
    risk_confidence: float
    affected_entities: list[str]
    predicted_consequence: str
    prediction_horizon_s: float
    time_to_risk_s: float | None
    explanation: str
    suggested_action: str
    spatial_hint: str
    prompt_version: str
    schema_version: str
    model_id: str
    backend_model_id: str
    quantization: str
    latency_ms: float
    text: str
    vlm_keywords: list[str]
    vlm_risk_type: str
    vlm_risk_confidence: float
    vlm_suggested_action: str
    vlm_explanation: str
    remote_contract: str = "canonical_v0a"


def _prepare_awq_config(config: Any) -> Any:
    quantization_config = config.quantization_config
    if type(quantization_config) is not dict:
        raise RuntimeError("model quantization_config is not a plain object")
    skipped_modules = quantization_config.get("modules_to_not_convert")
    if skipped_modules != ["visual"]:
        raise RuntimeError("model AWQ exclusion list drifted from the pinned snapshot")
    # This pinned Qwen checkpoint stores lm_head.weight as an ordinary FP
    # tensor, not qweight/qzeros/scales. Transformers otherwise replaces that
    # head with an AWQ layer and leaves it randomly initialized.
    quantization_config["modules_to_not_convert"] = ["visual", "lm_head"]
    return config


def _load_model() -> None:
    global _model, _processor
    if TEST_MODE:
        return
    if QUANTIZATION != "awq":
        raise RuntimeError(f"unsupported VLM_QUANTIZATION={QUANTIZATION!r}")
    if AWQ_BACKEND not in {"torch_fallback", "triton"}:
        raise RuntimeError(f"unsupported VLM_AWQ_BACKEND={AWQ_BACKEND!r}")
    if AWQ_BACKEND == "torch_fallback":
        # AutoAWQ 0.2.9's Triton int4 GEMM does not lower for Blackwell
        # compute capability 12.0. Its built-in torch implementation is slower
        # but portable and fail-obvious, which is preferable for recovery.
        import awq.modules.linear.gemm as awq_gemm

        awq_gemm.TRITON_AVAILABLE = False
    config = _prepare_awq_config(
        AutoConfig.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, local_files_only=True
        )
    )
    kwargs: dict[str, Any] = {
        "config": config,
        "revision": MODEL_REVISION,
        "device_map": "auto",
        "local_files_only": True,
        # AutoAWQ's torch fallback performs GEMM in FP16; the snapshot's BF16
        # default would produce a Half/BFloat16 matrix multiplication error.
        "torch_dtype": torch.float16,
    }
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, **kwargs)
    if type(_model.lm_head) is not torch.nn.Linear:
        raise RuntimeError("lm_head was unexpectedly replaced by a quantized layer")
    if _model.lm_head.weight.dtype != torch.float16 or _model.lm_head.weight.device.type == "meta":
        raise RuntimeError("lm_head weight was not loaded as a concrete FP16 tensor")
    _processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_model()
    yield


app = FastAPI(title="GM-SafePick VLM", version="1.0.0", lifespan=lifespan)


def _decode_image(req: AnalyzeRequest) -> Image.Image:
    if req.image_b64:
        if len(req.image_b64) > ((MAX_IMAGE_BYTES + 2) // 3) * 4 + 4:
            raise HTTPException(413, "encoded image exceeds the configured limit")
        try:
            raw = base64.b64decode(req.image_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(400, "image_b64 is not canonical base64") from exc
        if not raw or len(raw) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "image payload is empty or exceeds the configured limit")
        try:
            image = Image.open(io.BytesIO(raw))
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(413, "image dimensions exceed the configured limit")
            image.load()
            return image.convert("RGB")
        except Image.DecompressionBombError as exc:
            raise HTTPException(413, "image dimensions exceed the configured limit") from exc
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(400, "image payload is invalid") from exc
    if req.image_path:
        if not ALLOW_IMAGE_PATH:
            raise HTTPException(400, "image_path is disabled; use image_b64")
        try:
            path = Path(req.image_path).resolve(strict=True)
            image = Image.open(path)
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(413, "image dimensions exceed the configured limit")
            image.load()
            return image.convert("RGB")
        except Image.DecompressionBombError as exc:
            raise HTTPException(413, "image dimensions exceed the configured limit") from exc
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(400, "image_path is invalid") from exc
    raise HTTPException(400, "image_b64 is required")


def _extract_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for offset, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            continue
        if type(value) is dict:
            return value
    raise ValueError("model output does not contain a JSON object")


def _strict_text(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str or (not allow_empty and not value.strip()):
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _validate_semantics(value: dict[str, Any]) -> dict[str, Any]:
    required = {
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
    }
    if set(value) != required:
        missing = sorted(required.difference(value))
        unexpected = sorted(set(value).difference(required))
        raise ValueError(
            f"model output key mismatch: missing={missing}, unexpected={unexpected}"
        )
    keywords = value["keywords"]
    entities = value["affected_entities"]
    if type(keywords) is not list or any(
        type(item) is not str or not item.strip() for item in keywords
    ):
        raise ValueError("keywords must be a list of nonempty strings")
    if type(entities) is not list or any(
        type(item) is not str or not item.strip() for item in entities
    ):
        raise ValueError("affected_entities must be a list of nonempty strings")
    risk_type = _strict_text(value["risk_type"], "risk_type").lower()
    action = _strict_text(value["suggested_action"], "suggested_action").lower()
    hint = _strict_text(value["spatial_hint"], "spatial_hint").lower()
    if risk_type not in _RISK_TYPES or action not in _ACTIONS or hint not in _SPATIAL_HINTS:
        raise ValueError("model output contains an invalid enum")
    if type(value["risk_confidence"]) not in (int, float):
        raise ValueError("risk_confidence must be numeric")
    confidence = float(value["risk_confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("risk_confidence is outside [0,1]")
    if type(value["prediction_horizon_s"]) not in (int, float):
        raise ValueError("prediction_horizon_s must be numeric")
    horizon = float(value["prediction_horizon_s"])
    if not math.isfinite(horizon) or not 0.0 <= horizon <= 30.0:
        raise ValueError("prediction_horizon_s is outside [0,30]")
    time_to_risk_raw = value["time_to_risk_s"]
    if time_to_risk_raw is None:
        time_to_risk = None
    else:
        if type(time_to_risk_raw) not in (int, float):
            raise ValueError("time_to_risk_s must be numeric or null")
        time_to_risk = float(time_to_risk_raw)
        if not math.isfinite(time_to_risk) or not 0.0 <= time_to_risk <= 30.0:
            raise ValueError("time_to_risk_s is outside [0,30]")
    return {
        "scene_summary": _strict_text(value["scene_summary"], "scene_summary"),
        "keywords": [item.strip() for item in keywords],
        "risk_type": risk_type,
        "risk_confidence": confidence,
        "affected_entities": [item.strip() for item in entities],
        "predicted_consequence": _strict_text(
            value["predicted_consequence"], "predicted_consequence", allow_empty=True
        ),
        "prediction_horizon_s": horizon,
        "time_to_risk_s": time_to_risk,
        "explanation": _strict_text(value["explanation"], "explanation"),
        "suggested_action": action,
        "spatial_hint": hint,
    }


def _generate(image: Image.Image, prompt: str, max_output_tokens: int) -> tuple[str, float]:
    if _model is None or _processor is None:
        raise RuntimeError("model is not loaded")
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temporary:
            temporary_name = temporary.name
            image.save(temporary, format="PNG")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": temporary_name},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        started = time.monotonic()
        # Processor tensors and generation both consume GPU/host memory. Keep
        # the complete inference path single-flight, not only model.generate().
        with _model_lock:
            text = _processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = _processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(_model.device)
            with torch.inference_mode():
                output_ids = _model.generate(**inputs, max_new_tokens=max_output_tokens)
            trimmed = [
                output[len(source) :]
                for source, output in zip(inputs.input_ids, output_ids)
            ]
            response = _processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        return response, (time.monotonic() - started) * 1000.0
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)


@app.get("/health")
def health() -> dict[str, Any]:
    loaded = TEST_MODE or (_model is not None and _processor is not None)
    return {
        "status": "ok" if loaded else "warming",
        "model_id": API_MODEL_ID,
        "backend_model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "quantization": QUANTIZATION,
        "awq_backend": AWQ_BACKEND,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_loaded": loaded,
        "contract_mode": "canonical_v0a",
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    if req.model_id != API_MODEL_ID:
        raise HTTPException(409, "requested model_id does not match the deployed model")
    if req.prompt_version != PROMPT_VERSION or req.schema_version != SCHEMA_VERSION:
        raise HTTPException(409, "prompt/schema version does not match the deployed contract")
    image = _decode_image(req)
    try:
        raw_text, latency_ms = _generate(image, req.prompt, req.max_output_tokens)
        semantic = _validate_semantics(_extract_json(raw_text))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(502, f"model output failed canonical validation: {exc}") from exc
    return AnalyzeResponse(
        request_id=req.request_id,
        frame_id=req.frame_id,
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        model_id=API_MODEL_ID,
        backend_model_id=MODEL_ID,
        quantization=QUANTIZATION,
        latency_ms=round(latency_ms, 3),
        text=raw_text,
        vlm_keywords=semantic["keywords"],
        vlm_risk_type=semantic["risk_type"],
        vlm_risk_confidence=semantic["risk_confidence"],
        vlm_suggested_action=semantic["suggested_action"],
        vlm_explanation=semantic["explanation"],
        **semantic,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, workers=1)
