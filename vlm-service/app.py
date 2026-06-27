#!/usr/bin/env python3
"""Minimal VLM HTTP stub for VLMClient remote backend."""
from __future__ import annotations

import base64
import io
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

MODEL_ID = os.environ.get("VLM_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
PORT = int(os.environ.get("VLM_PORT", "8080"))

_model = None
_processor = None


class AnalyzeRequest(BaseModel):
    prompt: str = Field(default="Describe the scene for robot safety.")
    image_b64: str | None = None
    image_path: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class AnalyzeResponse(BaseModel):
    model_id: str
    latency_ms: float
    text: str
    vlm_risk_type: str = "none"
    vlm_severity: str = "low"
    vlm_suggested_action: str = "continue"
    vlm_confidence: float = 0.5


def _load_image(req: AnalyzeRequest) -> Image.Image:
    if req.image_b64:
        raw = base64.b64decode(req.image_b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if req.image_path and os.path.isfile(req.image_path):
        return Image.open(req.image_path).convert("RGB")
    return Image.new("RGB", (640, 480), color=(40, 40, 40))


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _model, _processor
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    yield


app = FastAPI(title="GM-SafePick VLM Service", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_id": MODEL_ID, "gpu": torch.cuda.get_device_name(0)}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if _model is None or _processor is None:
        raise HTTPException(503, "model not loaded")
    image = _load_image(req)
    tmp = "/tmp/vlm_req.jpg"
    image.save(tmp)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": tmp},
            {"type": "text", "text": req.prompt},
        ],
    }]
    text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(_model.device)
    t0 = time.time()
    with torch.inference_mode():
        out_ids = _model.generate(**inputs, max_new_tokens=128)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    response = _processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    latency_ms = (time.time() - t0) * 1000
    return AnalyzeResponse(
        model_id=f"{MODEL_ID.split(chr(47))[-1]}-4bit-nf4",
        latency_ms=latency_ms,
        text=response.strip(),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
