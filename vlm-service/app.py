#!/usr/bin/env python3
"""VLM service — structured JSON safety output (G1–G4)."""
from __future__ import annotations

import base64, io, json, os, re, time, threading
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

_model = None; _processor = None; _model_lock = threading.Lock()

SAFETY_SYSTEM_PROMPT = (
    "You monitor a UR10e robot doing pick-and-place in a factory. "
    "The top-down camera shows the robot arm, colored blocks (parts), "
    "and two containers (source A on the left, target B on the right).\n\n"
    "NORMAL operation: robot gripper holds ONE block during transit, "
    "gripper is EMPTY after placing into target B.  A red ball is a "
    "human hand — it should NOT be near the gripper.\n\n"
    "Return ONLY valid JSON (no markdown):\n"
    '{"keywords": ["objects", "you", "see"], '
    '"risk_type": "static"|"dynamic"|"functional"|"none", '
    '"risk_confidence": 0.0-1.0, '
    '"explanation": "what you see and whether it is safe", '
    '"suggested_action": "continue"|"slow_down"|"replan"|"stop"}\n\n'
    "Risk assessment guide:\n"
    "- risk_type=none confidence=0.9 action=continue → NORMAL: gripper "
    "holding part during transit, or empty gripper after placement, "
    "red ball far from robot\n"
    "- risk_type=static → red ball is NEAR the robot arm or gripper\n"
    "- risk_type=dynamic → red ball is MOVING toward the robot\n"
    "- risk_type=functional → part on floor/table (not in gripper/container), "
    "gripper EMPTY during transit, part at wrong angle\n"
    "- suggested_action=replan ONLY if you see a DANGEROUS situation "
    "(hand near gripper, part dropped).  For normal operation, use continue.\n\n"
    "IMPORTANT: the robot normally holds parts during transit and has "
    "an empty gripper after placing.  These are SAFE states.  Only flag "
    "risks when you see the RED BALL near the gripper or a part OUTSIDE "
    "the containers."
)

def _parse_json(text: str) -> dict:
    """Extract the first complete JSON object using brace counting.

    Unlike regex ``r\"\\{[^{}]*\\}\"``, brace counting correctly handles
    nested objects and arrays (e.g. ``{\"parts\": [{\"label\": \"A\"}]}``).
    """
    if not text:
        return {}
    start = text.find("{")
    if start < 0:
        return {}
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except (json.JSONDecodeError, ValueError):
                    return {}
    # Unclosed brace — try regex as fallback for flat objects.
    try:
        m = re.search(r"\{[^{}]*\}", text)
        if m:
            return json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        pass
    return {}

def _model_id_short() -> str:
    return f"{MODEL_ID.split(chr(47))[-1]}-4bit-nf4"

class AnalyzeRequest(BaseModel):
    prompt: str = Field(default=SAFETY_SYSTEM_PROMPT)
    image_b64: str | None = None
    image_path: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

class AnalyzeResponse(BaseModel):
    model_id: str
    latency_ms: float
    text: str
    vlm_keywords: list[str] = Field(default_factory=list)
    vlm_risk_type: str = "none"
    vlm_risk_confidence: float = 0.0
    vlm_suggested_action: str = "continue"
    vlm_explanation: str = ""

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
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto",
        trust_remote_code=True,
    )
    _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    yield

app = FastAPI(title="GM-SafePick VLM v2", version="0.2.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok", "model_id": MODEL_ID, "gpu": torch.cuda.get_device_name(0)}

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if _model is None or _processor is None:
        raise HTTPException(503, "model not loaded")
    image = _load_image(req)
    tmp = "/tmp/vlm_req.jpg"; image.save(tmp)
    prompt_text = req.prompt if req.prompt else SAFETY_SYSTEM_PROMPT
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": tmp},
            {"type": "text", "text": prompt_text},
        ],
    }]
    text = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(_model.device)
    t0 = time.time()
    with _model_lock, torch.inference_mode():
        out_ids = _model.generate(**inputs, max_new_tokens=256)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    response = _processor.batch_decode(
        trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    latency_ms = (time.time() - t0) * 1000
    parsed = _parse_json(response)
    return AnalyzeResponse(
        model_id=_model_id_short(),
        latency_ms=latency_ms,
        text=response,
        vlm_keywords=list(parsed.get("keywords", [])),
        vlm_risk_type=str(parsed.get("risk_type", "none")),
        vlm_risk_confidence=float(parsed.get("risk_confidence", 0.0)),
        vlm_suggested_action=str(parsed.get(
            "suggested_action",
            parsed.get("action", "continue"),
        )),
        vlm_explanation=str(parsed.get("explanation", response[:300])),
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
