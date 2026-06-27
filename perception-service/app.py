#!/root/gpufree-data/conda-envs/vlm/bin/python
"""GM-SafePick Layer 3 Stage 2: Grounding DINO + SAM2 (minimal MVP)."""
from __future__ import annotations

import base64
import io
import math
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

GDINO_ID = os.environ.get("GDINO_MODEL_ID", "IDEA-Research/grounding-dino-tiny")
SAM2_CFG = os.environ.get("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_t.yaml")
SAM2_CKPT = os.environ.get(
    "SAM2_CHECKPOINT",
    "/root/gpufree-data/perception-service/checkpoints/sam2.1_hiera_tiny.pt",
)
PORT = int(os.environ.get("PERCEPTION_PORT", "8082"))
HOST = os.environ.get("PERCEPTION_HOST", "127.0.0.1")
_CONTROL_DT_S = float(os.environ.get("TRACK_CONTROL_DT_S", "0.02"))

_lock = threading.Lock()
_gdino_model = None
_gdino_processor = None
_sam2_predictor = None
_sessions: dict[str, dict] = {}


def _ensure_models() -> None:
    global _gdino_model, _gdino_processor, _sam2_predictor
    if _gdino_model is not None and _sam2_predictor is not None:
        return
    with _lock:
        if _gdino_model is None:
            _gdino_processor = AutoProcessor.from_pretrained(GDINO_ID)
            _gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_ID).to("cuda")
            _gdino_model.eval()
        if _sam2_predictor is None:
            _sam2_predictor = SAM2ImagePredictor(
                build_sam2(SAM2_CFG, SAM2_CKPT, device="cuda")
            )


def _load_image(image_b64: str | None, image_path: str | None) -> Image.Image:
    if image_b64:
        raw = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if image_path and os.path.isfile(image_path):
        return Image.open(image_path).convert("RGB")
    raise HTTPException(status_code=400, detail="image_b64 or image_path required")


class GroundRequest(BaseModel):
    text_prompt: str = Field(..., description="Grounding DINO phrases, use ' . ' separator")
    image_b64: str | None = None
    image_path: str | None = None
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    max_detections: int = 10
    run_sam2: bool = True
    meta: dict[str, Any] = Field(default_factory=dict)


class Detection(BaseModel):
    label: str
    score: float
    box_xyxy: list[float]
    mask_area: int | None = None
    sam2_score: float | None = None


class GroundResponse(BaseModel):
    gdino_model_id: str
    sam2_checkpoint: str
    latency_ms: float
    detections: list[Detection]


class TrackInitParams(BaseModel):
    target_label: str = "hand"
    text_prompt: str = "gloved hand . robot gripper"
    box_threshold: float = 0.25
    re_detect_every_n: int = 100
    box_xyxy: list[float] | None = None


class TrackRequest(BaseModel):
    action: str
    frame_index: int = 0
    image_b64: str
    session_id: str | None = None
    init: TrackInitParams | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class TrackItem(BaseModel):
    track_id: int = 0
    label: str
    box_xyxy: list[float]
    center_xy: list[float]
    velocity_xy_px_s: list[float] | None = None
    speed_px_s: float | None = None
    direction_deg: float | None = None
    mask_area: int | None = None
    sam2_score: float | None = None


class TrackResponse(BaseModel):
    session_id: str
    frame_index: int
    re_detected: bool = False
    latency_ms: float
    tracks: list[TrackItem]


def _box_center(box: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0


def _gdino_best_box(image: Image.Image, text_prompt: str, box_threshold: float) -> tuple[list[float], str, float] | None:
    assert _gdino_model is not None and _gdino_processor is not None
    inputs = _gdino_processor(images=image, text=text_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = _gdino_model(**inputs)
    results = _gdino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=box_threshold,
        target_sizes=[image.size[::-1]],
    )[0]
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results.get("text_labels") or results.get("labels") or []
    if len(boxes) == 0:
        return None
    best_i = int(np.argmax(scores))
    label = labels[best_i] if best_i < len(labels) else "object"
    if not isinstance(label, str):
        label = str(label)
    return boxes[best_i].tolist(), label, float(scores[best_i])


def _sam2_track_box(image: Image.Image, box: list[float]) -> tuple[list[float], int, float]:
    assert _sam2_predictor is not None
    img_np = np.array(image)
    _sam2_predictor.set_image(img_np)
    masks, m_scores, _ = _sam2_predictor.predict(
        box=np.array(box, dtype=np.float32), multimask_output=False
    )
    ys, xs = np.where(masks[0])
    if len(xs) == 0:
        out_box = [float(x) for x in box]
    else:
        out_box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    return out_box, int(masks[0].sum()), float(m_scores[0])


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Lazy load on first /ground to keep VLM startup unaffected.
    yield


app = FastAPI(title="GM-SafePick Perception Service", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    loaded = _gdino_model is not None and _sam2_predictor is not None
    return {
        "status": "ok" if loaded else "warming",
        "gdino_model_id": GDINO_ID,
        "sam2_checkpoint": os.path.basename(SAM2_CKPT),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "models_loaded": loaded,
    }


@app.post("/ground", response_model=GroundResponse)
def ground(req: GroundRequest):
    t0 = time.time()
    _ensure_models()
    image = _load_image(req.image_b64, req.image_path)
    assert _gdino_model is not None and _gdino_processor is not None

    inputs = _gdino_processor(images=image, text=req.text_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = _gdino_model(**inputs)
    results = _gdino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=req.box_threshold,
        text_threshold=req.text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results.get("text_labels") or results.get("labels") or []

    detections: list[Detection] = []
    if len(boxes):
        order = np.argsort(-scores)[: req.max_detections]
        img_np = np.array(image)
        if req.run_sam2 and _sam2_predictor is not None:
            _sam2_predictor.set_image(img_np)
        for i in order:
            label = labels[i] if i < len(labels) else str(labels[i]) if labels else "object"
            if not isinstance(label, str):
                label = str(label)
            box = boxes[i].tolist()
            det = Detection(label=label, score=float(scores[i]), box_xyxy=[float(x) for x in box])
            if req.run_sam2 and _sam2_predictor is not None:
                masks, m_scores, _ = _sam2_predictor.predict(
                    box=np.array(box, dtype=np.float32), multimask_output=False
                )
                det.mask_area = int(masks[0].sum())
                det.sam2_score = float(m_scores[0])
            detections.append(det)

    return GroundResponse(
        gdino_model_id=GDINO_ID,
        sam2_checkpoint=os.path.basename(SAM2_CKPT),
        latency_ms=round((time.time() - t0) * 1000, 1),
        detections=detections,
    )


@app.post("/track", response_model=TrackResponse)
def track(req: TrackRequest):
    t0 = time.time()
    _ensure_models()
    if req.action not in ("init", "step"):
        raise HTTPException(status_code=400, detail="action must be init or step")

    image = _load_image(req.image_b64, None)
    re_detected = False

    if req.action == "init":
        init = req.init or TrackInitParams()
        if init.box_xyxy:
            box = [float(x) for x in init.box_xyxy]
            label = init.target_label
        else:
            det = _gdino_best_box(image, init.text_prompt, init.box_threshold)
            if det is None:
                raise HTTPException(status_code=422, detail="no detection for track init")
            box, label, _ = det
            re_detected = True
        out_box, mask_area, sam2_score = _sam2_track_box(image, box)
        cx, cy = _box_center(out_box)
        session_id = str(uuid.uuid4())
        _sessions[session_id] = {
            "label": label,
            "target_label": init.target_label,
            "text_prompt": init.text_prompt,
            "box_threshold": init.box_threshold,
            "re_detect_every_n": init.re_detect_every_n,
            "last_box": out_box,
            "last_center": (cx, cy),
            "last_frame_index": req.frame_index,
        }
        tracks = [
            TrackItem(
                track_id=0,
                label=label,
                box_xyxy=out_box,
                center_xy=[cx, cy],
                velocity_xy_px_s=[0.0, 0.0],
                speed_px_s=0.0,
                direction_deg=0.0,
                mask_area=mask_area,
                sam2_score=sam2_score,
            )
        ]
        return TrackResponse(
            session_id=session_id,
            frame_index=req.frame_index,
            re_detected=re_detected,
            latency_ms=round((time.time() - t0) * 1000, 1),
            tracks=tracks,
        )

    if not req.session_id or req.session_id not in _sessions:
        raise HTTPException(status_code=404, detail="unknown session_id")
    state = _sessions[req.session_id]
    init = TrackInitParams(
        target_label=state["target_label"],
        text_prompt=state["text_prompt"],
        box_threshold=state["box_threshold"],
        re_detect_every_n=state["re_detect_every_n"],
    )
    label = state["label"]
    box = state["last_box"]
    if (
        init.re_detect_every_n > 0
        and req.frame_index > 0
        and req.frame_index % init.re_detect_every_n == 0
    ):
        det = _gdino_best_box(image, init.text_prompt, init.box_threshold)
        if det is not None:
            box, label, _ = det
            re_detected = True

    out_box, mask_area, sam2_score = _sam2_track_box(image, box)
    cx, cy = _box_center(out_box)
    prev_cx, prev_cy = state["last_center"]
    prev_idx = state["last_frame_index"]
    steps = max(req.frame_index - prev_idx, 1)
    elapsed = steps * _CONTROL_DT_S
    vx = (cx - prev_cx) / elapsed
    vy = (cy - prev_cy) / elapsed
    speed = math.hypot(vx, vy)
    direction = math.degrees(math.atan2(vy, vx)) if speed > 1e-6 else 0.0

    state["label"] = label
    state["last_box"] = out_box
    state["last_center"] = (cx, cy)
    state["last_frame_index"] = req.frame_index

    tracks = [
        TrackItem(
            track_id=0,
            label=label,
            box_xyxy=out_box,
            center_xy=[cx, cy],
            velocity_xy_px_s=[vx, vy],
            speed_px_s=speed,
            direction_deg=direction,
            mask_area=mask_area,
            sam2_score=sam2_score,
        )
    ]
    return TrackResponse(
        session_id=req.session_id,
        frame_index=req.frame_index,
        re_detected=re_detected,
        latency_ms=round((time.time() - t0) * 1000, 1),
        tracks=tracks,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
