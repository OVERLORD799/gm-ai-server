#!/usr/bin/env python3
"""GM-SafePick Layer 3 Stage 2: Grounding DINO + SAM2 (minimal MVP)."""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
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
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

GDINO_ID = os.environ.get("GDINO_MODEL_ID", "IDEA-Research/grounding-dino-base")
GDINO_REVISION = os.environ.get(
    "GDINO_MODEL_REVISION", "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
)
SAM2_CFG = os.environ.get("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_s.yaml")
SAM2_CKPT = os.environ.get(
    "SAM2_CHECKPOINT",
    "/data/checkpoints/sam2.1_hiera_small.pt",
)
SAM2_MODEL_ID = os.environ.get("SAM2_MODEL_ID", os.path.basename(SAM2_CKPT))
SAM2_CHECKPOINT_SHA256 = os.environ.get(
    "SAM2_CHECKPOINT_SHA256",
    "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38",
)
PORT = int(os.environ.get("PERCEPTION_PORT", "8082"))
HOST = os.environ.get("PERCEPTION_HOST", "127.0.0.1")
_CONTROL_DT_S = float(os.environ.get("TRACK_CONTROL_DT_S", "0.02"))
_MAX_IMAGE_BYTES = int(
    os.environ.get("PERCEPTION_MAX_IMAGE_BYTES", str(16 * 1024 * 1024))
)
_MAX_IMAGE_PIXELS = int(
    os.environ.get("PERCEPTION_MAX_IMAGE_PIXELS", str(4096 * 4096))
)
_ALLOW_IMAGE_PATH = os.environ.get("PERCEPTION_ALLOW_IMAGE_PATH", "0") == "1"
_EAGER_LOAD = os.environ.get("PERCEPTION_EAGER_LOAD", "0") == "1"
_TEST_MODE = os.environ.get("GM_AI_TEST_MODE", "0") == "1"
_MAX_TRACK_SESSIONS = int(os.environ.get("MAX_TRACK_SESSIONS", "128"))

_lock = threading.Lock()
_track_lock = threading.Lock()
_gdino_model = None
_gdino_processor = None
_sam2_predictor = None
_sessions: dict[str, dict] = {}


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_models() -> None:
    global _gdino_model, _gdino_processor, _sam2_predictor
    if _gdino_model is not None and _sam2_predictor is not None:
        return
    with _lock:
        if _TEST_MODE:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required by the perception service")
        if _gdino_model is None:
            _gdino_processor = AutoProcessor.from_pretrained(
                GDINO_ID, revision=GDINO_REVISION, local_files_only=True
            )
            _gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                GDINO_ID, revision=GDINO_REVISION, local_files_only=True
            ).to("cuda")
            _gdino_model.eval()
        if _sam2_predictor is None:
            if not os.path.isfile(SAM2_CKPT):
                raise RuntimeError(f"SAM2 checkpoint is missing: {SAM2_CKPT}")
            if _sha256_file(SAM2_CKPT) != SAM2_CHECKPOINT_SHA256:
                raise RuntimeError("SAM2 checkpoint SHA-256 mismatch")
            _sam2_predictor = SAM2ImagePredictor(
                build_sam2(SAM2_CFG, SAM2_CKPT, device="cuda")
            )


def _load_image(image_b64: str | None, image_path: str | None) -> Image.Image:
    if image_b64:
        if len(image_b64) > ((_MAX_IMAGE_BYTES + 2) // 3) * 4 + 4:
            raise HTTPException(413, "encoded image exceeds the configured limit")
        try:
            raw = base64.b64decode(image_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(400, "image_b64 is not canonical base64") from exc
        if not raw or len(raw) > _MAX_IMAGE_BYTES:
            raise HTTPException(413, "image payload is empty or exceeds the configured limit")
        try:
            image = Image.open(io.BytesIO(raw))
            if image.width * image.height > _MAX_IMAGE_PIXELS:
                raise HTTPException(413, "image dimensions exceed the configured limit")
            image.load()
            return image.convert("RGB")
        except Image.DecompressionBombError as exc:
            raise HTTPException(413, "image dimensions exceed the configured limit") from exc
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(400, "image payload is invalid") from exc
    if image_path:
        if not _ALLOW_IMAGE_PATH:
            raise HTTPException(400, "image_path is disabled; use image_b64")
        try:
            image = Image.open(image_path)
            if image.width * image.height > _MAX_IMAGE_PIXELS:
                raise HTTPException(413, "image dimensions exceed the configured limit")
            image.load()
            return image.convert("RGB")
        except Image.DecompressionBombError as exc:
            raise HTTPException(413, "image dimensions exceed the configured limit") from exc
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(400, "image_path is invalid") from exc
    raise HTTPException(status_code=400, detail="image_b64 is required")


class GroundRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text_prompt: str = Field(..., description="Grounding DINO phrases, use ' . ' separator")
    keywords: list[str] = Field(default_factory=list)
    image_b64: str | None = None
    image_path: str | None = None
    box_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    text_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    max_detections: int = Field(default=10, ge=1, le=100)
    run_sam2: bool = True
    return_mask_rle: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)
    request_id: str = Field(default="", max_length=256)
    frame_id: str = Field(default="", max_length=256)


class Detection(BaseModel):
    detection_id: str
    label: str
    score: float
    box_xyxy: list[float]
    mask_area: int | None = None
    sam2_score: float | None = None
    mask_rle: dict[str, Any] | None = None


class GroundResponse(BaseModel):
    ok: bool = True
    request_id: str
    frame_id: str
    gdino_model_id: str
    sam2_checkpoint: str
    sam2_model_id: str
    model_versions: dict[str, str]
    latency_ms: float
    detections: list[Detection]
    keyword_detection_map: dict[str, list[str]]
    perception_status: str
    remote_contract: str = "canonical_v0a"


class TrackInitParams(BaseModel):
    target_label: str = Field(default="hand", min_length=1, max_length=256)
    text_prompt: str = Field(
        default="gloved hand . robot gripper", min_length=1, max_length=4096
    )
    box_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    re_detect_every_n: int = Field(default=100, ge=0, le=100000)
    box_xyxy: list[float] | None = Field(default=None, min_length=4, max_length=4)


class TrackRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    action: str
    frame_index: int = Field(default=0, ge=0)
    image_b64: str = Field(min_length=1)
    session_id: str | None = Field(default=None, max_length=256)
    init: TrackInitParams | None = None
    return_mask_rle: bool = False
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
    mask_rle: dict[str, Any] | None = None


class TrackResponse(BaseModel):
    ok: bool = True
    session_id: str
    frame_index: int
    re_detected: bool = False
    latency_ms: float
    tracks: list[TrackItem]
    model_versions: dict[str, str]
    remote_contract: str = "canonical_v0a"


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


def _encode_mask_rle(mask: np.ndarray) -> dict[str, Any]:
    """Encode a boolean mask as row-major runs beginning with a zero run."""
    flat = np.asarray(mask, dtype=bool).ravel(order="C")
    if flat.size == 0:
        return {"size": [0, 0], "order": "C", "counts": []}
    edges = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate(([0], edges, [flat.size]))
    counts = np.diff(bounds).tolist()
    if flat[0]:
        counts = [0] + counts
    return {
        "size": [int(mask.shape[0]), int(mask.shape[1])],
        "order": "C",
        "counts": [int(count) for count in counts],
    }


def _sam2_track_box(
    image: Image.Image, box: list[float]
) -> tuple[list[float], int, float, np.ndarray]:
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
    return out_box, int(masks[0].sum()), float(m_scores[0]), masks[0]


def _model_versions() -> dict[str, str]:
    return {
        "gdino_model_id": GDINO_ID,
        "gdino_model_revision": GDINO_REVISION,
        "sam2_model_id": SAM2_MODEL_ID,
        "sam2_checkpoint_sha256": SAM2_CHECKPOINT_SHA256,
        "sam2_checkpoint": os.path.basename(SAM2_CKPT),
    }


@asynccontextmanager
async def lifespan(_: FastAPI):
    if _EAGER_LOAD:
        _ensure_models()
    yield


app = FastAPI(title="GM-SafePick Perception Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    loaded = _TEST_MODE or (_gdino_model is not None and _sam2_predictor is not None)
    return {
        "status": "ok" if loaded else "warming",
        "gdino_model_id": GDINO_ID,
        "gdino_model_revision": GDINO_REVISION,
        "sam2_checkpoint": os.path.basename(SAM2_CKPT),
        "sam2_model_id": SAM2_MODEL_ID,
        "sam2_checkpoint_sha256": SAM2_CHECKPOINT_SHA256,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "models_loaded": loaded,
        "contract_mode": "canonical_v0a",
    }


def _ground_impl(req: GroundRequest) -> GroundResponse:
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
            box_values = [float(x) for x in box]
            detection_key = json.dumps(
                [req.frame_id, label, box_values, int(i)],
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
            det = Detection(
                detection_id="det-" + hashlib.sha256(detection_key).hexdigest()[:24],
                label=label,
                score=float(scores[i]),
                box_xyxy=box_values,
            )
            if req.run_sam2 and _sam2_predictor is not None:
                masks, m_scores, _ = _sam2_predictor.predict(
                    box=np.array(box, dtype=np.float32), multimask_output=False
                )
                det.mask_area = int(masks[0].sum())
                det.sam2_score = float(m_scores[0])
                if req.return_mask_rle:
                    det.mask_rle = _encode_mask_rle(masks[0])
            detections.append(det)

    keyword_map: dict[str, list[str]] = {}
    for keyword in req.keywords:
        if type(keyword) is not str or not keyword.strip():
            continue
        token = keyword.strip().casefold()
        keyword_map[keyword.strip()] = [
            detection.detection_id
            for detection in detections
            if token in detection.label.casefold() or detection.label.casefold() in token
        ]
    return GroundResponse(
        request_id=req.request_id,
        frame_id=req.frame_id,
        gdino_model_id=GDINO_ID,
        sam2_checkpoint=os.path.basename(SAM2_CKPT),
        sam2_model_id=SAM2_MODEL_ID,
        model_versions=_model_versions(),
        latency_ms=round((time.time() - t0) * 1000, 1),
        detections=detections,
        keyword_detection_map=keyword_map,
        perception_status="detected" if detections else "no_detections",
    )


@app.post("/ground", response_model=GroundResponse)
def ground(req: GroundRequest) -> GroundResponse:
    # GDINO and the shared SAM2 image predictor are not concurrency-safe.
    with _track_lock:
        return _ground_impl(req)


def _track_impl(req: TrackRequest) -> TrackResponse:
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
        out_box, mask_area, sam2_score, mask = _sam2_track_box(image, box)
        cx, cy = _box_center(out_box)
        session_id = str(uuid.uuid4())
        if len(_sessions) >= _MAX_TRACK_SESSIONS:
            oldest = min(
                _sessions,
                key=lambda key: float(_sessions[key].get("updated_monotonic_s", 0.0)),
            )
            del _sessions[oldest]
        _sessions[session_id] = {
            "label": label,
            "target_label": init.target_label,
            "text_prompt": init.text_prompt,
            "box_threshold": init.box_threshold,
            "re_detect_every_n": init.re_detect_every_n,
            "last_box": out_box,
            "last_center": (cx, cy),
            "last_frame_index": req.frame_index,
            "updated_monotonic_s": time.monotonic(),
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
                mask_rle=_encode_mask_rle(mask) if req.return_mask_rle else None,
            )
        ]
        return TrackResponse(
            session_id=session_id,
            frame_index=req.frame_index,
            re_detected=re_detected,
            latency_ms=round((time.time() - t0) * 1000, 1),
            tracks=tracks,
            model_versions=_model_versions(),
        )

    if not req.session_id or req.session_id not in _sessions:
        raise HTTPException(status_code=404, detail="unknown session_id")
    state = _sessions[req.session_id]
    prev_idx = state["last_frame_index"]
    if req.frame_index <= prev_idx:
        raise HTTPException(status_code=409, detail="frame_index must increase strictly")
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

    out_box, mask_area, sam2_score, mask = _sam2_track_box(image, box)
    cx, cy = _box_center(out_box)
    prev_cx, prev_cy = state["last_center"]
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
    state["updated_monotonic_s"] = time.monotonic()

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
            mask_rle=_encode_mask_rle(mask) if req.return_mask_rle else None,
        )
    ]
    return TrackResponse(
        session_id=req.session_id,
        frame_index=req.frame_index,
        re_detected=re_detected,
        latency_ms=round((time.time() - t0) * 1000, 1),
        tracks=tracks,
        model_versions=_model_versions(),
    )


@app.post("/track", response_model=TrackResponse)
def track(req: TrackRequest) -> TrackResponse:
    # One in-flight tracking operation preserves session/frame ordering.
    with _track_lock:
        return _track_impl(req)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
