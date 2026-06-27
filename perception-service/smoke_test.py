#!/root/gpufree-data/conda-envs/vlm/bin/python
"""Offline smoke: GDINO + SAM2 on sample image (no HTTP)."""
import os
import time

os.environ.setdefault("HF_HOME", "/root/gpufree-data/huggingface")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

SAMPLE = "/root/gpufree-data/vlm-service/sample.jpg"
PROMPT = "person . hand . cup . table"
GDINO_ID = "IDEA-Research/grounding-dino-tiny"
SAM2_CKPT = "/root/gpufree-data/perception-service/checkpoints/sam2.1_hiera_tiny.pt"


def main() -> None:
    t0 = time.time()
    image = Image.open(SAMPLE).convert("RGB")
    processor = AutoProcessor.from_pretrained(GDINO_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_ID).to("cuda")
    inputs = processor(images=image, text=PROMPT, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    res = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=0.2, text_threshold=0.2, target_sizes=[image.size[::-1]]
    )[0]
    n = len(res["boxes"])
    print(f"gdino: {n} boxes in {time.time()-t0:.1f}s")

    predictor = SAM2ImagePredictor(
        build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", SAM2_CKPT, device="cuda")
    )
    predictor.set_image(np.array(image))
    if n:
        box = res["boxes"][0].cpu().numpy()
        masks, scores, _ = predictor.predict(box=box, multimask_output=False)
        print(f"sam2: score={float(scores[0]):.3f} area={int(masks[0].sum())}")
    else:
        h, w = image.size[1], image.size[0]
        box = np.array([w * 0.3, h * 0.3, w * 0.7, h * 0.7])
        masks, scores, _ = predictor.predict(box=box, multimask_output=False)
        print(f"sam2 (fallback box): score={float(scores[0]):.3f}")
    print("vram_mb", torch.cuda.memory_allocated() // 1024 // 1024)
    print("smoke_ok")


if __name__ == "__main__":
    main()
