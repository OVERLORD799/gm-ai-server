#!/usr/bin/env python3
"""Qwen2.5-VL-7B 4-bit smoke test."""
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/root/gpufree-data/huggingface")

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

MODEL_ID = os.environ.get("VLM_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
IMAGE_PATH = Path("/root/gpufree-data/vlm-service/sample.jpg")

def main():
    if not IMAGE_PATH.exists():
        img = Image.new("RGB", (640, 480), color=(30, 120, 200))
        img.save(IMAGE_PATH)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    t0 = time.time()
    print(f"Loading {MODEL_ID} (4-bit)...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"Loaded in {time.time()-t0:.1f}s")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(IMAGE_PATH)},
            {"type": "text", "text": "Describe this image in one sentence."},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    t1 = time.time()
    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=64)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    response = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f"Inference {time.time()-t1:.2f}s")
    print("Response:", response.strip())

if __name__ == "__main__":
    main()
