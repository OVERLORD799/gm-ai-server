"""Qwen2.5-VL-7B-AWQ FastAPI (:8083)."""
import time, io, base64
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoProcessor

MODEL_ID = 'Qwen/Qwen2.5-VL-7B-Instruct-AWQ'
HOST, PORT = '127.0.0.1', 8083

app = FastAPI(title='VLM-AWQ')
model = None
processor = None

class AnalyzeRequest(BaseModel):
    prompt: str = 'Is there a safety risk?'
    image_b64: str | None = None
    image_path: str | None = None
    meta: dict | None = None

def load_model():
    global model, processor
    if model is not None:
        return
    model = AutoModel.from_pretrained(MODEL_ID, device_map='auto', torch_dtype='auto')
    processor = AutoProcessor.from_pretrained(MODEL_ID)

@app.get('/health')
def health():
    return {'status': 'ok', 'model_id': MODEL_ID, 'gpu': torch.cuda.get_device_name(0)}

@app.post('/analyze')
async def analyze(req: AnalyzeRequest):
    t0 = time.monotonic()
    load_model()
    if req.image_b64:
        img = Image.open(io.BytesIO(base64.b64decode(req.image_b64))).convert('RGB')
    elif req.image_path:
        img = Image.open(req.image_path).convert('RGB')
    else:
        raise HTTPException(400, 'image_b64 or image_path required')
    inputs = processor(text=[req.prompt], images=[img], return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)
    text = processor.decode(outputs[0], skip_special_tokens=True)
    latency = (time.monotonic() - t0) * 1000
    return {'ok': True, 'model_id': MODEL_ID, 'text': text, 'latency_ms': latency}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
