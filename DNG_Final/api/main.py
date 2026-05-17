"""
FastAPI REST API for Detail Level Classification
Samsung Multi-Level Detail Enhancement Detection Project
"""

import io, time, logging, sys, tempfile, os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.append(str(Path(__file__).parent.parent))
from model.classifier import load_model, build_model, DetailLevelClassifier
from utils.metrics import compute_all_metrics, compute_detail_score
from pipeline.dng_processor import DNGProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Samsung RAW Detail Level Classifier", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_PATH = Path("checkpoints_v3/best_model.pt")
NUM_LEVELS = 6
IMG_SIZE   = 224
_model: Optional[DetailLevelClassifier] = None
_processor = DNGProcessor()

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@app.on_event("startup")
async def load_model_on_startup():
    global _model
    if MODEL_PATH.exists():
        _model = load_model(str(MODEL_PATH))
    else:
        logger.warning("No checkpoint — using untrained model for demo.")
        _model = build_model(num_classes=NUM_LEVELS, pretrained=False)
    _model.eval()
    logger.info("Model ready.")


class ClassificationResult(BaseModel):
    detail_level: int
    confidence: float
    level_probabilities: dict
    sharpness_score: float
    processing_time_ms: float
    filename: str


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None,
            "device": "cuda" if torch.cuda.is_available() else "cpu"}


@app.get("/model/info")
async def model_info():
    return {"num_levels": NUM_LEVELS, "backbone": "ConvNeXt-Small",
            "input_size": IMG_SIZE, "checkpoint_exists": MODEL_PATH.exists(),
            "accuracy": "98.53%", "kappa": "0.9968"}


@app.post("/classify", response_model=ClassificationResult)
async def classify(file: UploadFile = File(...)):
    t_start  = time.perf_counter()
    data     = await file.read()
    filename = file.filename or "upload"
    suffix   = Path(filename).suffix.lower()

    try:
        if suffix in (".dng", ".raw", ".nef", ".cr2", ".arw"):
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(data); tmp_path = f.name
            try:
                img = _processor.process(tmp_path)
            finally:
                os.unlink(tmp_path)
        else:
            arr = np.frombuffer(data, np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise HTTPException(status_code=422, detail="Cannot decode image")
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {e}")

    t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    t = (t - IMAGENET_MEAN) / IMAGENET_STD

    with torch.no_grad():
        logits = _model(t)
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = int(probs.argmax().item())
        conf   = float(probs.max().item())

    sharpness  = compute_detail_score(img)
    elapsed_ms = (time.perf_counter() - t_start) * 1000
    level = pred + 5  # 0-indexed → L5..L10
    logger.info(f"[{filename}] Level=L{level} Conf={conf:.3f} Time={elapsed_ms:.1f}ms")

    return ClassificationResult(
        detail_level        = level,
        confidence          = round(conf, 4),
        level_probabilities = {i + 5: round(float(p), 4) for i, p in enumerate(probs)},
        sharpness_score     = round(sharpness, 4),
        processing_time_ms  = round(elapsed_ms, 2),
        filename            = filename,
    )


DASHBOARD_DIR = Path(__file__).parent.parent / "dashboard"
if DASHBOARD_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")

    @app.get("/")
    async def serve_dashboard():
        idx = DASHBOARD_DIR / "index.html"
        return FileResponse(str(idx)) if idx.exists() else {"message": "Run the frontend separately."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
