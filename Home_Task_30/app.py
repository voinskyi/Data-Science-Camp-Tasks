# app.py
# Simple Object Detection API (YOLOv8 + FastAPI)
# Run: uvicorn app:app --host 0.0.0.0 --port 8000 --reload

from typing import List, Dict, Any, Optional
import io
import base64
import logging
import os

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from PIL import Image
import numpy as np

import torch
from ultralytics import YOLO

# ---------------------------
# Config & Logging
# ---------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("yolo_api")

# ---------------------------
# App
# ---------------------------
app = FastAPI(
    title="Simple Object Detection API",
    description="FastAPI service serving YOLOv8 object detection (pretrained on COCO).",
    version="1.0.0",
)

# Allow local tools / browsers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Model Load
# ---------------------------
# Select device automatically
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = os.environ.get("YOLO_MODEL", "yolov8n.pt")  # small & fast

try:
    log.info(f"Loading YOLO model '{MODEL_NAME}' on device '{DEVICE}' ...")
    model = YOLO(MODEL_NAME)
    # Ultralytics uses torch under the hood; no explicit .to(DEVICE) needed for CPU/GPU switch at predict time
    # We still keep a note:
    if DEVICE == "cuda":
        log.info("CUDA is available. Inference will use GPU.")
    else:
        log.info("CUDA not available. Inference will use CPU.")
    # names mapping
    CLASS_NAMES = model.model.names if hasattr(model.model, "names") else model.names
    log.info("Model loaded successfully.")
except Exception as e:
    log.exception("Failed to load YOLO model.")
    raise

# ---------------------------
# Schemas
# ---------------------------
class Detection(BaseModel):
    cls_id: int
    cls_name: str
    confidence: float
    bbox_xyxy: List[int]  # [x1, y1, x2, y2]

class PredictResponse(BaseModel):
    detections: List[Detection]
    width: int
    height: int
    annotated_image_base64: Optional[str] = None  # PNG if requested

# ---------------------------
# Helpers
# ---------------------------
def pil_image_from_upload(file: UploadFile) -> Image.Image:
    try:
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return image
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

def encode_annotated_image(result) -> str:
    """
    result.plot() returns a BGR numpy array; convert to RGB PIL and then to PNG base64.
    """
    bgr = result.plot()  # numpy array in BGR
    rgb = bgr[..., ::-1]  # BGR -> RGB
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def run_inference(
    image: Image.Image,
    conf: float,
    iou: float,
    max_det: int,
) -> Dict[str, Any]:
    # Convert PIL to numpy (Ultralytics accepts PIL/np/path)
    np_img = np.array(image)
    # Predict
    results = model.predict(
        source=np_img,
        conf=conf,
        iou=iou,
        max_det=max_det,
        verbose=False,
        device=0 if DEVICE == "cuda" else "cpu",
    )
    if not results:
        return {"detections": [], "width": image.width, "height": image.height, "result_obj": None}

    res = results[0]
    detections: List[Detection] = []

    # res.boxes contains bboxes (xyxy), confidence and class
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)

        for i in range(len(clss)):
            x1, y1, x2, y2 = xyxy[i]
            conf_i = float(confs[i])
            cls_i = int(clss[i])
            name_i = CLASS_NAMES.get(cls_i, str(cls_i)) if isinstance(CLASS_NAMES, dict) else str(cls_i)

            detections.append(
                Detection(
                    cls_id=cls_i,
                    cls_name=name_i,
                    confidence=round(conf_i, 4),
                    bbox_xyxy=[int(x1), int(y1), int(x2), int(y2)],
                )
            )

    return {
        "detections": detections,
        "width": res.orig_shape[1],
        "height": res.orig_shape[0],
        "result_obj": res,
    }

# ---------------------------
# Endpoints
# ---------------------------
@app.get("/", tags=["health"])
def root() -> Dict[str, str]:
    return {"status": "ok", "message": "Welcome to the Object Detection API (YOLOv8 + FastAPI)."}

@app.get("/labels", tags=["meta"])
def labels() -> Dict[str, Any]:
    """Return class index-to-name mapping."""
    # Normalize mapping to {int: str}
    if isinstance(CLASS_NAMES, dict):
        mapping = {int(k): str(v) for k, v in CLASS_NAMES.items()}
    else:
        mapping = {i: str(name) for i, name in enumerate(CLASS_NAMES)}
    return {"num_classes": len(mapping), "classes": mapping}

@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["inference"],
    summary="Detect objects on an uploaded image",
)
async def predict(
    file: UploadFile = File(..., description="Image file (jpg, jpeg, png)"),
    conf: float = Query(0.25, ge=0.05, le=0.95, description="Confidence threshold"),
    iou: float = Query(0.45, ge=0.1, le=0.95, description="IoU threshold for NMS"),
    max_det: int = Query(300, ge=1, le=3000, description="Maximum number of detections"),
    return_image: bool = Query(False, description="Return annotated image as base64 PNG"),
):
    # Basic file type check
    content_type = (file.content_type or "").lower()
    if not any(t in content_type for t in ["image/jpeg", "image/jpg", "image/png"]):
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {content_type}")

    # Read image
    img = pil_image_from_upload(file)

    # Inference
    try:
        out = run_inference(img, conf=conf, iou=iou, max_det=max_det)
    except Exception as e:
        log.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    detections: List[Detection] = out["detections"]
    res_obj = out["result_obj"]

    # Optional annotated image
    annotated_b64 = None
    if return_image and res_obj is not None:
        try:
            annotated_b64 = encode_annotated_image(res_obj)
        except Exception as e:
            log.warning(f"Failed to encode annotated image: {e}")

    resp = PredictResponse(
        detections=detections,
        width=out["width"],
        height=out["height"],
        annotated_image_base64=annotated_b64,
    )
    return JSONResponse(content=resp.dict())
