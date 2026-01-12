"""
Clothing Recolor API Server
FastAPI server for recoloring clothing in images.
"""
import traceback
from functools import lru_cache

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from ultralytics import YOLO

app = FastAPI(title="Clothing Recolor API", version="1.0.0")

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_model():
    """Load model once and cache."""
    return YOLO('yolo11m-seg.pt')


@app.on_event("startup")
async def startup():
    get_model()  # Preload


def hex_to_hsv(hex_color: str) -> tuple:
    """Convert #RRGGBB to HSV."""
    hex_color = hex_color.lstrip('#')
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    hsv = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)[0][0]
    return int(hsv[0]), int(hsv[1]), int(hsv[2])


def get_masks(image: np.ndarray) -> np.ndarray:
    """Get clothing mask (person - skin)."""
    h, w = image.shape[:2]
    
    # Person segmentation
    results = get_model()(image, verbose=False, conf=0.5, classes=[0])[0]
    person_mask = np.zeros((h, w), dtype=np.uint8)
    if results.masks is not None and len(results.masks.data) > 0:
        combined = results.masks.data.sum(0).clamp(0, 1).cpu().numpy()
        combined = (combined * 255).astype(np.uint8)  # Convert to uint8 BEFORE resize
        person_mask = cv2.resize(combined, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Skin detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    skin = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
    skin |= cv2.inRange(hsv, np.array([170, 20, 70]), np.array([180, 255, 255]))
    skin = cv2.dilate(skin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=2)
    
    # Clothing = person - skin
    mask = cv2.subtract(person_mask, skin)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.GaussianBlur(mask, (5, 5), 0)


def recolor(image: np.ndarray, target_hsv: tuple, mask: np.ndarray) -> np.ndarray:
    """Recolor clothing preserving texture."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    m = (mask / 255.0).astype(np.float32)
    
    original_v = hsv[:, :, 2].copy()
    hsv[:, :, 0] = hsv[:, :, 0] * (1 - m) + target_hsv[0] * m
    hsv[:, :, 1] = hsv[:, :, 1] * (1 - m * 0.7) + target_hsv[1] * m * 0.7
    hsv[:, :, 2] = original_v
    
    hsv = np.clip(hsv, [0, 0, 0], [179, 255, 255])
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Edge blending
    m_blur = cv2.GaussianBlur(m, (15, 15), 0)[:, :, np.newaxis]
    return (result * m_blur + image * (1 - m_blur)).astype(np.uint8)


@app.post("/recolor")
async def recolor_endpoint(
    image: UploadFile = File(...),
    color: str = Form(..., description="Hex color (#RRGGBB)")
):
    """Recolor clothing. Returns JPG image."""
    try:
        color = color.strip()
        if not color.startswith('#') or len(color) != 7:
            raise HTTPException(400, "Color must be #RRGGBB format")
        
        # Decode image
        img = cv2.imdecode(np.frombuffer(await image.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "Invalid image")
        
        # Process
        mask = get_masks(img)
        result = recolor(img, hex_to_hsv(color), mask)
        
        # Encode JPG
        _, buffer = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return Response(content=buffer.tobytes(), media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
