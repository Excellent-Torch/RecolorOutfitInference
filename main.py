import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# YOLO11 segmentation model (loaded once)
YOLO_SEG = None

# COCO classes for clothing (person=0, but we segment clothes on person)
CLOTHING_CLASSES = [0]  # Person class - we'll extract clothing from person mask


def get_yolo_model():
    """Load YOLO11-medium segmentation model (singleton)."""
    global YOLO_SEG
    if YOLO_SEG is None:
        YOLO_SEG = YOLO('yolo11m-seg.pt')
    return YOLO_SEG


def refine_mask(mask: np.ndarray) -> np.ndarray:
    """Refine mask with morphological operations."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask


def get_clothing_mask(image: np.ndarray) -> np.ndarray:
    """Segment clothing area from person image using YOLO11."""
    model = get_yolo_model()
    results = model(image, verbose=False, conf=0.5)[0]
    
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if results.masks is not None:
        for i, cls in enumerate(results.boxes.cls):
            if int(cls) in CLOTHING_CLASSES:
                seg_mask = results.masks.data[i].cpu().numpy()
                seg_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                mask = np.maximum(mask, (seg_mask * 255).astype(np.uint8))
    
    return refine_mask(mask)


def extract_skin_mask(image: np.ndarray) -> np.ndarray:
    """Detect skin regions to exclude from clothing."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin1 = cv2.inRange(hsv, lower_skin, upper_skin)
    
    lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
    upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
    skin2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    
    skin_mask = cv2.bitwise_or(skin1, skin2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    return skin_mask


def recolor_clothing(image: np.ndarray, target_color: tuple, clothing_mask: np.ndarray) -> np.ndarray:
    """
    Recolor clothing while preserving texture and shading.
    
    Args:
        image: BGR image
        target_color: Target color as (H, S, V) where H is 0-179, S and V are 0-255
        clothing_mask: Binary mask of clothing region
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Create normalized mask
    mask_norm = (clothing_mask / 255.0).astype(np.float32)
    mask_3ch = np.stack([mask_norm] * 3, axis=-1)
    
    # Target HSV
    target_h, target_s, target_v = target_color
    
    # Preserve original value (brightness) for texture
    original_v = hsv[:, :, 2].copy()
    
    # Shift hue to target
    hsv[:, :, 0] = hsv[:, :, 0] * (1 - mask_norm) + target_h * mask_norm
    
    # Blend saturation (keep some original for natural look)
    hsv[:, :, 1] = hsv[:, :, 1] * (1 - mask_norm * 0.7) + target_s * mask_norm * 0.7
    
    # Keep original brightness/texture
    hsv[:, :, 2] = original_v
    
    # Clip values
    hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    
    # Convert back to BGR
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Smooth blending at edges
    mask_blur = cv2.GaussianBlur(mask_norm, (15, 15), 0)
    mask_3ch_blur = np.stack([mask_blur] * 3, axis=-1)
    result = (result * mask_3ch_blur + image * (1 - mask_3ch_blur)).astype(np.uint8)
    
    return result


def recolor(input_path: str, output_path: str, color: str = 'red'):
    """
    Recolor clothing in an image.
    
    Args:
        input_path: Path to input image (person wearing clothes)
        output_path: Path to save result
        color: Target color name or HSV tuple
    """
    # Predefined colors (H, S, V)
    COLORS = {
        'red': (0, 200, 200),
        'blue': (110, 200, 200),
        'green': (60, 200, 200),
        'yellow': (25, 200, 200),
        'purple': (140, 200, 200),
        'pink': (160, 150, 200),
        'orange': (15, 220, 220),
        'cyan': (90, 200, 200),
        'black': (0, 0, 50),
        'white': (0, 20, 240),
        'brown': (10, 150, 120),
        'navy': (110, 200, 100),
    }
    
    # Get target color
    if isinstance(color, str):
        target_hsv = COLORS.get(color.lower(), COLORS['red'])
    else:
        target_hsv = color
    
    # Load image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Could not load image: {input_path}")
    
    # Get clothing mask (person segmentation minus skin)
    person_mask = get_clothing_mask(image)
    skin_mask = extract_skin_mask(image)
    
    # Clothing = person - skin
    clothing_mask = cv2.subtract(person_mask, skin_mask)
    clothing_mask = refine_mask(clothing_mask)
    
    # Recolor
    result = recolor_clothing(image, target_hsv, clothing_mask)
    
    # Save
    cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 100])
    print(f'Saved: {output_path}')
    
    return result


if __name__ == '__main__':
    # Example: Change clothing to blue
    recolor('person.jpg', 'recolored_result.jpg', color='pink')
