# src/transformations.py
"""
Attention-Driven Action Space:
Defenses are strictly applied inside the boundaries of the vulnerable target_box.
"""
import cv2
import numpy as np

ACTION_COST = {0: 0.0, 1: 0.05, 2: 0.03, 3: 0.04}

def apply_transformation(frame: np.ndarray, action: int, target_box: list | None) -> np.ndarray:
    if action == 0 or target_box is None:
        return frame
        
    out = frame.copy()
    H, W = frame.shape[:2]
    x, y, w, h = map(int, target_box)
    
    # Clip ROI to frame boundaries
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    
    if x2 <= x1 or y2 <= y1:
        return frame # Target out of bounds
        
    roi = out[y1:y2, x1:x2]
    roi_h, roi_w = roi.shape[:2]

    if action == 1:   # T1 — Localized Spatial Warp
        src = np.float32([[0, 0], [roi_w, 0], [0, roi_h], [roi_w, roi_h]])
        jitter_amp = max(2, int(roi_w * 0.1)) # Scale distortion to bounding box size
        jitter = np.random.uniform(-jitter_amp, jitter_amp, src.shape).astype(np.float32)
        dst = np.clip(src + jitter, 0, [roi_w, roi_h]).astype(np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        out[y1:y2, x1:x2] = cv2.warpPerspective(roi, M, (roi_w, roi_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    elif action == 2:   # T2 — Localized Gaussian Noise
        noise = np.random.normal(0, 20, roi.shape).astype(np.int16)
        out[y1:y2, x1:x2] = np.clip(roi.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    elif action == 3:   # T3 — Localized Block Cutout
        bh, bw = max(1, roi_h // 4), max(1, roi_w // 4)
        y0 = np.random.randint(0, roi_h - bh + 1)
        x0 = np.random.randint(0, roi_w - bw + 1)
        out[y1 + y0:y1 + y0 + bh, x1 + x0:x1 + x0 + bw] = 128
        
    return out