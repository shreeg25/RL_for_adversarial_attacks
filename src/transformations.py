# src/transformations.py
import cv2
import numpy as np

ACTION_COST = {0: 0.0, 1: 0.05, 2: 0.03, 3: 0.04}

def apply_transformation(frame: np.ndarray, action: int, target_box: list | None) -> np.ndarray:
    if action == 0 or target_box is None:
        return frame
        
    out = frame.copy()
    H, W = frame.shape[:2]
    x, y, w, h = map(int, target_box)
    
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    
    if x2 <= x1 or y2 <= y1:
        return frame 
        
    roi = out[y1:y2, x1:x2]
    roi_h, roi_w = roi.shape[:2]

    if action == 1:   
        src = np.float32([[0, 0], [roi_w, 0], [0, roi_h], [roi_w, roi_h]])
        jitter_amp = max(2, int(roi_w * 0.1)) 
        jitter = np.random.uniform(-jitter_amp, jitter_amp, src.shape).astype(np.float32)
        dst = np.clip(src + jitter, 0, [roi_w, roi_h]).astype(np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        out[y1:y2, x1:x2] = cv2.warpPerspective(roi, M, (roi_w, roi_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    elif action == 2:   
        # Gaussian Defocus: Melts the adversarial patch gradients but keeps the human shape
        k_size = max(3, (roi_w // 10) | 1) # Must be odd
        out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k_size, k_size), 0)

    elif action == 3:   
        # Grid Dropout: Destroys localized patch coherence without hiding the whole bounding box
        grid = np.ones(roi.shape[:2], dtype=np.float32)
        step_y, step_x = max(2, roi_h // 8), max(2, roi_w // 8)
        grid[::step_y, :] = 0.5  
        grid[:, ::step_x] = 0.5
        grid_3d = np.expand_dims(grid, axis=-1)
        out[y1:y2, x1:x2] = (roi * grid_3d).astype(np.uint8)
        
    return out