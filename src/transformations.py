# src/transformations.py
"""
Action space:
  T0 = clean pass       (identity)
  T1 = spatial warp     (perspective distortion — destroys patch geometry)
  T2 = Gaussian noise   (disrupts L_inf gradient calculations)
  T3 = block cutout     (masks random sector — blinds patch location)
"""
import cv2
import numpy as np

# Small constant cost per action, used in reward function
ACTION_COST = {0: 0.0, 1: 0.05, 2: 0.03, 3: 0.04}


def apply_transformation(frame: np.ndarray, action: int) -> np.ndarray:
    """
    Args:
        frame:  H×W×3 uint8 RGB image
        action: int in {0, 1, 2, 3}
    Returns:
        transformed H×W×3 uint8 RGB image
    """
    if action == 0:
        return frame

    elif action == 1:   # T1 — Spatial Warp
        h, w = frame.shape[:2]
        # Jitter the four corners by up to ±15px
        src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        jitter = np.random.uniform(-15, 15, src.shape).astype(np.float32)
        # Fix the broadcasting boundary to match the (4, 2) shape of [x, y] coordinates
        dst = np.clip(src + jitter, 0, [w, h]).astype(np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(frame, M, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)

    elif action == 2:   # T2 — Gaussian Noise
        noise = np.random.normal(0, 15, frame.shape).astype(np.int16)
        noisy = np.clip(frame.astype(np.int16) + noise, 0, 255)
        return noisy.astype(np.uint8)

    elif action == 3:   # T3 — Block Cutout
        out = frame.copy()
        h, w = frame.shape[:2]
        bh, bw = h // 5, w // 5
        y0 = np.random.randint(0, h - bh)
        x0 = np.random.randint(0, w - bw)
        out[y0:y0 + bh, x0:x0 + bw] = 128   # neutral gray patch
        return out

    else:
        raise ValueError(f"Unknown action {action}")