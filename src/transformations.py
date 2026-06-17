import cv2
import numpy as np

# Preserving your cost dictionary for the environment
ACTION_COST = {0: 0.0, 1: 0.05, 2: 0.03, 3: 0.04}

def apply_transformation(frame: np.ndarray, action: int, target_box: list | None) -> np.ndarray:
    if action == 0 or target_box is None:
        return frame.copy()
        
    out = frame.copy()
    H, W = frame.shape[:2]
    
    try:
        x, y, w, h = map(int, target_box)
    except (ValueError, TypeError):
        return out
    
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    
    # Structural Guardrail 1: Ghost Box Prevention
    if x2 <= x1 or y2 <= y1:
        return out 
        
    roi = out[y1:y2, x1:x2]
    if roi.size == 0:
        return out

    # Structural Guardrail 2: C++ Memory Contiguity 
    roi_contiguous = np.ascontiguousarray(roi)

    # THE ULTIMATE SHIELD: Prevent C++ exceptions from killing the RL process
    try:
        if action == 1:   
            # Anti-PGD: Bit-Depth Reduction 
            shift = 5
            quantized = (roi_contiguous >> shift) << shift
            out[y1:y2, x1:x2] = quantized

        elif action == 2:   
            # Anti-EoT: Structural Grid Dropout
            grid = np.ones(roi.shape[:2], dtype=np.float32)
            step_y, step_x = max(2, (y2 - y1) // 8), max(2, (x2 - x1) // 8)
            grid[::step_y, :] = 0.3  
            grid[:, ::step_x] = 0.3
            grid_3d = np.expand_dims(grid, axis=-1)
            out[y1:y2, x1:x2] = (roi_contiguous * grid_3d).astype(np.uint8)
            
        elif action == 3:   
            # Anti-BPDA: Dynamic Resolution Rescale 
            orig_h, orig_w = roi.shape[:2]
            down_h, down_w = max(4, orig_h // 2), max(4, orig_w // 2)
            low_res = cv2.resize(roi_contiguous, (down_w, down_h), interpolation=cv2.INTER_AREA)
            high_res = cv2.resize(low_res, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            out[y1:y2, x1:x2] = high_res

    except Exception as e:
        # If OpenCV panics for any reason, do not crash. Just return the un-transformed frame.
        pass
        
    return out