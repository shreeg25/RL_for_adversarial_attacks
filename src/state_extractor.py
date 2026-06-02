# src/state_extractor.py
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from src.device import DEVICE

class TrackingStateExtractor:
    def __init__(self):
        # Explicitly configure the appearance embedder to run on the GPU device
        use_gpu_flag = (DEVICE.type == "cuda")
        
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nn_budget=100,
            max_cosine_distance=0.4,
            embedder_gpu=use_gpu_flag, # FLIP TO TRUE: Offloads ReID CNN to CUDA
            half=use_gpu_flag,         # FP16 precision cut for faster GPU inference
        )
        self._prev_conf = {}
        self._prev_cxcy = {}

    def reset(self):
        self.tracker.delete_all_tracks()
        self._prev_conf.clear()
        self._prev_cxcy.clear()

    def update(self, frame_rgb: np.ndarray, detections_xywh: list, confidences: list) -> tuple[np.ndarray, list]:
        import math
        def sigmoid(x):
            x = max(-10.0, min(10.0, float(x)))
            return 1.0 / (1.0 + math.exp(-x))
        
        def bbox_iou(b1, b2):
            x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
            x2 = min(b1[0]+b1[2], b2[0]+b2[2])
            y2 = min(b1[1]+b1[3], b2[1]+b2[3])
            inter = max(0, x2-x1) * max(0, y2-y1)
            union = b1[2]*b1[3] + b2[2]*b2[3] - inter
            return inter / union if union > 0 else 0.0

        raw = [[d, sigmoid(c), "0"] for d, c in zip(detections_xywh, confidences)]
        
        # DeepSORT will now internally call the ReID extraction on the GPU
        tracks = self.tracker.update_tracks(raw, frame=frame_rgb)

        conf_vels, spatial_jumps, feat_dists = [], [], []

        for t in tracks:
            if not t.is_confirmed():
                continue
            
            tid = t.track_id 
            tlwh = t.to_tlwh()
            cur_conf = 0.0 
            best_iou = 0.0
            
            for idx, det_xywh in enumerate(detections_xywh):
                iou = bbox_iou(tlwh, det_xywh)
                if iou > best_iou:
                    best_iou = iou
                    cur_conf = sigmoid(confidences[idx])
            
            if tid in self._prev_conf:
                conf_vels.append(cur_conf - self._prev_conf[tid])
            self._prev_conf[tid] = cur_conf

            obs_cx = float(tlwh[0] + tlwh[2] / 2)
            obs_cy = float(tlwh[1] + tlwh[3] / 2)
            
            if tid in self._prev_cxcy:
                prev_cx, prev_cy = self._prev_cxcy[tid]
                conf_vels.append(np.sqrt((obs_cx - prev_cx)**2 + (obs_cy - prev_cy)**2))
            self._prev_cxcy[tid] = (obs_cx, obs_cy)

            if t.features and len(t.features) >= 2:
                e1 = np.array(t.features[-2], dtype=np.float32)
                e2 = np.array(t.features[-1], dtype=np.float32)
                denom = np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8
                feat_dists.append(float(1.0 - np.dot(e1, e2) / denom))

        state = np.array([
            np.min(conf_vels)     if conf_vels     else 0.0, 
            np.max(conf_vels)     if conf_vels     else 0.0, 
            np.max(feat_dists)    if feat_dists    else 0.0, 
        ], dtype=np.float32)

        active_ids = [t.track_id for t in tracks if t.is_confirmed()]
        return state, active_ids