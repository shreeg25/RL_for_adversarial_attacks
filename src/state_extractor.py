# src/state_extractor.py
import math
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

EMBEDDER_GPU = True 

def _best_iou_vectorised(tlwh: np.ndarray, det_boxes: np.ndarray, det_confs: np.ndarray) -> float:
    if len(det_boxes) == 0:
        return 0.0
    t_x1, t_y1 = tlwh[0], tlwh[1]
    t_x2, t_y2 = tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]
    d_x1, d_y1 = det_boxes[:, 0], det_boxes[:, 1]
    d_x2, d_y2 = det_boxes[:, 0] + det_boxes[:, 2], det_boxes[:, 1] + det_boxes[:, 3]

    ix1, iy1 = np.maximum(t_x1, d_x1), np.maximum(t_y1, d_y1)
    ix2, iy2 = np.minimum(t_x2, d_x2), np.minimum(t_y2, d_y2)
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)

    t_area = tlwh[2] * tlwh[3]
    d_area = det_boxes[:, 2] * det_boxes[:, 3]
    union  = t_area + d_area - inter + 1e-6

    iou = inter / union
    best_idx = int(np.argmax(iou))
    return float(det_confs[best_idx]) if iou[best_idx] > 0.0 else 0.0

def _image_stats(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return np.array([
        gray.mean(), gray.std(), 
        np.mean(np.abs(np.diff(gray, axis=1))), 
        np.mean(np.abs(np.diff(gray, axis=0)))
    ], dtype=np.float32)

class TrackingStateExtractor:
    def __init__(self):
        self.tracker = DeepSort(max_age=30, n_init=3, nn_budget=100, max_cosine_distance=0.4, embedder_gpu=EMBEDDER_GPU)
        self._prev_conf: dict[int, float] = {}
        self._prev_cxcy: dict[int, tuple] = {}

    def reset(self):
        self.tracker = DeepSort(max_age=30, n_init=3, nn_budget=100, max_cosine_distance=0.4, embedder_gpu=EMBEDDER_GPU)
        self._prev_conf.clear()
        self._prev_cxcy.clear()

    def update(self, frame_rgb: np.ndarray, detections_xywh: list, confidences: list) -> tuple[np.ndarray, list, list | None]:
        # No sigmoid: detector scores are already probabilities in [0,1].
        raw = [[d, c, "0"] for d, c in zip(detections_xywh, confidences)]

        if detections_xywh:
            det_arr = np.array(detections_xywh, dtype=np.float32)
            conf_arr = np.array(confidences, dtype=np.float32)
        else:
            det_arr = np.empty((0, 4), dtype=np.float32)
            conf_arr = np.empty((0,), dtype=np.float32)

        tracks = self.tracker.update_tracks(raw, frame=frame_rgb)
        
        conf_vels, spatial_jumps, feat_dists = [], [], []
        min_conf = 1.0
        vulnerable_box = None

        for t in tracks:
            if not t.is_confirmed(): continue
            tid, tlwh = t.track_id, t.to_tlwh()
            
            cur_conf = _best_iou_vectorised(tlwh, det_arr, conf_arr)
            
            if cur_conf < min_conf:
                min_conf = cur_conf
                vulnerable_box = tlwh

            if tid in self._prev_conf:
                conf_vels.append(cur_conf - self._prev_conf[tid])
            self._prev_conf[tid] = cur_conf

            obs_cx, obs_cy = float(tlwh[0] + tlwh[2] / 2), float(tlwh[1] + tlwh[3] / 2)
            if tid in self._prev_cxcy:
                px, py = self._prev_cxcy[tid]
                spatial_jumps.append(math.sqrt((obs_cx - px) ** 2 + (obs_cy - py) ** 2))
            self._prev_cxcy[tid] = (obs_cx, obs_cy)

            if t.features and len(t.features) >= 2:
                e1, e2 = np.asarray(t.features[-2], dtype=np.float32), np.asarray(t.features[-1], dtype=np.float32)
                denom = np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8
                feat_dists.append(float(1.0 - np.dot(e1, e2) / denom))

        global_state = np.array([
            float(np.min(conf_vels)) if conf_vels else 0.0,
            float(np.max(spatial_jumps)) if spatial_jumps else 0.0,
            float(np.max(feat_dists)) if feat_dists else 0.0,
        ], dtype=np.float32)
        global_state = np.concatenate([global_state, _image_stats(frame_rgb)])

        H, W = frame_rgb.shape[:2]
        if vulnerable_box is not None:
            local_state = np.array([
                vulnerable_box[0] / W, vulnerable_box[1] / H, 
                vulnerable_box[2] / W, vulnerable_box[3] / H, 
                min_conf
            ], dtype=np.float32)
            target_out = list(vulnerable_box)
        else:
            local_state = np.zeros(5, dtype=np.float32)
            target_out = None

        final_state = np.concatenate([global_state, local_state])
        active_ids = [t.track_id for t in tracks if t.is_confirmed()]
        
        return final_state, active_ids, target_out