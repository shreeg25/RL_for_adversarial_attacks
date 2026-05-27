# src/state_extractor.py
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from src.device import DEVICE

class TrackingStateExtractor:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nn_budget=100,
            max_cosine_distance=0.4,
            embedder_gpu=(DEVICE.type == "cuda"),
        )
        self._prev_conf:  dict[str, float] = {}
        self._prev_cxcy:  dict[str, tuple] = {}

    def reset(self):
        self.tracker.delete_all_tracks() # Cleaner than re-instantiating the object
        self._prev_conf.clear()
        self._prev_cxcy.clear()

    def update(
        self,
        frame_rgb: np.ndarray,
        detections_xywh: list,
        confidences: list,
    ) -> tuple[np.ndarray, list[str]]:
        
        # FIX: deep_sort_realtime expects [ [ [x,y,w,h], confidence, class_id ], ... ]
        # Using "0" instead of None prevents internal tracking crashes on some pip versions
        # FIX: Force conf to 1.0 to bypass deep_sort_realtime's silent dropping of raw logits
        raw = [[d, 1.0, "0"] for d in detections_xywh]
        
        tracks = self.tracker.update_tracks(raw, frame=frame_rgb)

        conf_vels, spatial_jumps, feat_dists = [], [], []

        for t in tracks:
            if not t.is_confirmed():
                continue
            
            tid = t.track_id # Note: deep_sort_realtime uses strings for IDs

            # 1. Confidence Velocity (Negative values = Attack / Bounding box fading)
            cur_conf = float(t.det_conf) if t.det_conf is not None else 1.0
            if tid in self._prev_conf:
                conf_vels.append(cur_conf - self._prev_conf[tid])
            self._prev_conf[tid] = cur_conf

            # 2. Spatial Jump (Detecting bounding box snapping/detachment)
            tlwh = t.to_tlwh()
            obs_cx = float(tlwh[0] + tlwh[2] / 2)
            obs_cy = float(tlwh[1] + tlwh[3] / 2)
            
            if tid in self._prev_cxcy:
                prev_cx, prev_cy = self._prev_cxcy[tid]
                jump = np.sqrt((obs_cx - prev_cx)**2 + (obs_cy - prev_cy)**2)
                spatial_jumps.append(jump)
            self._prev_cxcy[tid] = (obs_cx, obs_cy)

            # 3. Feature Cosine Distance (Visual corruption)
            if t.features and len(t.features) >= 2:
                e1 = np.array(t.features[-2], dtype=np.float32)
                e2 = np.array(t.features[-1], dtype=np.float32)
                denom = np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8
                feat_dists.append(float(1.0 - np.dot(e1, e2) / denom))

        # 4. AGGREGATION: Isolate the worst-case anomaly.
        # We want the defense to trigger if ANY track in the frame is under attack.
        state = np.array([
            np.min(conf_vels)     if conf_vels     else 0.0, # Min because severe drops are negative
            np.max(spatial_jumps) if spatial_jumps else 0.0, # Max because large jumps are bad
            np.max(feat_dists)    if feat_dists    else 0.0, # Max because high distance means visual corruption
        ], dtype=np.float32)

        active_ids = [t.track_id for t in tracks if t.is_confirmed()]
        return state, active_ids