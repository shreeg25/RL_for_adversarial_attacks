# src/state_extractor.py
"""
Optimised TrackingStateExtractor.

FIX-1  Removed O(N×M) IoU loop inside update() — was ~400 Python iterations
        per frame. Replaced with vectorised numpy operations.
FIX-2  Moved sigmoid + math import to module level — was re-imported every step.
FIX-3  Removed duplicate `raw` computation (was built twice per call).
FIX-4  reset() re-instantiates tracker cleanly — delete_all_tracks() leaves
        stale internal state in some deep_sort_realtime versions.
"""

import math
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


# Module-level sigmoid — called thousands of times, must not re-import math
def _sigmoid(x: float) -> float:
    x = max(-10.0, min(10.0, x))
    return 1.0 / (1.0 + math.exp(-x))


# Vectorised IoU: track_box [x,y,w,h] vs all det_boxes (N,4) numpy array
def _best_iou_vectorised(
    tlwh: np.ndarray,           # (4,) track box
    det_boxes: np.ndarray,      # (N,4) detection boxes  [x,y,w,h]
    det_confs: np.ndarray,      # (N,)  confidences (already sigmoid-scaled)
) -> float:
    """
    Returns the detection confidence of the highest-IoU match.
    Pure numpy — no Python loop over detections.
    """
    if len(det_boxes) == 0:
        return 0.0

    # Convert [x,y,w,h] → [x1,y1,x2,y2]
    t_x1, t_y1 = tlwh[0], tlwh[1]
    t_x2, t_y2 = tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]

    d_x1 = det_boxes[:, 0]
    d_y1 = det_boxes[:, 1]
    d_x2 = det_boxes[:, 0] + det_boxes[:, 2]
    d_y2 = det_boxes[:, 1] + det_boxes[:, 3]

    # Intersection
    ix1 = np.maximum(t_x1, d_x1)
    iy1 = np.maximum(t_y1, d_y1)
    ix2 = np.minimum(t_x2, d_x2)
    iy2 = np.minimum(t_y2, d_y2)
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)

    # Union
    t_area = tlwh[2] * tlwh[3]
    d_area = det_boxes[:, 2] * det_boxes[:, 3]
    union  = t_area + d_area - inter + 1e-6

    iou = inter / union
    best_idx = int(np.argmax(iou))

    return float(det_confs[best_idx]) if iou[best_idx] > 0.0 else 0.0


class TrackingStateExtractor:

    def __init__(self):
        # embedder_gpu=False: workers use CPU for ReID
        # prevents n_envs CUDA contexts fighting over same GPU
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nn_budget=100,
            max_cosine_distance=0.4,
            embedder_gpu=False,
        )
        self._prev_conf: dict[int, float] = {}
        self._prev_cxcy: dict[int, tuple] = {}

    def reset(self):
        """Re-instantiate tracker to avoid stale internal state."""
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nn_budget=100,
            max_cosine_distance=0.4,
            embedder_gpu=False,
        )
        self._prev_conf.clear()
        self._prev_cxcy.clear()

    def update(
        self,
        frame_rgb: np.ndarray,          # (H, W, 3) uint8
        detections_xywh: list,          # list of [x, y, w, h]
        confidences: list,              # list of raw confidence floats
    ) -> tuple[np.ndarray, list]:

        # ── Pre-process detections once (not inside the track loop) ──
        sig_confs = [_sigmoid(c) for c in confidences]
        raw       = [[d, sc, "0"] for d, sc in zip(detections_xywh, sig_confs)]

        # Vectorised arrays for fast IoU lookup
        if detections_xywh:
            det_arr   = np.array(detections_xywh, dtype=np.float32)   # (N,4)
            conf_arr  = np.array(sig_confs,        dtype=np.float32)   # (N,)
        else:
            det_arr  = np.empty((0, 4), dtype=np.float32)
            conf_arr = np.empty((0,),   dtype=np.float32)

        # ── Run DeepSORT ──────────────────────────────────────────────
        tracks = self.tracker.update_tracks(raw, frame=frame_rgb)

        conf_vels     = []
        spatial_jumps = []
        feat_dists    = []

        for t in tracks:
            if not t.is_confirmed():
                continue

            tid  = t.track_id
            tlwh = t.to_tlwh()

            # ── 1. Confidence velocity (vectorised IoU) ───────────────
            cur_conf = _best_iou_vectorised(tlwh, det_arr, conf_arr)
            if tid in self._prev_conf:
                conf_vels.append(cur_conf - self._prev_conf[tid])
            self._prev_conf[tid] = cur_conf

            # ── 2. Spatial jump ───────────────────────────────────────
            obs_cx = float(tlwh[0] + tlwh[2] / 2)
            obs_cy = float(tlwh[1] + tlwh[3] / 2)
            if tid in self._prev_cxcy:
                px, py = self._prev_cxcy[tid]
                spatial_jumps.append(
                    math.sqrt((obs_cx - px) ** 2 + (obs_cy - py) ** 2)
                )
            self._prev_cxcy[tid] = (obs_cx, obs_cy)

            # ── 3. Feature cosine distance ────────────────────────────
            if t.features and len(t.features) >= 2:
                e1 = np.asarray(t.features[-2], dtype=np.float32)
                e2 = np.asarray(t.features[-1], dtype=np.float32)
                denom = np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8
                feat_dists.append(float(1.0 - np.dot(e1, e2) / denom))

        state = np.array([
            float(np.min(conf_vels))     if conf_vels     else 0.0,
            float(np.max(spatial_jumps)) if spatial_jumps else 0.0,
            float(np.max(feat_dists))    if feat_dists    else 0.0,
        ], dtype=np.float32)

        active_ids = [t.track_id for t in tracks if t.is_confirmed()]
        return state, active_ids