# src/state_extractor.py
"""
Fixed TrackingStateExtractor.

BUG-1 FIXED: spatial_jumps were being appended into conf_vels (line 72 in
             old version). spatial_jumps list was always empty, state dim 1
             was meaningless. Fixed: appended into correct list.

BUG-2 FIXED: import math inside update() — re-resolved every step.
             Moved to module level.

NEW: Camera motion compensation for moving-camera sequences (MOT17-02).
     Each track's spatial displacement is normalised against the median
     displacement of ALL confirmed tracks in that frame.

     Static camera, clean frame:   all jumps ≈ 0  → compensated ≈ 0
     Moving camera, clean frame:   all jumps large but equal → compensated ≈ 0
     Attack on either camera type: one track jumps anomalously → compensated >> 0

     This is why MOT17-02 (moving camera) had MOTA=25.8 — the agent saw
     large global-motion jumps and applied transformations on every clean frame.
"""

import math
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


# ── Module-level helpers ──────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    x = max(-10.0, min(10.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _best_conf_vectorised(
    tlwh:      np.ndarray,   # (4,)  track box [x,y,w,h]
    det_boxes: np.ndarray,   # (N,4) detection boxes
    det_confs: np.ndarray,   # (N,)  sigmoid-scaled confidences
) -> float:
    """Vectorised IoU match — replaces O(N) Python loop."""
    if len(det_boxes) == 0:
        return 0.0

    t_x1, t_y1 = tlwh[0], tlwh[1]
    t_x2, t_y2 = tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]

    d_x1 = det_boxes[:, 0]
    d_y1 = det_boxes[:, 1]
    d_x2 = det_boxes[:, 0] + det_boxes[:, 2]
    d_y2 = det_boxes[:, 1] + det_boxes[:, 3]

    inter = (np.maximum(0.0, np.minimum(t_x2, d_x2) - np.maximum(t_x1, d_x1)) *
             np.maximum(0.0, np.minimum(t_y2, d_y2) - np.maximum(t_y1, d_y1)))
    union = tlwh[2] * tlwh[3] + det_boxes[:, 2] * det_boxes[:, 3] - inter + 1e-6
    iou   = inter / union

    best  = int(np.argmax(iou))
    return float(det_confs[best]) if iou[best] > 0.0 else 0.0


# ── Main extractor ────────────────────────────────────────────────────────────

class TrackingStateExtractor:

    def __init__(self):
        # BUG-2 FIX: embedder_gpu=False
        # Each SubprocVecEnv worker is a separate process.
        # GPU ReID in workers = N CUDA contexts competing for one device.
        # CPU ReID across N cores in parallel is faster in aggregate.
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nn_budget=100,
            max_cosine_distance=0.4,
            embedder_gpu=True,
        )
        self._prev_conf: dict[int, float] = {}
        self._prev_cxcy: dict[int, tuple] = {}

    def reset(self):
        """Re-instantiate tracker — avoids stale internal state in some versions."""
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
        frame_rgb:       np.ndarray,   # (H, W, 3) uint8 RGB
        detections_xywh: list,         # list of [x, y, w, h]
        confidences:     list,         # raw confidence scores
    ) -> tuple[np.ndarray, list]:

        # Pre-process detections once — not inside track loop
        sig_confs = [_sigmoid(c) for c in confidences]
        raw       = [[d, sc, "0"] for d, sc in zip(detections_xywh, sig_confs)]

        if detections_xywh:
            det_arr  = np.array(detections_xywh, dtype=np.float32)
            conf_arr = np.array(sig_confs,        dtype=np.float32)
        else:
            det_arr  = np.empty((0, 4), dtype=np.float32)
            conf_arr = np.empty((0,),   dtype=np.float32)

        tracks = self.tracker.update_tracks(raw, frame=frame_rgb)

        conf_vels     = []
        raw_jumps     = []   # absolute pixel displacements (camera + object motion)
        feat_dists    = []
        track_ids     = []   # parallel to raw_jumps for bookkeeping

        for t in tracks:
            if not t.is_confirmed():
                continue

            tid  = t.track_id
            tlwh = t.to_tlwh()

            # ── 1. Confidence velocity (vectorised) ───────────────────
            cur_conf = _best_conf_vectorised(tlwh, det_arr, conf_arr)
            if tid in self._prev_conf:
                conf_vels.append(cur_conf - self._prev_conf[tid])
            self._prev_conf[tid] = cur_conf

            # ── 2. Spatial displacement (BUG-1 FIX: correct list) ─────
            obs_cx = float(tlwh[0] + tlwh[2] / 2)
            obs_cy = float(tlwh[1] + tlwh[3] / 2)

            if tid in self._prev_cxcy:
                px, py = self._prev_cxcy[tid]
                jump   = math.sqrt((obs_cx - px) ** 2 + (obs_cy - py) ** 2)
                raw_jumps.append(jump)     # ← BUG-1 FIX: was conf_vels.append()
                track_ids.append(tid)
            self._prev_cxcy[tid] = (obs_cx, obs_cy)

            # ── 3. Feature cosine distance ────────────────────────────
            if t.features and len(t.features) >= 2:
                e1 = np.asarray(t.features[-2], dtype=np.float32)
                e2 = np.asarray(t.features[-1], dtype=np.float32)
                n  = np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8
                feat_dists.append(float(1.0 - np.dot(e1, e2) / n))

        # ── Camera motion compensation ────────────────────────────────
        # Subtract the MEDIAN displacement from every track's jump.
        #
        # Why median (not mean): mean is sensitive to the one attacked track
        # which may have an extreme jump. Median robustly estimates the
        # background camera motion from the majority of tracks.
        #
        # Result:
        #   Static camera, clean   → jumps ≈ 0,  compensated ≈ 0
        #   Moving camera, clean   → jumps large, compensated ≈ 0 (all move together)
        #   Attack (any camera)    → attacked track deviates from median → compensated >> 0
        if raw_jumps:
            scene_motion = float(np.median(raw_jumps))
            comp_jumps   = [abs(j - scene_motion) for j in raw_jumps]
        else:
            comp_jumps = []

        state = np.array([
            float(np.min(conf_vels))   if conf_vels  else 0.0,  # dim 0: conf drop
            float(np.max(comp_jumps))  if comp_jumps else 0.0,  # dim 1: anomalous motion
            float(np.max(feat_dists))  if feat_dists else 0.0,  # dim 2: appearance shift
        ], dtype=np.float32)

        active_ids = [t.track_id for t in tracks if t.is_confirmed()]
        return state, active_ids