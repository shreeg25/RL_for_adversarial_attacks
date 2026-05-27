# src/state_extractor.py
"""
Wraps DeepSORT and extracts the 3-dimensional state vector S_t:
  [conf_velocity, kf_residual, feature_distance]

All pixel data is kept OUTSIDE the returned vector (State-Space Isolation).
"""
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


class TrackingStateExtractor:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nn_budget=100,
            max_cosine_distance=0.4,
        )
        # Per-track history (keyed by track_id)
        self._prev_conf: dict[int, float] = {}
        self._prev_feats: dict[int, np.ndarray] = {}
        self._prev_tlwh: dict[int, np.ndarray] = {}

    def reset(self):
        """Call at the start of every episode."""
        self.tracker = DeepSort(max_age=30, n_init=3, nn_budget=100,
                                max_cosine_distance=0.4)
        self._prev_conf.clear()
        self._prev_feats.clear()
        self._prev_tlwh.clear()

    def update(
        self,
        frame_rgb: np.ndarray,
        detections_xywh: list,
        confidences: list,
    ) -> tuple[np.ndarray, list[int]]:
        """
        Args:
            frame_rgb:        H×W×3 uint8 numpy array
            detections_xywh:  list of [x, y, w, h] floats
            confidences:      list of float detection scores

        Returns:
            state_vec:    np.float32 array of shape (3,)
            active_ids:   list of confirmed track IDs this frame
        """
        raw = [[d, c, None] for d, c in zip(detections_xywh, confidences)]
        tracks = self.tracker.update_tracks(raw, frame=frame_rgb)

        conf_vels, kf_residuals, feat_dists = [], [], []

        for t in tracks:
            if not t.is_confirmed():
                continue
            tid = t.track_id

            # ── 1. Conf-Velocity ──────────────────────────────────────
            cur_conf = float(t.det_conf) if t.det_conf is not None else 0.0
            if tid in self._prev_conf:
                conf_vels.append(abs(cur_conf - self._prev_conf[tid]))
            self._prev_conf[tid] = cur_conf

            # ── 2. KF-Residual ────────────────────────────────────────
            # t.mean[:4] is [cx, cy, aspect_ratio, height] in DeepSORT state
            kf_pred_cx = float(t.mean[0])
            kf_pred_cy = float(t.mean[1])
            tlwh = t.to_tlwh()
            obs_cx = float(tlwh[0] + tlwh[2] / 2)
            obs_cy = float(tlwh[1] + tlwh[3] / 2)
            residual = np.sqrt((kf_pred_cx - obs_cx) ** 2 +
                                (kf_pred_cy - obs_cy) ** 2)
            kf_residuals.append(residual)

            # ── 3. Feature Distance ───────────────────────────────────
            if t.features and len(t.features) >= 2:
                e1 = np.array(t.features[-2], dtype=np.float32)
                e2 = np.array(t.features[-1], dtype=np.float32)
                denom = np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8
                cos_dist = float(1.0 - np.dot(e1, e2) / denom)
                feat_dists.append(cos_dist)

        state = np.array([
            np.mean(conf_vels)    if conf_vels    else 0.0,
            np.mean(kf_residuals) if kf_residuals else 0.0,
            np.mean(feat_dists)   if feat_dists   else 0.0,
        ], dtype=np.float32)

        active_ids = [t.track_id for t in tracks if t.is_confirmed()]
        return state, active_ids