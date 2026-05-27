# src/state_extractor.py
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from src.device import DEVICE


class TrackingStateExtractor:
    def __init__(self):
        # Tell DeepSORT to run its MobileNet ReID on GPU
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nn_budget=100,
            max_cosine_distance=0.4,
            embedder_gpu=DEVICE.type == "cuda",   # ← GPU flag
        )
        self._prev_conf:  dict[int, float]      = {}
        self._prev_feats: dict[int, np.ndarray] = {}
        self._prev_tlwh:  dict[int, np.ndarray] = {}

    def reset(self):
        self.tracker = DeepSort(
            max_age=30, n_init=3, nn_budget=100,
            max_cosine_distance=0.4,
            embedder_gpu=DEVICE.type == "cuda",
        )
        self._prev_conf.clear()
        self._prev_feats.clear()
        self._prev_tlwh.clear()

    def update(
        self,
        frame_rgb: np.ndarray,
        detections_xywh: list,
        confidences: list,
    ) -> tuple[np.ndarray, list[int]]:
        raw = [[d, c, None] for d, c in zip(detections_xywh, confidences)]
        tracks = self.tracker.update_tracks(raw, frame=frame_rgb)

        conf_vels, kf_residuals, feat_dists = [], [], []

        for t in tracks:
            if not t.is_confirmed():
                continue
            tid = t.track_id

            cur_conf = float(t.det_conf) if t.det_conf is not None else 0.0
            if tid in self._prev_conf:
                conf_vels.append(abs(cur_conf - self._prev_conf[tid]))
            self._prev_conf[tid] = cur_conf

            kf_pred_cx = float(t.mean[0])
            kf_pred_cy = float(t.mean[1])
            tlwh = t.to_tlwh()
            obs_cx = float(tlwh[0] + tlwh[2] / 2)
            obs_cy = float(tlwh[1] + tlwh[3] / 2)
            kf_residuals.append(
                np.sqrt((kf_pred_cx - obs_cx)**2 + (kf_pred_cy - obs_cy)**2)
            )

            if t.features and len(t.features) >= 2:
                e1 = np.array(t.features[-2], dtype=np.float32)
                e2 = np.array(t.features[-1], dtype=np.float32)
                denom = np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8
                feat_dists.append(float(1.0 - np.dot(e1, e2) / denom))

        state = np.array([
            np.mean(conf_vels)    if conf_vels    else 0.0,
            np.mean(kf_residuals) if kf_residuals else 0.0,
            np.mean(feat_dists)   if feat_dists   else 0.0,
        ], dtype=np.float32)

        active_ids = [t.track_id for t in tracks if t.is_confirmed()]
        return state, active_ids