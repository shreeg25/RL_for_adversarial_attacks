import math
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


def _sigmoid(x: float) -> float:
    x = max(-10.0, min(10.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _best_conf_vectorised(
    tlwh: np.ndarray,
    det_boxes: np.ndarray,
    det_confs: np.ndarray,
) -> float:
    if len(det_boxes) == 0:
        return 0.0

    t_x1, t_y1 = tlwh[0], tlwh[1]
    t_x2, t_y2 = tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]

    d_x1 = det_boxes[:, 0]
    d_y1 = det_boxes[:, 1]
    d_x2 = det_boxes[:, 0] + det_boxes[:, 2]
    d_y2 = det_boxes[:, 1] + det_boxes[:, 3]

    inter = (
        np.maximum(0.0, np.minimum(t_x2, d_x2) - np.maximum(t_x1, d_x1))
        * np.maximum(0.0, np.minimum(t_y2, d_y2) - np.maximum(t_y1, d_y1))
    )
    union = tlwh[2] * tlwh[3] + det_boxes[:, 2] * det_boxes[:, 3] - inter + 1e-6
    iou = inter / union

    best = int(np.argmax(iou))
    return float(det_confs[best]) if iou[best] > 0.0 else 0.0


class TrackingStateExtractor:
    def __init__(self):
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
        frame_rgb: np.ndarray,
        detections_xywh: list,
        confidences: list,
    ) -> tuple[np.ndarray, list]:
        sig_confs = [_sigmoid(c) for c in confidences]
        raw = [[d, sc, "0"] for d, sc in zip(detections_xywh, sig_confs)]

        if detections_xywh:
            det_arr = np.array(detections_xywh, dtype=np.float32)
            conf_arr = np.array(sig_confs, dtype=np.float32)
        else:
            det_arr = np.empty((0, 4), dtype=np.float32)
            conf_arr = np.empty((0,), dtype=np.float32)

        tracks = self.tracker.update_tracks(raw, frame=frame_rgb)

        conf_vels = []
        raw_jumps = []
        feat_dists = []

        for t in tracks:
            if not t.is_confirmed():
                continue

            tid = t.track_id
            tlwh = t.to_tlwh()

            cur_conf = _best_conf_vectorised(tlwh, det_arr, conf_arr)
            if tid in self._prev_conf:
                conf_vels.append(cur_conf - self._prev_conf[tid])
            self._prev_conf[tid] = cur_conf

            obs_cx = float(tlwh[0] + tlwh[2] / 2)
            obs_cy = float(tlwh[1] + tlwh[3] / 2)
            if tid in self._prev_cxcy:
                px, py = self._prev_cxcy[tid]
                jump = math.sqrt((obs_cx - px) ** 2 + (obs_cy - py) ** 2)
                raw_jumps.append(jump)
            self._prev_cxcy[tid] = (obs_cx, obs_cy)

            if t.features and len(t.features) >= 2:
                e1 = np.asarray(t.features[-2], dtype=np.float32)
                e2 = np.asarray(t.features[-1], dtype=np.float32)
                n = np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8
                feat_dists.append(float(1.0 - np.dot(e1, e2) / n))

        if raw_jumps:
            scene_motion = float(np.median(raw_jumps))
            comp_jumps = [abs(j - scene_motion) for j in raw_jumps]
        else:
            comp_jumps = []

        state = np.array(
            [
                float(np.min(conf_vels)) if conf_vels else 0.0,
                float(np.max(comp_jumps)) if comp_jumps else 0.0,
                float(np.max(feat_dists)) if feat_dists else 0.0,
            ],
            dtype=np.float32,
        )

        active_ids = [t.track_id for t in tracks if t.is_confirmed()]
        return state, active_ids