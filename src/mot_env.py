# src/mot_env.py
"""
Optimised MOT17 Gymnasium environment.

FIX-2  apply_transformation() called directly on numpy array — removes the
        gpu_apply_transformation wrapper and its redundant tensor↔numpy conversions.
        T2 (Gaussian noise) is now done in numpy — faster than GPU round-trip
        for a single 720×576 frame.

FIX-3  FramePrefetcher queue_size reduced to 4 — with 4 envs each having a
        prefetch queue of 8, you had 32 frames buffered = ~100MB RAM per env.
        Queue of 4 keeps prefetch ahead by 4 frames, plenty for sequential access.

FIX-4  Staggered prefetcher startup — prevents all 4 workers hitting disk
        simultaneously at frame 0 on episode reset.
"""

import os
import time
import random
import threading
import queue
import numpy as np
import pandas as pd
import cv2
import gymnasium as gym

from src.state_extractor import TrackingStateExtractor
from src.transformations import apply_transformation
from src.reward import compute_reward


# ── Frame prefetcher (CPU only) ───────────────────────────────────────────────

class FramePrefetcher:
    """
    Background thread that reads frames from disk and queues them as
    numpy uint8 arrays (H,W,3) RGB.

    CPU-only — no GPU transfers. DeepSORT needs numpy anyway.
    """

    def __init__(self, img_dir: str, frame_files: list, queue_size: int = 4):
        self._img_dir     = img_dir
        self._frame_files = frame_files
        self._q           = queue.Queue(maxsize=queue_size)
        self._stop        = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, start_idx: int = 0, stagger_ms: int = 0):
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._worker,
            args=(start_idx, stagger_ms),
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def get(self) -> np.ndarray | None:
        return self._q.get()

    def _worker(self, start_idx: int, stagger_ms: int):
        if stagger_ms > 0:
            time.sleep(stagger_ms / 1000.0)

        for i in range(start_idx, len(self._frame_files)):
            if self._stop.is_set():
                break
            path = os.path.join(self._img_dir, self._frame_files[i])
            bgr  = cv2.imread(path)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self._q.put(rgb)          # numpy uint8 (H,W,3)

        self._q.put(None)             # sentinel


# ── Gymnasium environment ─────────────────────────────────────────────────────

class MOT17Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, seq_path: str, w1: float = 2.0,
                 w2: float = 1.5, w3: float = 0.01, w4: float = 3.0):
        super().__init__()
        self.seq_path = seq_path
        self.w1, self.w2, self.w3, self.w4 = w1, w2, w3, w4

        # ── Load pre-computed detections ──────────────────────────────
        det_file = os.path.join(seq_path, "det", "det.txt")
        cols = ["frame", "id", "x", "y", "w", "h", "conf", "_1", "_2", "_3"]
        df   = pd.read_csv(det_file, header=None, names=cols)

        self._det_map: dict[int, tuple] = {}
        for frame_no, group in df.groupby("frame"):
            xywh  = group[["x", "y", "w", "h"]].values.tolist()
            confs = group["conf"].tolist()
            self._det_map[int(frame_no)] = (xywh, confs)

        # ── Frame list ────────────────────────────────────────────────
        self._img_dir     = os.path.join(seq_path, "img1")
        self._frame_files = sorted(os.listdir(self._img_dir))
        self._n_frames    = len(self._frame_files)

        # ── Spaces ────────────────────────────────────────────────────
        self.observation_space = gym.spaces.Box(
            low=0.0, high=np.inf, shape=(7,), dtype=np.float32   # <-- now 7‑dim
        )
        self.action_space = gym.spaces.Discrete(4)

        # ── Internal state ────────────────────────────────────────────
        self._extractor:   TrackingStateExtractor | None = None
        self._prefetcher:  FramePrefetcher        | None = None
        self._frame_idx:   int  = 0
        self._prev_id_set: set  = set()

        # Stagger offset assigned once per env instance
        # prevents all workers hitting disk at the same moment on reset
        self._stagger_ms: int = random.randint(0, 200)

        # Load attack probability from the global config (will be set later by train.py)
        # Default to 0.0 if not present – train.py will overwrite the attribute.
        self.attack_prob: float = 0.0

    # optional setter used by train.py for curriculum learning
    def set_attack_prob(self, prob: float):
        self.attack_prob = float(prob)

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self._prefetcher is not None:
            self._prefetcher.stop()

        self._frame_idx   = 0
        self._prev_id_set = set()
        self._extractor   = TrackingStateExtractor()

        self._prefetcher = FramePrefetcher(
            self._img_dir, self._frame_files, queue_size=4
        )
        self._prefetcher.start(start_idx=0, stagger_ms=self._stagger_ms)

        obs, _, _ = self._run_frame(action=0)
        return obs, {}

    def step(self, action: int):
        obs, active_ids, frame_rgb = self._run_frame(action)   # <-- also return raw frame

        reward, id_switches = compute_reward(
            prev_id_set=self._prev_id_set,
            current_ids=active_ids,
            action=action,
            frame=frame_rgb,                     # <-- pass raw frame for gating bonus
            w1=self.w1, w2=self.w2, w3=self.w3, w4=self.w4,
            w0=self._cfg.get("reward", {}).get("w0", 1.0) if hasattr(self, "_cfg") else 1.0,
        )

        self._prev_id_set = set(active_ids)
        self._frame_idx  += 1
        done = self._frame_idx >= self._n_frames

        return obs, reward, done, False, {
            "id_switches": id_switches,
            "frame":       self._frame_idx,
        }

    def close(self):
        if self._prefetcher is not None:
            self._prefetcher.stop()

    def render(self):
        return None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _maybe_attack(self, frame: np.ndarray) -> np.ndarray:
        """
        With probability self.attack_prob, apply a strong, fixed adversarial warp
        (here we reuse the spatial‑warp transformation with a larger jitter).
        """
        if np.random.rand() < self.attack_prob:
            # Re‑use spatial warp but increase jitter range to make it a stronger attack
            h, w = frame.shape[:2]
            src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            jitter = np.random.uniform(-30, 30, src.shape).astype(np.float32)   # stronger jitter
            dst = np.clip(src + jitter, 0, [w, h]).astype(np.float32)
            M = cv2.getPerspectiveTransform(src, dst)
            return cv2.warpPerspective(frame, M, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT)
        return frame

    def _run_frame(self, action: int) -> tuple[np.ndarray, list, np.ndarray]:
        """
        Returns:
            obs        – the 7‑dim observation from the state extractor
            active_ids – list of confirmed track IDs after processing this frame
            frame_rgb  – the raw (unaltered) RGB frame as np.uint8, needed for reward gating bonus
        """
        frame_rgb = self._prefetcher.get()
        if frame_rgb is None:
            return np.zeros(7, dtype=np.float32), [], np.zeros((0,0,3), dtype=np.uint8)

        # Agent‑chosen transformation (T0‑T3)
        transformed = apply_transformation(frame_rgb, action)

        # Optional on‑the‑fly adversarial perturbation (independent of agent action)
        transformed = self._maybe_attack(transformed)

        frame_no         = self._frame_idx + 1
        xywh, confs      = self._det_map.get(frame_no, ([], []))
        state, active_ids = self._extractor.update(transformed, xywh, confs)

        return state, active_ids, frame_rgb