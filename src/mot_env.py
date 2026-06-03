# src/mot_env.py
"""
Fixed MOT17 Gymnasium environment.

FIX-1  FramePrefetcher stores frames as CPU numpy arrays.
        Old version pushed to GPU then pulled back to CPU for every action
        (T0/T1/T3 all called .cpu() explicitly). Pure waste of PCIe bandwidth.
        DeepSORT needs numpy anyway — frames never need GPU during training.

FIX-2  apply_transformation() called on numpy directly.
        Removes gpu_apply_transformation and its redundant tensor conversions.

FIX-3  queue_size reduced from 8 to 4 per env.
        With 4 envs × 8 queue = 32 frames buffered = ~100 MB RAM per env.
        4 frames ahead is sufficient for sequential access.

FIX-4  Staggered prefetcher startup per env instance.
        Prevents all workers hitting disk at frame 0 simultaneously on reset.
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
    Background thread — reads frames from disk as CPU numpy uint8 arrays.
    No GPU transfers. DeepSORT (CPU) consumes numpy directly.
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
            self._q.put(rgb)       # numpy uint8 (H, W, 3)

        self._q.put(None)          # sentinel


# ── Gymnasium environment ─────────────────────────────────────────────────────

class MOT17Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, seq_path: str, w1: float = 1.0,
                 w2: float = 5.0, w3: float = 0.5):
        super().__init__()
        self.seq_path      = seq_path
        self.w1, self.w2, self.w3 = w1, w2, w3

        # Load pre-computed detections
        det_file = os.path.join(seq_path, "det", "det.txt")
        cols = ["frame", "id", "x", "y", "w", "h", "conf", "_1", "_2", "_3"]
        df   = pd.read_csv(det_file, header=None, names=cols)

        self._det_map: dict[int, tuple] = {}
        for frame_no, group in df.groupby("frame"):
            xywh  = group[["x", "y", "w", "h"]].values.tolist()
            confs = group["conf"].tolist()
            self._det_map[int(frame_no)] = (xywh, confs)

        self._img_dir     = os.path.join(seq_path, "img1")
        self._frame_files = sorted(os.listdir(self._img_dir))
        self._n_frames    = len(self._frame_files)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(4)

        self._extractor:   TrackingStateExtractor | None = None
        self._prefetcher:  FramePrefetcher        | None = None
        self._frame_idx:   int  = 0
        self._prev_id_set: set  = set()

        # Each env instance gets a fixed random stagger offset
        # so 4 workers don't all hit disk at frame 0 simultaneously
        self._stagger_ms: int = random.randint(0, 300)

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

        obs, _ = self._run_frame(action=0)
        return obs, {}

    def step(self, action: int):
        obs, active_ids = self._run_frame(action)

        reward, id_switches = compute_reward(
            prev_id_set=self._prev_id_set,
            current_ids=active_ids,
            action=action,
            w1=self.w1, w2=self.w2, w3=self.w3,
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

    def _run_frame(self, action: int) -> tuple[np.ndarray, list]:
        frame_rgb = self._prefetcher.get()
        if frame_rgb is None:
            return np.zeros(3, dtype=np.float32), []

        # Transformation on CPU numpy — no GPU round-trip needed
        transformed      = apply_transformation(frame_rgb, action)
        frame_no         = self._frame_idx + 1
        xywh, confs      = self._det_map.get(frame_no, ([], []))
        state, active_ids = self._extractor.update(transformed, xywh, confs)

        return state, active_ids