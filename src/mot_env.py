# src/mot_env.py
import os
import threading
import queue
import numpy as np
import pandas as pd
import cv2
import torch
import torchvision.transforms.functional as TF
import gymnasium as gym

from src.state_extractor import TrackingStateExtractor
from src.transformations import apply_transformation
from src.reward import compute_reward
from src.device import DEVICE


class FramePrefetcher:
    """
    Loads frames from disk in a background thread and pushes them
    to GPU memory ahead of time. Eliminates the disk→CPU→GPU bottleneck.
    """
    def __init__(self, img_dir: str, frame_files: list, queue_size: int = 8):
        self._img_dir     = img_dir
        self._frame_files = frame_files
        self._q           = queue.Queue(maxsize=queue_size)
        self._stop        = threading.Event()
        self._thread      = None

    def start(self, start_idx: int = 0):
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._worker, args=(start_idx,), daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        # drain queue so worker isn't blocked on put()
        while not self._q.empty():
            try: self._q.get_nowait()
            except queue.Empty: break

    def get(self) -> np.ndarray:
        return self._q.get()

    def _worker(self, start_idx: int):
        for i in range(start_idx, len(self._frame_files)):
            if self._stop.is_set():
                break
            path = os.path.join(self._img_dir, self._frame_files[i])
            bgr  = cv2.imread(path)
            rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            # Move to GPU as a pinned tensor for fast transfer
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            if DEVICE.type == "cuda":
                tensor = tensor.to(DEVICE, non_blocking=True)
            self._q.put(tensor)
        self._q.put(None)   # sentinel


def gpu_apply_transformation(tensor: torch.Tensor, action: int) -> np.ndarray:
    """
    Runs T1/T2/T3 on GPU via torchvision, then returns numpy uint8.
    T0 skips GPU entirely (no copy needed).
    """
    if action == 0:
        # Fast path: go directly to numpy without GPU round-trip
        return (tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    elif action == 1:   # T1 — Spatial Warp (GPU perspective transform)
        _, h, w = tensor.shape
        angle   = float(np.random.uniform(-5, 5))
        shear   = float(np.random.uniform(-5, 5))
        out = TF.affine(
            tensor.cpu(),           # torchvision affine works on CPU tensors
            angle=angle,
            translate=[0, 0],
            scale=1.0,
            shear=shear,
        )
        return (out.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    elif action == 2:   # T2 — Gaussian Noise (pure GPU)
        noise = torch.randn_like(tensor) * (15.0 / 255.0)
        noisy = (tensor + noise).clamp(0.0, 1.0)
        return (noisy.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    elif action == 3:   # T3 — Block Cutout (GPU tensor slice)
        out = tensor.clone()
        _, h, w = out.shape
        bh, bw = h // 5, w // 5
        y0 = np.random.randint(0, h - bh)
        x0 = np.random.randint(0, w - bw)
        out[:, y0:y0+bh, x0:x0+bw] = 0.5
        return (out.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    else:
        raise ValueError(f"Unknown action {action}")


class MOT17Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, seq_path: str, w1=1.0, w2=5.0, w3=0.5):
        super().__init__()
        self.seq_path = seq_path
        self.w1, self.w2, self.w3 = w1, w2, w3

        det_file = os.path.join(seq_path, "det", "det.txt")
        cols = ["frame","id","x","y","w","h","conf","_1","_2","_3"]
        df = pd.read_csv(det_file, header=None, names=cols)
        self._det_map: dict[int, tuple] = {}
        for frame_no, group in df.groupby("frame"):
            xywh  = group[["x","y","w","h"]].values.tolist()
            confs = group["conf"].tolist()
            self._det_map[int(frame_no)] = (xywh, confs)

        self._img_dir     = os.path.join(seq_path, "img1")
        self._frame_files = sorted(os.listdir(self._img_dir))
        self._n_frames    = len(self._frame_files)

        self.observation_space = gym.spaces.Box(
            low=0.0, high=np.inf, shape=(3,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(4)

        self._extractor:   TrackingStateExtractor | None = None
        self._prefetcher:  FramePrefetcher        | None = None
        self._frame_idx:   int  = 0
        self._prev_id_set: set  = set()
        self._current_tensor: torch.Tensor | None = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Stop any previous prefetcher
        if self._prefetcher is not None:
            self._prefetcher.stop()

        self._frame_idx    = 0
        self._prev_id_set  = set()
        self._extractor    = TrackingStateExtractor()

        # Start background frame loading
        self._prefetcher = FramePrefetcher(
            self._img_dir, self._frame_files, queue_size=8
        )
        self._prefetcher.start(start_idx=0)

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

    def _run_frame(self, action: int):
        tensor = self._prefetcher.get()
        if tensor is None:
            return np.zeros(3, dtype=np.float32), []

        self._current_tensor = tensor
        frame_rgb = gpu_apply_transformation(tensor, action)

        frame_no = self._frame_idx + 1
        xywh, confs = self._det_map.get(frame_no, ([], []))

        state, active_ids = self._extractor.update(frame_rgb, xywh, confs)
        return state, active_ids