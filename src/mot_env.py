# src/mot_env.py  — DETECTOR-IN-LOOP + COUNTERFACTUAL REWARD
"""
This supersedes the earlier mot_env_FIXED.py. It does two things:

1. DETECTOR IN THE LOOP (the core fix)
   The agent's transformation is re-detected, so the defense can actually
   repair suppressed detections. See _run_frame.

2. COUNTERFACTUAL REWARD SUPPORT
   Every step we run the detector on BOTH:
     - the untransformed frame (T0 baseline — what no-defense would see)
     - the agent's chosen frame (what the defense achieved)
   and pass both, plus GT, to the attack-aware reward in reward.py.
   When action==0 the two are identical, so only ONE detection pass happens.

   Cost: up to 2 Faster R-CNN passes per frame during training (1 when the
   agent picks T0). This is the price of a defense that can actually act.
   On an RTX 6000 Ada, expect training to take ~a day for 150k timesteps with
   3 sequences. Mitigations if too slow: lower total_timesteps, subsample
   frames, or swap to fasterrcnn_mobilenet_v3_large_fpn for training.

GT loading: filtered identically to evaluate_accuracy.py
   (active==1, class==1, visibility>=0.25) so training and eval agree.
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
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

from src.state_extractor import TrackingStateExtractor
from src.transformations import apply_transformation
from src.reward import compute_reward
from src.transformations import ACTION_COST
from src.device import DEVICE


# ── Frame prefetcher ──────────────────────────────────────────────────────────

class FramePrefetcher:
    def __init__(self, img_dir: str, frame_files: list, queue_size: int = 4):
        self._img_dir     = img_dir
        self._frame_files = frame_files
        self._q           = queue.Queue(maxsize=queue_size)
        self._stop        = threading.Event()
        self._thread      = None

    def start(self, start_idx: int = 0, stagger_ms: int = 0):
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._worker, args=(start_idx, stagger_ms), daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def get(self):
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
            self._q.put(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        self._q.put(None)


# ── Lazy per-process detector ──────────────────────────────────────────────────

_DETECTOR = None


from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import sys
import torch

def _get_detector():
    global _DETECTOR
    if _DETECTOR is None:
        # 1. Initialize the raw architecture
        m = fasterrcnn_resnet50_fpn(weights=None)
        
        # 2. Amputate the generic 91-class head and replace it with a 2-class head
        in_features = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
        
        # 3. Securely load the domain-specific weights
        weight_path = "weights/faster_rcnn_mot17.pth"
        try:
            # SECURITY FIX: Enforce weights_only=True
            state_dict = torch.load(weight_path, map_location=DEVICE, weights_only=True)
            m.load_state_dict(state_dict)
            print(f"[env] Successfully loaded MOT17 domain weights from {weight_path}")
        except FileNotFoundError:
            print(f"\n[FATAL] Domain gap fix failed. Could not find {weight_path}.")
            print("You must run scripts/finetune_detector.py first to generate this file.\n")
            sys.exit(1)
            
        m.eval().to(DEVICE)
        _DETECTOR = m
    return _DETECTOR


# ── Environment ────────────────────────────────────────────────────────────────

class MOT17Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, seq_path: str,
                 w_rec: float = 5.0, w_fp: float = 2.0,
                 w_lost: float = 0.5, w_cost: float = 1.0,
                 score_thresh: float = 0.4, person_label: int = 1):
        super().__init__()
        self.seq_path = seq_path
        self.w_rec, self.w_fp, self.w_lost, self.w_cost = w_rec, w_fp, w_lost, w_cost
        self.score_thresh = score_thresh
        self.person_label = person_label

        self._img_dir     = os.path.join(seq_path, "img1")
        self._frame_files = sorted(os.listdir(self._img_dir))
        self._n_frames    = len(self._frame_files)

        self._gt = self._load_gt(seq_path)

        self.observation_space = gym.spaces.Box(
    low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
)
        self.action_space = gym.spaces.Discrete(4)

        self._extractor   = None
        self._prefetcher  = None
        self._frame_idx   = 0
        self._prev_id_set = set()
        self._stagger_ms  = random.randint(0, 200)

    # ── GT loading (matches evaluate_accuracy.py filtering) ─────────────────────

    @staticmethod
    def _load_gt(seq_path: str) -> dict:
        gt_file = os.path.join(seq_path, "gt", "gt.txt")
        if not os.path.exists(gt_file):
            return {}
        cols = ["frame", "id", "x", "y", "w", "h", "active", "class", "visibility"]
        df = pd.read_csv(gt_file, header=None, names=cols)
        df = df[(df["active"] == 1) & (df["class"] == 1) & (df["visibility"] >= 0.25)]
        gt = {}
        for frame_no, grp in df.groupby("frame"):
            gt[int(frame_no)] = grp[["x", "y", "w", "h"]].values.tolist()
        return gt

    # ── Detection ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _detect(self, frame_rgb: np.ndarray):
        model  = _get_detector()
        tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div_(255.0).to(DEVICE)
        
        # Force FP16 inference to accelerate Tensor Cores
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
            preds  = model([tensor])[0]
            
        labels, scores, boxes = preds["labels"], preds["scores"], preds["boxes"]
        keep = (labels == self.person_label) & (scores > self.score_thresh)
        boxes  = boxes[keep].cpu().numpy()
        scores = scores[keep].cpu().numpy()
        xywh  = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                 for (x1, y1, x2, y2) in boxes]
        confs = [float(s) for s in scores]
        return xywh, confs

    # ── Gym API ──────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._prefetcher is not None:
            self._prefetcher.stop()
        self._frame_idx   = 0
        self._prev_id_set = set()
        self._last_target_box = None  # Reset targeting memory
        self._extractor   = TrackingStateExtractor()
        self._prefetcher  = FramePrefetcher(self._img_dir, self._frame_files, queue_size=4)
        self._prefetcher.start(start_idx=0, stagger_ms=self._stagger_ms)
        obs, _, _ = self._run_frame(action=0)
        return obs, {}

    def step(self, action: int):
        obs, active_ids, rew_payload = self._run_frame(action)

        if rew_payload is None:
            reward, info = 0.0, {"recovery": 0.0}
        else:
            det_a, conf_a, det_0, conf_0, gt_boxes = rew_payload
            reward, info = compute_reward(
                det_action=det_a, conf_action=conf_a,
                det_t0=det_0,     conf_t0=conf_0,
                gt_boxes=gt_boxes, action=action,
                prev_id_set=self._prev_id_set, current_ids=active_ids,
                frame_idx=self._frame_idx, action_cost_table=ACTION_COST,
                w_rec=self.w_rec, w_fp=self.w_fp, w_lost=self.w_lost, w_cost=self.w_cost,
            )

        self._prev_id_set = set(active_ids)
        self._frame_idx  += 1
        done = self._frame_idx >= self._n_frames
        info["frame"]       = self._frame_idx
        info["id_switches"] = info.get("lost", 0) 
        return obs, reward, done, False, info

    def close(self):
        if self._prefetcher is not None:
            self._prefetcher.stop()

    def render(self):
        return None

    def _run_frame(self, action: int):
        frame_rgb = self._prefetcher.get()
        if frame_rgb is None:
            return np.zeros(12, dtype=np.float32), [], None

        frame_no = self._frame_idx + 1
        gt_boxes = self._gt.get(frame_no, [])

        # T0 baseline detection
        det_t0, conf_t0 = self._detect(frame_rgb)

        # Agent's action, utilizing the localized target box
        if action == 0 or self._last_target_box is None:
            transformed     = frame_rgb
            det_a, conf_a   = det_t0, conf_t0          
        else:
            transformed     = apply_transformation(frame_rgb, action, self._last_target_box)
            det_a, conf_a   = self._detect(transformed)  

        # Tracker updates and outputs the NEXT vulnerable box
        state, active_ids, target_box = self._extractor.update(transformed, det_a, conf_a)
        self._last_target_box = target_box

        payload = (det_a, conf_a, det_t0, conf_t0, gt_boxes)
        return state, active_ids, payload