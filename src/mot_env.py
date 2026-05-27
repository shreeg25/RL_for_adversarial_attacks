# mot_env.py
import gymnasium as gym
import numpy as np
import cv2
import os
import pandas as pd

class MOT17Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Action cost weights for C(A_t)
    ACTION_COST = {0: 0.0, 1: 0.05, 2: 0.03, 3: 0.04}

    def __init__(self, seq_path, w1=1.0, w2=5.0, w3=0.5):
        super().__init__()
        self.seq_path = seq_path
        self.w1, self.w2, self.w3 = w1, w2, w3

        # Load pre-computed FRCNN detections
        det_path = os.path.join(seq_path, "det", "det.txt")
        cols = ["frame","id","x","y","w","h","conf","_","__","___"]
        df = pd.read_csv(det_path, header=None, names=cols)
        self.detections = {int(f): g for f, g in df.groupby("frame")}

        self.img_dir = os.path.join(seq_path, "img1")
        self.frames  = sorted(os.listdir(self.img_dir))
        self.n_frames = len(self.frames)

        self.observation_space = gym.spaces.Box(
            low=0.0, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)

        self.extractor = TrackingStateExtractor()   # from milestone 2
        self._reset_internals()

    def _reset_internals(self):
        self.frame_idx    = 0
        self.prev_iou_map = {}    # track_id → last IoU (bbox overlap t-1)
        self.prev_id_set  = set()
        self.extractor    = TrackingStateExtractor()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_internals()
        obs, _ = self._get_obs_and_info()
        return obs, {}

    def step(self, action: int):
        # 1. Apply transformation
        frame_rgb = self._load_frame(self.frame_idx)
        transformed = self._apply_action(frame_rgb, action)

        # 2. Get detections for this frame
        dets, confs = self._get_detections(self.frame_idx + 1)  # 1-indexed

        # 3. Run tracker → extract state
        state, active_ids = self.extractor.update(transformed, dets, confs)

        # 4. Reward calculation
        iou_reward   = self._compute_iou_reward(active_ids)
        id_switch    = self._count_id_switches(active_ids)
        action_cost  = self.ACTION_COST[action]
        reward = (self.w1 * iou_reward
                - self.w2 * id_switch
                - self.w3 * action_cost)

        self.prev_id_set = set(active_ids)
        self.frame_idx  += 1
        done = self.frame_idx >= self.n_frames

        return state, reward, done, False, {"id_switches": id_switch}

    # ──────────────────────────────────────────────────────────────
    def _load_frame(self, idx):
        path = os.path.join(self.img_dir, self.frames[idx])
        bgr  = cv2.imread(path)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _get_detections(self, frame_no):
        if frame_no not in self.detections:
            return [], []
        df = self.detections[frame_no]
        dets  = df[["x","y","w","h"]].values.tolist()
        confs = df["conf"].tolist()
        return dets, confs

    def _apply_action(self, frame, action):
        if action == 0:   # T0: clean pass
            return frame
        elif action == 1: # T1: spatial warp
            h, w = frame.shape[:2]
            pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
            jitter = np.random.uniform(-15, 15, pts1.shape).astype(np.float32)
            pts2 = pts1 + jitter
            M = cv2.getPerspectiveTransform(pts1, pts2)
            return cv2.warpPerspective(frame, M, (w, h))
        elif action == 2: # T2: Gaussian noise
            noise = np.random.normal(0, 15, frame.shape).astype(np.int16)
            return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif action == 3: # T3: block cutout
            out = frame.copy()
            h, w = frame.shape[:2]
            bh, bw = h//5, w//5
            y0 = np.random.randint(0, h - bh)
            x0 = np.random.randint(0, w - bw)
            out[y0:y0+bh, x0:x0+bw] = 128  # gray patch
            return out

    def _compute_iou_reward(self, active_ids):
        # Simplified: reward 1.0 if any tracks survived, 0 otherwise
        # Replace with actual bbox IoU overlap if you store prev bboxes
        return 1.0 if active_ids else 0.0

    def _count_id_switches(self, active_ids):
        current = set(active_ids)
        # New IDs that weren't present last frame = switches
        new_ids = current - self.prev_id_set
        return len(new_ids) 