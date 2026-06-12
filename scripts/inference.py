# scripts/inference.py
import sys, os
sys.path.insert(0, os.path.abspath("."))

import yaml
import numpy as np
from stable_baselines3 import PPO
from src.mot_env import MOT17Env

# ── 1. Load config ────────────────────────────────────────────────────────
cfg = yaml.safe_load(open("config.yaml"))
model_path = cfg["paths"]["model_save"]
if not model_path.endswith(".zip"):
    model_path += ".zip"

# ── 2. Initialize environment with NEW reward weights ─────────────────────
env = MOT17Env(
    seq_path=cfg["data"]["seq_path"],
    w_rec=cfg["reward"]["w_rec"],
    w_fp=cfg["reward"]["w_fp"],
    w_lost=cfg["reward"]["w_lost"],
    w_cost=cfg["reward"]["w_cost"],
)

# ── 3. Load Agent ─────────────────────────────────────────────────────────
print(f"[inference] Loading model from {model_path}...")
model = PPO.load(model_path)

obs, _ = env.reset()
total_switches = 0
action_counts  = {0: 0, 1: 0, 2: 0, 3: 0}

print("[inference] Running inference loop...")
while True:
    # deterministic=False allows EOT stochastic sampling
    action, _ = model.predict(obs, deterministic=False)
    action = int(action)
    
    obs, reward, done, _, info = env.step(action)

    total_switches += info.get("id_switches", 0)
    action_counts[action] += 1

    if done:
        break

# ── 4. Print Results ──────────────────────────────────────────────────────
total = sum(action_counts.values())
print(f"\nTotal ID switches:  {total_switches}")
print(f"Action distribution:")
for a, count in action_counts.items():
    label = ["T0 (Clean)", "T1 (Warp)", "T2 (Noise)", "T3 (Cutout)"][a]
    pct = (count / max(total, 1)) * 100
    print(f"  {label:<12}: {count} ({pct:.1f}%)")