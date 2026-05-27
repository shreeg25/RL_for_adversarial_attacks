# scripts/verify_tracker.py
"""
Smoke test: run DeepSORT on raw MOT17-04 frames (no transformations),
dump the state vector per frame to CSV, and print basic statistics.

Expected output: all 3 state dims should have non-zero std.
If feat_dist is always 0 → DeepSORT's ReID weights aren't loading.
If kf_residual is always 0 → Kalman predict == observe (impossible).
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import yaml
from src.mot_env import MOT17Env

cfg = yaml.safe_load(open("config.yaml"))
env = MOT17Env(cfg["data"]["seq_path"])

obs, _ = env.reset()
rows = []

done = False
step = 0
while not done:
    obs, reward, done, _, info = env.step(0)   # T0 = clean pass
    rows.append({
        "frame":        info["frame"],
        "conf_vel":     float(obs[0]),
        "kf_residual":  float(obs[1]),
        "feat_dist":    float(obs[2]),
        "reward":       reward,
        "id_switches":  info["id_switches"],
    })
    step += 1
    if step % 100 == 0:
        print(f"Frame {step}/{env._n_frames}")

df = pd.DataFrame(rows)
out = cfg["paths"]["debug_csv"]
os.makedirs(os.path.dirname(out), exist_ok=True)
df.to_csv(out, index=False)
print(f"\nSaved to {out}")
print(df[["conf_vel", "kf_residual", "feat_dist"]].describe())