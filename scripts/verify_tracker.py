# scripts/verify_tracker.py
import sys, os
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import yaml
from src.mot_env import MOT17Env

print("[DEBUG] Loading config...", flush=True)
cfg = yaml.safe_load(open("config.yaml"))

print(f"[DEBUG] Initializing MOT17Env using path: {cfg['data']['seq_path']}", flush=True)
print("[DEBUG] WARNING: If it hangs right here, PyTorch is downloading weights or compiling CUDA kernels...", flush=True)
env = MOT17Env(cfg["data"]["seq_path"])

print("[DEBUG] Environment initialized successfully. Running env.reset()...", flush=True)
obs, _ = env.reset()
rows = []

done = False
step = 0
print("[DEBUG] Entering inference loop. Executing T0 (clean pass)...", flush=True)

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
    
    # Print every frame for the first 10 frames to prove it's alive, then every 10
    if step <= 10 or step % 10 == 0:
        print(f"[DEBUG] Processed Frame {step}/{env._n_frames} | State: {obs}", flush=True)

df = pd.DataFrame(rows)
out = cfg["paths"]["debug_csv"]
os.makedirs(os.path.dirname(out), exist_ok=True)
df.to_csv(out, index=False)
print(f"\n[DEBUG] Saved to {out}")
print(df[["conf_vel", "kf_residual", "feat_dist"]].describe())