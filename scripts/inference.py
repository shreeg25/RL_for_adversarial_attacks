# scripts/inference.py
import sys, os
sys.path.insert(0, os.path.abspath("."))

import yaml
import numpy as np
from stable_baselines3 import PPO
from src.mot_env import MOT17Env

cfg  = yaml.safe_load(open("config.yaml"))
env  = MOT17Env(cfg["data"]["seq_path"])
model = PPO.load(cfg["paths"]["model_save"])

obs, _ = env.reset()
total_switches = 0
action_counts  = {0: 0, 1: 0, 2: 0, 3: 0}

while True:
    # deterministic=False  ← stochastic sampling, not argmax
    # This is the live EOT defense — the agent's output is non-deterministic
    action, _ = model.predict(obs, deterministic=False)
    action = int(action)
    obs, reward, done, _, info = env.step(action)

    total_switches        += info["id_switches"]
    action_counts[action] += 1

    if done:
        break

total = sum(action_counts.values())
print(f"\nTotal ID switches:  {total_switches}")
print(f"Action distribution:")
for a, count in action_counts.items():
    label = ["T0 clean", "T1 warp", "T2 noise", "T3 cutout"][a]
    print(f"  {label}: {count} ({100*count/total:.1f}%)")