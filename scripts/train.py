# scripts/train.py
import sys, os
sys.path.insert(0, os.path.abspath("."))

import yaml
from src.mot_env import MOT17Env
from src.ppo_agent import build_ppo

cfg = yaml.safe_load(open("config.yaml"))

def make_env():
    return MOT17Env(
        seq_path=cfg["data"]["seq_path"],
        w1=cfg["reward"]["w1"],
        w2=cfg["reward"]["w2"],
        w3=cfg["reward"]["w3"],
    )

model = build_ppo(make_env, cfg)
model.learn(total_timesteps=cfg["ppo"]["total_timesteps"])

save_path = cfg["paths"]["model_save"]
os.makedirs(os.path.dirname(save_path), exist_ok=True)
model.save(save_path)
print(f"Model saved to {save_path}")