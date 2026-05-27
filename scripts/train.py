# scripts/train.py
import sys, os
sys.path.insert(0, os.path.abspath("."))

import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback
)
from src.mot_env import MOT17Env
from src.device import DEVICE

cfg = yaml.safe_load(open("config.yaml"))

# ── GPU memory report ────────────────────────────────────────────────
if DEVICE.type == "cuda":
    total_vram = torch.cuda.get_device_properties(DEVICE).total_memory / 1e9
    print(f"[train] VRAM available: {total_vram:.1f} GB")
    # Rule of thumb: each parallel env needs ~0.4 GB for ReID model
    max_envs = max(1, int(total_vram / 0.4))
    n_envs   = min(cfg["device"].get("n_envs", 4), max_envs)
    print(f"[train] Launching {n_envs} parallel environments")
else:
    n_envs = 1
    print("[train] CPU mode — single environment")

def make_env():
    return MOT17Env(
        seq_path=cfg["data"]["seq_path"],
        w1=cfg["reward"]["w1"],
        w2=cfg["reward"]["w2"],
        w3=cfg["reward"]["w3"],
    )

# ─── WINDOWS MULTIPROCESSING GUARD ─────────────────────────────────────────
if __name__ == "__main__":
    
    # SubprocVecEnv runs each env in its own process
    vec_env = make_vec_env(
        make_env,
        n_envs=n_envs,
        vec_env_cls=DummyVecEnv,
    )

    eval_env = make_vec_env(make_env, n_envs=1)

    p = cfg["ppo"]
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        learning_rate=p["learning_rate"],
        n_steps=p["n_steps"],
        batch_size=p["batch_size"],
        n_epochs=p["n_epochs"],
        gamma=p["gamma"],
        gae_lambda=p["gae_lambda"],
        clip_range=p["clip_range"],
        ent_coef=p["ent_coef"],
        policy_kwargs=dict(
            net_arch=[dict(pi=p["net_arch"], vf=p["net_arch"])],
        ),
        tensorboard_log=cfg["paths"]["tb_logs"],
        device=DEVICE,
    )

    callbacks = [
        CheckpointCallback(
            save_freq=50_000 // max(1, n_envs),
            save_path=os.path.dirname(cfg["paths"]["model_save"]),
            name_prefix="mtd_ppo_ckpt",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.dirname(cfg["paths"]["model_save"]),
            eval_freq=25_000 // max(1, n_envs),
            n_eval_episodes=3,
            deterministic=False,
            verbose=1,
        ),
    ]

    print("[train] Starting PPO learning loop...")
    model.learn(
        total_timesteps=p["total_timesteps"],
        callback=callbacks,
        progress_bar=True,
    )

    save_path = cfg["paths"]["model_save"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"[train] Final model saved → {save_path}")

    if DEVICE.type == "cuda":
        allocated = torch.cuda.memory_allocated(DEVICE) / 1e9
        reserved  = torch.cuda.memory_reserved(DEVICE) / 1e9
        print(f"[train] GPU memory at end — allocated: {allocated:.2f} GB  reserved: {reserved:.2f} GB")