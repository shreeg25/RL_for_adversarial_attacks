# scripts/train.py
import sys
import os
import warnings
import multiprocessing

# 1. Clear any multi-threading CPU contention locks before loading frameworks
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="deep_sort_realtime")

sys.path.insert(0, os.path.abspath("."))

import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from src.mot_env import MOT17Env
from src.device import DEVICE

cfg = yaml.safe_load(open("config.yaml"))

def get_sequence_paths():
    sequences = [cfg["data"]["seq_path"]]
    if "extra_sequences" in cfg["data"] and cfg["data"]["extra_sequences"]:
        sequences.extend(cfg["data"]["extra_sequences"])
    return [p for p in sequences if os.path.exists(p)]

if __name__ == "__main__":
    # Force 'spawn' context to completely isolate CUDA allocations between processes
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)

    all_sequences = get_sequence_paths()
    print(f"[TRAIN] Discovered {len(all_sequences)} sequences for training.")

    if DEVICE.type == "cuda":
        total_vram = torch.cuda.get_device_properties(DEVICE).total_memory / 1e9
        print(f"[TRAIN] VRAM available: {total_vram:.1f} GB on RTX 6000 Ada")
        # Enforce environment allocation matched to the number of tracked sequences
        n_envs = min(cfg["device"].get("n_envs", 4), len(all_sequences))
        
        # Performance booster: Allow TF32 matmuls on Ada architecture
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        n_envs = 1
        print("[TRAIN] Warning: Running on CPU mode")

    def make_parallel_env(rank):
        def _init():
            import torch
            torch.set_num_threads(1) 
            assigned_seq = all_sequences[rank % len(all_sequences)]
            return MOT17Env(
                seq_path=assigned_seq,
                w1=cfg["reward"]["w1"],
                w2=cfg["reward"]["w2"],
                w3=cfg["reward"]["w3"],
                w4=cfg["reward"]["w4"], # Link the new weight
            )
        return _init

    print(f"[TRAIN] Launching {n_envs} true parallel workers via SubprocVecEnv...")
    vec_env = SubprocVecEnv([make_parallel_env(i) for i in range(n_envs)])
    eval_env = SubprocVecEnv([make_parallel_env(0)]) # Fixed evaluation environment

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
        policy_kwargs=dict(net_arch=[dict(pi=p["net_arch"], vf=p["net_arch"])]),
        tensorboard_log=cfg["paths"]["tb_logs"],
        device=DEVICE,
    )

    save_dir = os.path.dirname(cfg["paths"]["model_save"])
    os.makedirs(save_dir, exist_ok=True)

    callbacks = [
        CheckpointCallback(save_freq=50000 // n_envs, save_path=save_dir, name_prefix="trace_ckpt"),
        EvalCallback(eval_env, best_model_save_path=save_dir, eval_freq=25000 // n_envs, n_eval_episodes=2, deterministic=False, verbose=1)
    ]

    print("[TRAIN] Starting high-performance PPO learning loop...")
    model.learn(total_timesteps=p["total_timesteps"], callback=callbacks, progress_bar=True)

    model.save(cfg["paths"]["model_save"])
    print(f"[TRAIN] Final model saved successfully → {cfg['paths']['model_save']}")