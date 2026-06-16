# scripts/train.py
import sys
import os
import warnings
import multiprocessing
import numpy as np

# 1. Clear any multi-threading CPU contention locks before loading frameworks
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="deep_sort_realtime")

sys.path.insert(0, os.path.abspath("."))

import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from src.mot_env import MOT17Env
from src.device import DEVICE
import random
import gymnasium as gym
from stable_baselines3.common.vec_env import VecFrameStack

# --------------------------------------------------------------
# Load base config
# --------------------------------------------------------------
cfg = yaml.safe_load(open("config.yaml"))

def get_all_sequence_paths():
    sequences = [cfg["data"]["seq_path"]]
    if "extra_sequences" in cfg["data"] and cfg["data"]["extra_sequences"]:
        sequences.extend(cfg["data"]["extra_sequences"])
    return [p for p in sequences if os.path.exists(p)]

if __name__ == "__main__":
    # Force 'spawn' context to completely isolate CUDA allocations
    if multiprocessing.get_start_method(allow_none=None) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)

    all_sequences = get_all_sequence_paths()
    print(f"[TRAIN] Discovered {len(all_sequences)} total sequences.")

    validation_fraction = cfg["data"].get("validation_fraction", 0.2)
    rng = np.random.RandomState(seed=42)
    shuffled = rng.permutation(all_sequences)
    split_idx = int(len(shuffled) * (1.0 - validation_fraction))
    train_sequences = shuffled[:split_idx].tolist()
    val_sequences   = shuffled[split_idx:].tolist()
    
    # Ensure at least one training and one val sequence if data is limited
    if not train_sequences: train_sequences = all_sequences
    if not val_sequences: val_sequences = all_sequences

    print(f"[TRAIN] Using {len(train_sequences)} seqs for training, {len(val_sequences)} seqs for validation.")

    # Device setup & Ada/Ampere Optimizations
    if DEVICE.type == "cuda":
        total_vram = torch.cuda.get_device_properties(DEVICE).total_memory / 1e9
        print(f"[TRAIN] VRAM available: {total_vram:.1f} GB on {torch.cuda.get_device_name(DEVICE)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("[TRAIN] Warning: Running on CPU mode")

    class DomainRandomizationWrapper(gym.Wrapper):
        def __init__(self, seq_list, env_kwargs):
            self.seq_list = seq_list
            self.env_kwargs = env_kwargs
            # Boot with a random domain
            env = MOT17Env(seq_path=random.choice(self.seq_list), **self.env_kwargs)
            super().__init__(env)

        def reset(self, **kwargs):
            # The 8GB VRAM Hack: Tear down the old domain and load a new one
            self.env.close()
            self.env = MOT17Env(seq_path=random.choice(self.seq_list), **self.env_kwargs)
            return self.env.reset(**kwargs)

    def make_vec_envs(seq_list, force_n_envs=None):
        # Force single environment to protect 8GB VRAM
        def make_env():
            def _init():
                import torch
                torch.set_num_threads(1)
                
                # Wrap the environment in our Domain Randomizer
                env_kwargs = {
                    "w_rec": cfg["reward"]["w_rec"], 
                    "w_fp": cfg["reward"]["w_fp"],
                    "w_lost": cfg["reward"]["w_lost"], 
                    "w_cost": cfg["reward"]["w_cost"]
                }
                random_env = DomainRandomizationWrapper(seq_list, env_kwargs)
                
                from stable_baselines3.common.monitor import Monitor
                return Monitor(random_env)
            return _init
        
        # Build the dummy vector and immediately wrap it in FrameStack
        vec_env = DummyVecEnv([make_env()])
        vec_env = VecFrameStack(vec_env, n_stack=4) # Expands 12D to 48D Memory
        
        return vec_env, 1

    print("\n=== STARTING TRACE PPO TRAINING ===")
    
    # Force validation to only use 1 environment to save VRAM during eval phases
    vec_env, train_n_envs = make_vec_envs(train_sequences)
    eval_env, _ = make_vec_envs(val_sequences, force_n_envs=1) 

    p = cfg["ppo"]
    
    print(f"[TRAIN] Initializing PPO with {train_n_envs} parallel workers...")
    print(f"[TRAIN] Buffer size per update: {train_n_envs * p['n_steps']} steps.")

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

    # OPTIMIZATION: Drastically reduce evaluation overhead. 
    # Evaluate only every ~20,000 global steps.
    eval_freq_steps = max(20000 // train_n_envs, 1)

    callbacks = [
        CheckpointCallback(
            save_freq=eval_freq_steps,
            save_path=save_dir,
            name_prefix="trace_ckpt"
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=save_dir,
            eval_freq=eval_freq_steps,
            n_eval_episodes=1, # Only evaluate 1 episode to save massive amounts of time
            deterministic=False,
            verbose=1,
        )
    ]

    print(f"[TRAIN] Launching {cfg['ppo']['total_timesteps']} timesteps...")
    model.learn(
        total_timesteps=cfg["ppo"]["total_timesteps"],
        callback=callbacks,
        progress_bar=True,
    )

    model.save(cfg["paths"]["model_save"])
    print(f"[TRAIN] Final model saved successfully → {cfg['paths']['model_save']}")
    
    vec_env.close()
    eval_env.close()