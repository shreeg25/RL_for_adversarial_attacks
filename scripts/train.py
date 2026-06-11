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

# --------------------------------------------------------------
# Load base config
# --------------------------------------------------------------
cfg = yaml.safe_load(open("config.yaml"))

def get_all_sequence_paths():
    """Return list of all available sequences (clean + poisoned)."""
    sequences = [cfg["data"]["seq_path"]]
    if "extra_sequences" in cfg["data"] and cfg["data"]["extra_sequences"]:
        sequences.extend(cfg["data"]["extra_sequences"])
    # Filter to existing paths
    return [p for p in sequences if os.path.exists(p)]

if __name__ == "__main__":
    # Force 'spawn' context to completely isolate CUDA allocations between processes
    if multiprocessing.get_start_method(allow_none=None) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)

    all_sequences = get_all_sequence_paths()
    print(f"[TRAIN] Discovered {len(all_sequences)} total sequences.")

    # -------------------- Train/Validation split --------------------
    validation_fraction = cfg["data"].get("validation_fraction", 0.2)
    # Shuffle with a fixed seed for reproducibility across runs
    rng = np.random.RandomState(seed=42)
    shuffled = rng.permutation(all_sequences)
    split_idx = int(len(shuffled) * (1.0 - validation_fraction))
    train_sequences = shuffled[:split_idx].tolist()
    val_sequences   = shuffled[split_idx:].tolist()
    print(f"[TRAIN] Using {len(train_sequences)} sequences for training, "
          f"{len(val_sequences)} sequences for validation.")

    # Device setup
    if DEVICE.type == "cuda":
        total_vram = torch.cuda.get_device_properties(DEVICE).total_memory / 1e9
        print(f"[TRAIN] VRAM available: {total_vram:.1f} GB on {torch.cuda.get_device_name(DEVICE)}")
        # Performance booster: Allow TF32 matmuls on Ada architecture
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("[TRAIN] Warning: Running on CPU mode")

    # --------------------------------------------------------------
    # Helper to build vec_envs given a sequence list
    # --------------------------------------------------------------
    def make_vec_envs(seq_list):
        n_envs = min(cfg["device"].get("n_envs", 4), len(seq_list))
        
        if n_envs == 1:
            def make_env(rank):
                def _init():
                    import torch
                    torch.set_num_threads(1)
                    assigned_seq = seq_list[rank % len(seq_list)]
                    env = MOT17Env(
                        seq_path=assigned_seq,
                        w_rec=cfg["reward"]["w_rec"],
                        w_fp=cfg["reward"]["w_fp"],
                        w_lost=cfg["reward"]["w_lost"],
                        w_cost=cfg["reward"]["w_cost"],
                    )
                    env._cfg = cfg
                    from stable_baselines3.common.monitor import Monitor
                    return Monitor(env) # Added Monitor so you can see ep_rew_mean!
                return _init
            return DummyVecEnv([make_env(i) for i in range(n_envs)]), n_envs
        else:
            def make_parallel_env(rank):
                def _init():
                    import torch
                    torch.set_num_threads(1)
                    assigned_seq = seq_list[rank % len(seq_list)]
                    env = MOT17Env(
                        seq_path=assigned_seq,
                        w_rec=cfg["reward"]["w_rec"],
                        w_fp=cfg["reward"]["w_fp"],
                        w_lost=cfg["reward"]["w_lost"],
                        w_cost=cfg["reward"]["w_cost"],
                    )
                    env._cfg = cfg
                    from stable_baselines3.common.monitor import Monitor
                    return Monitor(env) # Added Monitor so you can see ep_rew_mean!
                return _init

            vec_env = SubprocVecEnv([make_parallel_env(i) for i in range(n_envs)])
            return vec_env, n_envs

    # --------------------------------------------------------------
    # Execute PPO Training
    # --------------------------------------------------------------
    print("\n=== STARTING TRACE PPO TRAINING ===")
    
    vec_env, train_n_envs = make_vec_envs(train_sequences)
    eval_env, _ = make_vec_envs(val_sequences)

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
        CheckpointCallback(
            save_freq=max(25000 // train_n_envs, 1),
            save_path=save_dir,
            name_prefix="trace_ckpt"
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=save_dir,
            eval_freq=max(25000 // train_n_envs, 1),
            n_eval_episodes=2,
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