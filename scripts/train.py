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
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from src.mot_env import MOT17Env
from src.device import DEVICE

  # --------------------------------------------------------------
  # Load base config
  # --------------------------------------------------------------
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

      # --------------------------------------------------------------
      # Curriculum settings for attack probability
      # --------------------------------------------------------------
      # Start with almost no attacks, linearly increase to a final value.
      ATTACK_START = 0.05   # 5 % of frames attacked at the beginning
      ATTACK_END   = 0.40   # 40 % of frames attacked by the end of training
      CURRICULUM_STEPS = 10  # how many stages we split the training into

      # Generate a list of attack probabilities, one per curriculum stage
      attack_schedule = np.linspace(ATTACK_START, ATTACK_END, num=CURRICULUM_STEPS)
      timesteps_per_stage = cfg["ppo"]["total_timesteps"] // CURRICULUM_STEPS

      print(f"[CURRICULUM] Will train for {CURRICULUM_STEPS} stages.")
      print(f"[CURRICULUM] Attack probability will go from {ATTACK_START:.2f} → {ATTACK_END:.2f}")

      # --------------------------------------------------------------
      # Helper to build vec_envs with a given attack probability
      # --------------------------------------------------------------
      def make_vec_envs(attack_prob: float):
          def make_parallel_env(rank):
              def _init():
                  import torch
                  torch.set_num_threads(1)
                  assigned_seq = all_sequences[rank % len(all_sequences)]
                  env = MOT17Env(
                      seq_path=assigned_seq,
                      w1=cfg["reward"]["w1"],
                      w2=cfg["reward"]["w2"],
                      w3=cfg["reward"]["w3"],
                      w4=cfg["reward"]["w4"],
                  )
                  # inject the attack probability for this env instance
                  env.set_attack_prob(attack_prob)
                  # also make the config available to the env for reward w0 lookup
                  env._cfg = cfg
                  return env
              return _init

          vec_env = SubprocVecEnv([make_parallel_env(i) for i in range(n_envs)])
          # eval env uses the same attack probability as the first training env (could also be 0)
          eval_env = SubprocVecEnv([make_parallel_env(0)])
          return vec_env, eval_env

      # --------------------------------------------------------------
      # Loop over curriculum stages
      # --------------------------------------------------------------
      for stage_idx, attack_prob in enumerate(attack_schedule, start=1):
          print(f"\n=== CURRICULUM STAGE {stage_idx}/{CURRICULUM_STEPS} ===")
          print(f"[TRAIN] Setting attack_prob = {attack_prob:.2f}")

          vec_env, eval_env = make_vec_envs(attack_prob)

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
                  save_freq=max(50000 // n_envs, 1),   # avoid zero
                  save_path=save_dir,
                  name_prefix=f"trace_ckpt_stage{stage_idx}"
              ),
              EvalCallback(
                  eval_env,
                  best_model_save_path=save_dir,
                  eval_freq=max(25000 // n_envs, 1),
                  n_eval_episodes=2,
                  deterministic=False,
                  verbose=1,
              )
          ]

          print("[TRAIN] Starting PPO learning loop...")
          model.learn(
              total_timesteps=timesteps_per_stage,
              callback=callbacks,
              progress_bar=True,
              reset_num_timesteps=False   # keep accumulating across stages
          )

          # Optional: save a stage‑specific model for inspection
          stage_save_path = os.path.join(save_dir, f"ppo_stage{stage_idx}.zip")
          model.save(stage_save_path)
          print(f"[TRAIN] Stage model saved → {stage_save_path}")

          vec_env.close()
          eval_env.close()

      # --------------------------------------------------------------
      # Final save (the same path as before for compatibility)
      # --------------------------------------------------------------
      model.save(cfg["paths"]["model_save"])
      print(f"[TRAIN] Final model saved successfully → {cfg['paths']['model_save']}")