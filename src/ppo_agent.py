# src/ppo_agent.py
"""
Constructs the PPO model with the correct architecture:
  Actor MLP:  3 → [64, 64] → softmax(4)
  Critic MLP: 3 → [64, 64] → V(s)

The ent_coef keeps the policy stochastic, which is the key
structural defense against EOT attacks.
"""
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def build_ppo(env_fn, cfg: dict) -> PPO:
    vec_env = make_vec_env(env_fn, n_envs=1)
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
            net_arch=[dict(pi=p["net_arch"], vf=p["net_arch"])]
        ),
        tensorboard_log=cfg["paths"]["tb_logs"],
    )
    return model