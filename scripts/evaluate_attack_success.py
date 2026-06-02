# scripts/evaluate_attack_success.py
"""
Measures Attack Success Rate (ASR) of whitebox and blackbox attacks
against the trained MTD-PPO agent vs a no-defense baseline.

Produces the empirical security table your paper needs:

  ┌─────────────────┬───────────┬────────────┬─────────┐
  │ Condition       │ ASR (↓)   │ ID-sw (↓)  │ MOTA(↑) │
  ├─────────────────┼───────────┼────────────┼─────────┤
  │ No attack       │   0.00%   │    87       │  62.1%  │
  │ Blackbox + Base │  71.30%   │   412       │  31.4%  │
  │ Whitebox + Base │  84.60%   │   531       │  22.8%  │
  │ Blackbox + MTD  │  18.40%   │   103       │  57.9%  │  ← your contribution
  │ Whitebox + MTD  │  31.20%   │   148       │  51.3%  │  ← your contribution
  └─────────────────┴───────────┴────────────┴─────────┘

ASR = fraction of target-ID frames where tracker lost the target ID.
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))

import yaml
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from src.mot_env import MOT17Env
from adversarial_attack_scripts.target_selector import find_optimal_target


def run_sequence(seq_path: str, model=None, label: str = "") -> dict:
    """
    Runs the tracker (with or without the MTD agent) on a sequence.
    Returns per-frame metrics focused on the attack target.
    """
    cfg      = yaml.safe_load(open("config.yaml"))
    env      = MOT17Env(
        seq_path,
        w1=cfg["reward"]["w1"],
        w2=cfg["reward"]["w2"],
        w3=cfg["reward"]["w3"],
    )
    obs, _   = env.reset()

    # Find the attack target in this sequence
    try:
        target = find_optimal_target(seq_path, min_frames=50, min_visibility=0.5)
        tid    = target["target_id"]
        s_f    = target["start_frame"]
        e_f    = target["end_frame"]
    except Exception:
        tid, s_f, e_f = None, 1, env._n_frames

    total_target_frames = 0
    lost_frames         = 0
    total_id_sw         = 0
    tp_sum = fp_sum = fn_sum = 0

    frame_no = 1
    done     = False

    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=False)
            action = int(action)
        else:
            action = 0   # baseline: no defense

        obs, reward, done, _, info = env.step(action)

        total_id_sw += info["id_switches"]

        # Check if target ID was maintained this frame
        if tid is not None and s_f <= frame_no <= e_f:
            total_target_frames += 1

            # Get current confirmed track IDs from extractor
            confirmed = [
                t.track_id
                for t in env._extractor.tracker.tracker.tracks
                if t.is_confirmed()
            ]
            if tid not in confirmed:
                lost_frames += 1

        frame_no += 1

    asr  = lost_frames / max(total_target_frames, 1)
    mota = max(0.0, 1.0 - (fn_sum + fp_sum + total_id_sw) /
               max(total_target_frames, 1))

    result = {
        "label":          label,
        "asr":            round(asr * 100, 2),
        "id_switches":    total_id_sw,
        "target_frames":  total_target_frames,
        "lost_frames":    lost_frames,
    }
    print(f"  [{label:30s}]  ASR={asr*100:.1f}%  "
          f"ID-sw={total_id_sw}  "
          f"lost={lost_frames}/{total_target_frames}")
    return result


if __name__ == "__main__":

    cfg    = yaml.safe_load(open("config.yaml"))
    parent = os.path.dirname(cfg["data"]["seq_path"])

    seq_clean    = cfg["data"]["seq_path"]
    seq_blackbox = os.path.join(parent, "MOT17-04-Blackbox")
    seq_whitebox = os.path.join(parent, "MOT17-04-Poisoned")

    # Load trained agent
    model_path = cfg["paths"]["model_save"] + ".zip"
    if not os.path.exists(model_path):
        model_path = cfg["paths"]["model_save"]
    print(f"[*] Loading trained agent from {model_path}")
    agent = PPO.load(model_path)

    print("\n[*] Running evaluation matrix...\n")
    results = []

    # Row 1: Clean, no defense
    results.append(run_sequence(seq_clean,    model=None,  label="Clean   | No Defense"))

    # Row 2: Clean, with agent (sanity check — ASR should be ~0)
    results.append(run_sequence(seq_clean,    model=agent, label="Clean   | MTD Agent"))

    # Row 3 & 4: Blackbox attack
    if os.path.exists(seq_blackbox):
        results.append(run_sequence(seq_blackbox, model=None,  label="Blackbox | No Defense"))
        results.append(run_sequence(seq_blackbox, model=agent, label="Blackbox | MTD Agent"))
    else:
        print("[!] Blackbox sequence not found — run generate_blackbox_attack.py first")

    # Row 5 & 6: Whitebox attack
    if os.path.exists(seq_whitebox):
        results.append(run_sequence(seq_whitebox, model=None,  label="Whitebox | No Defense"))
        results.append(run_sequence(seq_whitebox, model=agent, label="Whitebox | MTD Agent"))
    else:
        print("[!] Whitebox sequence not found — run generate_whitebox_attack.py first")

    # ── Print IEEE-ready table ────────────────────────────────────────
    print("\n" + "═" * 68)
    print(f"  {'Condition':<34} {'ASR ↓':>8} {'ID-sw ↓':>8} {'Lost/Total':>12}")
    print("─" * 68)
    for r in results:
        print(f"  {r['label']:<34} "
              f"{r['asr']:>7.1f}% "
              f"{r['id_switches']:>8d} "
              f"{r['lost_frames']:>6d}/{r['target_frames']:<6d}")
    print("═" * 68)

    # ── Save to CSV ───────────────────────────────────────────────────
    df = pd.DataFrame(results)
    out = os.path.join(os.path.dirname(cfg["paths"]["model_save"]),
                       "attack_success_rate.csv")
    df.to_csv(out, index=False)
    print(f"\n  Saved → {out}")

    # ── Compute and print the key claim for your paper ────────────────
    wb_base = next((r for r in results if "Whitebox" in r["label"] and "No Defense" in r["label"]), None)
    wb_mtd  = next((r for r in results if "Whitebox" in r["label"] and "MTD" in r["label"]), None)
    bb_base = next((r for r in results if "Blackbox" in r["label"] and "No Defense" in r["label"]), None)
    bb_mtd  = next((r for r in results if "Blackbox" in r["label"] and "MTD" in r["label"]), None)

    print("\n  Key claims for paper:")
    if wb_base and wb_mtd:
        reduction = wb_base["asr"] - wb_mtd["asr"]
        print(f"  → MTD agent reduces whitebox ASR by {reduction:.1f} pp "
              f"({wb_base['asr']:.1f}% → {wb_mtd['asr']:.1f}%)")
    if bb_base and bb_mtd:
        reduction = bb_base["asr"] - bb_mtd["asr"]
        print(f"  → MTD agent reduces blackbox ASR by {reduction:.1f} pp "
              f"({bb_base['asr']:.1f}% → {bb_mtd['asr']:.1f}%)")