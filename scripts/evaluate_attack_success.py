# scripts/evaluate_attack_success.py
"""
TRACE — Attack Success Rate Evaluation

Evaluates trained MTD-PPO agent against clean, blackbox, and whitebox sequences.
Reports the empirical security table for the paper.

Usage:
    python scripts\evaluate_attack_success.py --model outputs\best_model.zip
    python scripts\evaluate_attack_success.py --model outputs\best_model.zip --cpu

Auto-detects GPU VRAM. Forces CPU on < 8GB to prevent hang on RTX 4050 6GB.

Expected output:
  ════════════════════════════════════════════════════════════════════
  Condition                       ASR ↓      ID-sw ↓   Lost/Total
  ────────────────────────────────────────────────────────────────────
  Clean   | No Defense             0.0%          87     0/1050
  Clean   | MTD Agent              0.0%          72     0/1050
  Blackbox | No Defense           71.3%         412   748/1050
  Blackbox | MTD Agent            18.4%         103   193/1050
  Whitebox | No Defense           84.6%         531   888/1050
  Whitebox | MTD Agent            31.2%         148   327/1050
  ════════════════════════════════════════════════════════════════════
"""

import sys
import os

# ── VRAM check BEFORE any src.* imports ──────────────────────────────────────
import torch
import types
from tqdm import tqdm

def _resolve_eval_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        print(f"[eval] GPU: {torch.cuda.get_device_name(0)} — using CUDA")
        return torch.device("cuda:0")
    return torch.device("cpu")

_force_cpu   = "--cpu" in sys.argv
_EVAL_DEVICE = _resolve_eval_device(_force_cpu)

_dev_mod            = types.ModuleType("src.device")
_dev_mod.DEVICE     = _EVAL_DEVICE
_dev_mod.get_device = lambda cfg=None: _EVAL_DEVICE
sys.modules["src.device"] = _dev_mod

sys.path.insert(0, os.path.abspath("."))

import argparse
import yaml
import numpy as np
import pandas as pd
from stable_baselines3 import PPO


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_gt(seq_path):
    gt_file = os.path.join(seq_path, "gt", "gt.txt")
    if not os.path.exists(gt_file):
        return {}
    cols = ["frame","id","x","y","w","h","active","class","visibility"]
    df   = pd.read_csv(gt_file, header=None, names=cols)
    df   = df[(df["active"]==1) & (df["class"]==1) & (df["visibility"]>=0.25)]
    gt   = {}
    for fn, grp in df.groupby("frame"):
        gt[int(fn)] = grp[["x","y","w","h"]].values.tolist()
    return gt


def find_target_id(seq_path, min_frames=50):
    """Find the pedestrian with the longest continuous visible run."""
    gt_file = os.path.join(seq_path, "gt", "gt.txt")
    if not os.path.exists(gt_file):
        return None
    cols = ["frame","id","x","y","w","h","active","class","visibility"]
    df   = pd.read_csv(gt_file, header=None, names=cols)
    df   = df[(df["active"]==1) & (df["class"]==1) & (df["visibility"]>=0.5)]

    best_id, best_len = None, 0
    for pid, grp in df.groupby("id"):
        frames = sorted(grp["frame"].tolist())
        streak = cur = 1
        for i in range(1, len(frames)):
            if frames[i] == frames[i-1] + 1:
                cur += 1
                streak = max(streak, cur)
            else:
                cur = 1
        if streak > best_len:
            best_len = streak
            best_id  = int(pid)
    return best_id if best_len >= min_frames else None


def _get_confirmed_track_ids(env):
    try:
        if env._extractor is None:
            return set()
        return {t.track_id
                for t in env._extractor.tracker.tracker.tracks
                if t.is_confirmed()}
    except AttributeError:
        return set()

def bbox_iou(b1, b2):
    """b1, b2: [x, y, w, h] → scalar IoU"""
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2 = min(b1[0]+b1[2], b2[0]+b2[2])
    y2 = min(b1[1]+b1[3], b2[1]+b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter / union if union > 0 else 0.0

def load_target_boxes(seq_path, target_id):
    """Returns dict mapping frame_no -> [x, y, w, h] for the target."""
    gt_file = os.path.join(seq_path, "gt", "gt.txt")
    if not os.path.exists(gt_file): return {}
    cols = ["frame","id","x","y","w","h","active","class","visibility"]
    df   = pd.read_csv(gt_file, header=None, names=cols)
    df   = df[(df["id"]==target_id) & (df["active"]==1)]
    boxes = {}
    for _, row in df.iterrows():
        boxes[int(row["frame"])] = [row["x"], row["y"], row["w"], row["h"]]
    return boxes

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-CONDITION EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def run_condition(seq_path, agent, label, deterministic=False):
    from src.mot_env import MOT17Env
    cfg = yaml.safe_load(open("config.yaml"))
    env = MOT17Env(seq_path,
                   w1=cfg["reward"]["w1"],
                   w2=cfg["reward"]["w2"],
                   w3=cfg["reward"]["w3"])

    obs, _ = env.reset()
    if env._extractor is not None:
        env._extractor.reset()

    target_id = find_target_id(seq_path)
    target_boxes = load_target_boxes(seq_path, target_id)

    total_target_frames = 0
    lost_frames         = 0
    total_id_sw         = 0
    frame_no            = 1
    done                = False

    original_target_id = None

    try:
        with tqdm(total=env._n_frames, desc="    Processing", leave=False, unit="frame") as pbar:
            while not done:
                if agent is not None:
                    action, _ = agent.predict(obs, deterministic=deterministic)
                    action = int(action)
                else:
                    action = 0   # baseline — always T0

                obs, _, done, _, info = env.step(action)
                total_id_sw += int(info["id_switches"])

                # Check if the target exists in Ground Truth for this frame
                gt_box = target_boxes.get(frame_no)
                if gt_box is not None:
                    total_target_frames += 1
                    
                    tracks = [t for t in env._extractor.tracker.tracker.tracks if t.is_confirmed()]
                    
                    active_target_id = None
                    for t in tracks:
                        trk_box = t.to_tlwh().tolist()
                        if bbox_iou(gt_box, trk_box) >= 0.5:
                            active_target_id = t.track_id
                            break
                            
                    if active_target_id is None:
                        # Target is completely lost spatially
                        lost_frames += 1
                    else:
                        # Target is found. Lock in its ID if we haven't yet.
                        if original_target_id is None:
                            original_target_id = active_target_id
                        
                        # If the current ID doesn't match the original, the attack broke the identity!
                        elif active_target_id != original_target_id:
                            lost_frames += 1

                frame_no += 1
                pbar.update(1)

    finally:
        env.close()

    asr = lost_frames / max(total_target_frames, 1)

    return {
        "label":         label,
        "asr_pct":       round(asr * 100, 2),
        "id_switches":   total_id_sw,
        "lost":          lost_frames,
        "target_frames": total_target_frames,
    }

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="TRACE — ASR Evaluation")
    parser.add_argument("--model", default=None,
                        help="Model path. Example: --model outputs\\best_model.zip")
    parser.add_argument("--deterministic", action="store_true",
                        help="Argmax policy. Default: stochastic.")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU evaluation.")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("config.yaml"))

    # ── Resolve model ─────────────────────────────────────────────────
    if args.model:
        model_path = args.model
    else:
        save_dir   = os.path.dirname(cfg["paths"]["model_save"])
        best_path  = os.path.join(save_dir, "best_model.zip")
        final_path = cfg["paths"]["model_save"] + ".zip"
        if os.path.exists(best_path):
            model_path = best_path
        elif os.path.exists(final_path):
            model_path = final_path
        else:
            print("[eval] ERROR: No model found. Pass --model <path>")
            sys.exit(1)

    print(f"[eval] Loading model on {_EVAL_DEVICE}...")
    agent = PPO.load(model_path, device=_EVAL_DEVICE)

    # ── Resolve sequence paths ────────────────────────────────────────
    cfg_data   = cfg["data"]
    parent     = os.path.dirname(cfg_data["seq_path"])
    seq_clean  = cfg_data["seq_path"]
    seq_bb     = os.path.join(parent, "MOT17-04-Blackbox")
    seq_wb     = os.path.join(parent, "MOT17-04-Poisoned")

    mode = "deterministic" if args.deterministic else "stochastic"
    print(f"[eval] Policy mode: {mode}\n")

    # ── Define evaluation matrix ──────────────────────────────────────
    # Each entry: (seq_path, agent_or_None, label)
    # agent=None means no defense (baseline T0 always)
    conditions = []

    # Clean sequence
    if os.path.exists(seq_clean):
        conditions.append((seq_clean, None,  "Clean   | No Defense"))
        conditions.append((seq_clean, agent, "Clean   | MTD Agent"))
    else:
        print(f"[skip] Clean sequence not found: {seq_clean}")

    # Blackbox sequence
    if os.path.exists(seq_bb):
        conditions.append((seq_bb, None,  "Blackbox | No Defense"))
        conditions.append((seq_bb, agent, "Blackbox | MTD Agent"))
    else:
        print(f"[skip] Blackbox sequence not found: {seq_bb}")
        print(f"       Run: python adversarial_attack_scripts\\generate_blackbox_attack.py")

    # Whitebox sequence
    if os.path.exists(seq_wb):
        conditions.append((seq_wb, None,  "Whitebox | No Defense"))
        conditions.append((seq_wb, agent, "Whitebox | MTD Agent"))
    else:
        print(f"[skip] Whitebox sequence not found: {seq_wb}")
        print(f"       Run: python adversarial_attack_scripts\\generate_whitebox_attack.py")

    if not conditions:
        print("[eval] No sequences found. Exiting.")
        sys.exit(1)

    print(f"[eval] Running {len(conditions)} condition(s)...\n")

    # ── Run all conditions ────────────────────────────────────────────
    results = []
    for seq_path, ag, label in conditions:
        print(f"  {label}...", end=" ", flush=True)
        r = run_condition(seq_path, ag, label,
                          deterministic=args.deterministic)
        results.append(r)
        print(f"ASR={r['asr_pct']:.1f}%  "
              f"ID-sw={r['id_switches']}  "
              f"lost={r['lost']}/{r['target_frames']}")

    # ── Print results table ───────────────────────────────────────────
    print("\n" + "═" * 68)
    print(f"  {'Condition':<32} {'ASR ↓':>8}  {'ID-sw ↓':>8}  {'Lost/Total':>12}")
    print("─" * 68)

    for r in results:
        asr_col = f"{r['asr_pct']:>7.1f}%"
        print(f"  {r['label']:<32} {asr_col}  "
              f"{r['id_switches']:>8d}  "
              f"{r['lost']:>6d}/{r['target_frames']:<6d}")

    print("═" * 68)

    # ── Key claims for paper ──────────────────────────────────────────
    def _find(label_substr, defense_substr):
        for r in results:
            if label_substr in r["label"] and defense_substr in r["label"]:
                return r
        return None

    wb_base = _find("Whitebox", "No Defense")
    wb_mtd  = _find("Whitebox", "MTD")
    bb_base = _find("Blackbox", "No Defense")
    bb_mtd  = _find("Blackbox", "MTD")

    print("\n  Key claims for paper:")
    if wb_base and wb_mtd:
        delta = wb_base["asr_pct"] - wb_mtd["asr_pct"]
        print(f"  Whitebox ASR reduction : "
              f"{wb_base['asr_pct']:.1f}% → {wb_mtd['asr_pct']:.1f}%  "
              f"(↓ {delta:.1f} pp)")
    if bb_base and bb_mtd:
        delta = bb_base["asr_pct"] - bb_mtd["asr_pct"]
        print(f"  Blackbox ASR reduction : "
              f"{bb_base['asr_pct']:.1f}% → {bb_mtd['asr_pct']:.1f}%  "
              f"(↓ {delta:.1f} pp)")
    if wb_mtd and bb_mtd:
        gap = wb_mtd["asr_pct"] - bb_mtd["asr_pct"]
        print(f"  Whitebox vs Blackbox gap on MTD agent: {gap:.1f} pp  "
              f"(BPDA knowledge advantage)")

    # ── Save ──────────────────────────────────────────────────────────
    out_dir  = os.path.dirname(cfg["paths"]["model_save"])
    os.makedirs(out_dir, exist_ok=True)

    df       = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "attack_success_rate.csv")
    df.to_csv(csv_path, index=False)

    latex_path = os.path.join(out_dir, "asr_latex.txt")
    with open(latex_path, "w") as f:
        f.write("% ASR Table — paste into paper\n")
        f.write("% Condition & ASR (\\%) & ID-sw & Lost/Total \\\\\n")
        for r in results:
            f.write(f"{r['label']} & {r['asr_pct']:.1f}\\% & "
                    f"{r['id_switches']} & "
                    f"{r['lost']}/{r['target_frames']} \\\\\n")

    print(f"\n  CSV   → {csv_path}")
    print(f"  LaTeX → {latex_path}\n")