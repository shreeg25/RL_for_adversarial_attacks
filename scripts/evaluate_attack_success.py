# scripts/evaluate_attack_success.py
"""
TRACE — Attack Success Rate Evaluation

ROOT CAUSE FIX: Previous version compared GT pedestrian IDs against DeepSORT
track IDs. These are completely independent numbering systems. The coincidental
numeric match caused ASR=0.2% across ALL conditions (clean, blackbox, whitebox).

CORRECT APPROACH: IoU-based target presence check.
  - Load GT bbox for the target pedestrian at each frame
  - Check if ANY confirmed DeepSORT track bbox overlaps (IoU >= 0.3)
  - If nothing overlaps → target is not being tracked → attack succeeded

IoU threshold 0.3 (not 0.5) because:
  - We are checking presence, not quality
  - Adversarial frames may cause slightly shifted predictions
  - We want to distinguish "tracked loosely" from "completely lost"

Usage:
    python scripts\\evaluate_attack_success.py --model outputs\\best_model.zip
    python scripts\\evaluate_attack_success.py --model outputs\\best_model.zip --cpu
"""

import sys
import os

# ── Force CPU on low-VRAM GPUs BEFORE any src.* imports ──────────────────────
import torch
import types

def _resolve_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb >= 8.0:
            print(f"[eval] GPU: {torch.cuda.get_device_name(0)} "
                  f"({vram_gb:.1f}GB) — CUDA")
            return torch.device("cuda:0")
        print(f"[eval] GPU VRAM {vram_gb:.1f}GB < 8GB — forcing CPU")
    return torch.device("cpu")

_EVAL_DEVICE            = _resolve_device("--cpu" in sys.argv)
_dev_mod                = types.ModuleType("src.device")
_dev_mod.DEVICE         = _EVAL_DEVICE
_dev_mod.get_device     = lambda cfg=None: _EVAL_DEVICE
sys.modules["src.device"] = _dev_mod

sys.path.insert(0, os.path.abspath("."))

import argparse
import yaml
import numpy as np
import pandas as pd
from stable_baselines3 import PPO


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════

def bbox_iou(b1, b2):
    """IoU of two [x, y, w, h] boxes."""
    ix1 = max(b1[0], b2[0]);  iy1 = max(b1[1], b2[1])
    ix2 = min(b1[0]+b1[2], b2[0]+b2[2])
    iy2 = min(b1[1]+b1[3], b2[1]+b2[3])
    inter = max(0.0, ix2-ix1) * max(0.0, iy2-iy1)
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter / union if union > 0 else 0.0


def target_is_tracked(env, gt_bbox, iou_thresh=0.3):
    """
    Returns True if any confirmed DeepSORT track bbox overlaps with
    the GT target bbox at IoU >= iou_thresh.

    This is the CORRECT check — not GT_ID in DeepSORT_IDs.
    DeepSORT assigns its own sequential track IDs that have zero
    relationship to GT pedestrian IDs.
    """
    try:
        if env._extractor is None:
            return False
        tracks = [t for t in env._extractor.tracker.tracker.tracks
                  if t.is_confirmed()]
    except AttributeError:
        return False

    for t in tracks:
        pred_box = t.to_tlwh().tolist()
        if bbox_iou(gt_bbox, pred_box) >= iou_thresh:
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# GROUND TRUTH LOADING
# ══════════════════════════════════════════════════════════════════════════════

def find_best_target(seq_path, min_frames=80, min_visibility=0.6):
    """
    Returns (target_gt_id, target_bboxes_dict) where:
      target_gt_id        : GT pedestrian ID with the longest continuous run
      target_bboxes_dict  : {frame_no: [x, y, w, h]}

    Uses the pedestrian with the longest uninterrupted visible streak
    AND the largest average bounding box (bigger = more visible for attack).
    """
    gt_file = os.path.join(seq_path, "gt", "gt.txt")
    if not os.path.exists(gt_file):
        return None, {}

    cols = ["frame","id","x","y","w","h","active","class","visibility"]
    df   = pd.read_csv(gt_file, header=None, names=cols)
    df   = df[(df["active"]==1) & (df["class"]==1) &
              (df["visibility"] >= min_visibility)]

    best_id   = None
    best_score = 0.0

    for pid, grp in df.groupby("id"):
        frames = sorted(grp["frame"].tolist())
        streak = cur = 1
        for i in range(1, len(frames)):
            if frames[i] == frames[i-1] + 1:
                cur += 1
                streak = max(streak, cur)
            else:
                cur = 1

        if streak < min_frames:
            continue

        avg_area = float((grp["w"] * grp["h"]).mean())
        # Score: streak length + area bonus — longer + bigger = better target
        score = streak + avg_area * 0.01
        if score > best_score:
            best_score = score
            best_id    = int(pid)

    if best_id is None:
        # Fallback: take any pedestrian with the most frames
        if len(df) > 0:
            best_id = int(df.groupby("id").size().idxmax())
        else:
            return None, {}

    # Build per-frame bbox dict for this target
    tgt_df = df[df["id"] == best_id]
    bboxes = {int(row["frame"]): [row["x"], row["y"], row["w"], row["h"]]
              for _, row in tgt_df.iterrows()}

    print(f"  [target] GT ID={best_id}  "
          f"visible in {len(bboxes)} frames  "
          f"(streak score={best_score:.0f})")
    return best_id, bboxes


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE CONDITION
# ══════════════════════════════════════════════════════════════════════════════

def run_condition(seq_path, agent, label, target_bboxes,
                  deterministic=False, iou_thresh=0.3):
    """
    Runs one condition.
    agent=None → no defense (always T0).
    target_bboxes: {frame_no: [x,y,w,h]} from GT of the target pedestrian.
    """
    from src.mot_env import MOT17Env

    cfg = yaml.safe_load(open("config.yaml"))
    env = MOT17Env(seq_path,
                   w1=cfg["reward"]["w1"],
                   w2=cfg["reward"]["w2"],
                   w3=cfg["reward"]["w3"])

    obs, _ = env.reset()
    if env._extractor is not None:
        env._extractor.reset()

    total_target_frames = 0
    lost_frames         = 0
    total_id_sw         = 0
    action_counts       = {0: 0, 1: 0, 2: 0, 3: 0}
    frame_no            = 1
    done                = False

    try:
        while not done:
            if agent is not None:
                action, _ = agent.predict(obs, deterministic=deterministic)
                action = int(action)
            else:
                action = 0

            obs, _, done, _, info = env.step(action)
            total_id_sw          += int(info["id_switches"])
            action_counts[action] += 1

            # ── IoU-based target presence check ──────────────────────
            gt_bbox = target_bboxes.get(frame_no)
            if gt_bbox is not None:
                total_target_frames += 1
                if not target_is_tracked(env, gt_bbox, iou_thresh):
                    lost_frames += 1

            frame_no += 1

    finally:
        env.close()

    asr = lost_frames / max(total_target_frames, 1)

    return {
        "label":         label,
        "asr_pct":       round(asr * 100, 2),
        "id_switches":   total_id_sw,
        "lost":          lost_frames,
        "target_frames": total_target_frames,
        "action_dist":   action_counts,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="TRACE — ASR Evaluation")
    parser.add_argument("--model", default=None,
                        help="Model path. Example: --model outputs\\best_model.zip")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU evaluation.")
    parser.add_argument("--iou_thresh", type=float, default=0.3,
                        help="IoU threshold for target presence (default 0.3)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("config.yaml"))

    # ── Resolve model ─────────────────────────────────────────────────
    if args.model:
        model_path = args.model
    else:
        save_dir  = os.path.dirname(cfg["paths"]["model_save"])
        best      = os.path.join(save_dir, "best_model.zip")
        final     = cfg["paths"]["model_save"] + ".zip"
        model_path = best if os.path.exists(best) else final
        if not os.path.exists(model_path):
            print("[eval] ERROR: No model found. Pass --model <path>")
            sys.exit(1)

    print(f"[eval] Model  : {model_path}")
    agent = PPO.load(model_path, device=_EVAL_DEVICE)
    mode  = "deterministic" if args.deterministic else "stochastic"
    print(f"[eval] Policy : {mode}  |  IoU threshold: {args.iou_thresh}\n")

    # ── Sequence paths ────────────────────────────────────────────────
    parent    = os.path.dirname(cfg["data"]["seq_path"])
    seq_clean = cfg["data"]["seq_path"]
    seq_bb    = os.path.join(parent, "MOT17-04-Blackbox")
    seq_wb    = os.path.join(parent, "MOT17-04-Poisoned")

    # ── Find target pedestrian from CLEAN GT ──────────────────────────
    # We use the CLEAN GT for target selection because:
    #   1. Poisoned folders copy the clean GT (same pedestrian IDs + bboxes)
    #   2. The attacker targeted this same pedestrian
    #   3. Consistent target across all 6 conditions
    print("[eval] Selecting attack target from clean GT...")
    target_id, target_bboxes = find_best_target(seq_clean)

    if not target_bboxes:
        print("[eval] ERROR: Could not find a suitable attack target in GT.")
        print("       Check that gt/gt.txt exists and has class=1 pedestrians.")
        sys.exit(1)

    print(f"  [eval] Measuring tracking presence across "
          f"{len(target_bboxes)} target frames\n")

    # ── Build condition matrix ────────────────────────────────────────
    conditions = []

    if os.path.exists(seq_clean):
        conditions += [
            (seq_clean, None,  "Clean    | No Defense"),
            (seq_clean, agent, "Clean    | MTD Agent"),
        ]
    else:
        print(f"[skip] Clean sequence not found: {seq_clean}")

    if os.path.exists(seq_bb):
        conditions += [
            (seq_bb, None,  "Blackbox | No Defense"),
            (seq_bb, agent, "Blackbox | MTD Agent"),
        ]
    else:
        print(f"[skip] Blackbox not found: {seq_bb}")
        print(f"       python adversarial_attack_scripts\\generate_blackbox_attack.py")

    if os.path.exists(seq_wb):
        conditions += [
            (seq_wb, None,  "Whitebox | No Defense"),
            (seq_wb, agent, "Whitebox | MTD Agent"),
        ]
    else:
        print(f"[skip] Whitebox not found: {seq_wb}")
        print(f"       python adversarial_attack_scripts\\generate_whitebox_attack.py")

    if not conditions:
        print("[eval] No sequences found.")
        sys.exit(1)

    # ── Run ───────────────────────────────────────────────────────────
    print(f"[eval] Running {len(conditions)} conditions...\n")
    results = []
    for seq_path, ag, label in conditions:
        print(f"  {label}...", end=" ", flush=True)
        r = run_condition(seq_path, ag, label, target_bboxes,
                          deterministic=args.deterministic,
                          iou_thresh=args.iou_thresh)
        results.append(r)
        print(f"ASR={r['asr_pct']:.1f}%  "
              f"ID-sw={r['id_switches']}  "
              f"lost={r['lost']}/{r['target_frames']}")

    # ── Results table ─────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print(f"  {'Condition':<32} {'ASR ↓':>8}  {'ID-sw ↓':>8}  {'Lost/Total':>12}")
    print("─" * 70)
    for r in results:
        asr_color = ""
        print(f"  {r['label']:<32} {r['asr_pct']:>7.1f}%  "
              f"{r['id_switches']:>8d}  "
              f"{r['lost']:>6d}/{r['target_frames']:<6d}")
    print("═" * 70)

    # ── Key paper claims ──────────────────────────────────────────────
    def _get(lbl_a, lbl_b):
        for r in results:
            if lbl_a in r["label"] and lbl_b in r["label"]:
                return r
        return None

    wb_base = _get("Whitebox", "No Defense")
    wb_mtd  = _get("Whitebox", "MTD")
    bb_base = _get("Blackbox", "No Defense")
    bb_mtd  = _get("Blackbox", "MTD")

    print("\n  Key claims for paper:")
    if wb_base and wb_mtd:
        d = wb_base["asr_pct"] - wb_mtd["asr_pct"]
        print(f"  Whitebox ASR  : "
              f"{wb_base['asr_pct']:.1f}% → {wb_mtd['asr_pct']:.1f}%  "
              f"(MTD reduces by {d:.1f} pp)")
    if bb_base and bb_mtd:
        d = bb_base["asr_pct"] - bb_mtd["asr_pct"]
        print(f"  Blackbox ASR  : "
              f"{bb_base['asr_pct']:.1f}% → {bb_mtd['asr_pct']:.1f}%  "
              f"(MTD reduces by {d:.1f} pp)")
    if wb_mtd and bb_mtd:
        gap = wb_mtd["asr_pct"] - bb_mtd["asr_pct"]
        print(f"  WB vs BB gap on MTD: {gap:.1f} pp  "
              f"({'BPDA advantage confirmed' if gap > 5 else 'BPDA advantage marginal'})")

    # Action distribution for MTD conditions
    print("\n  MTD Agent action distribution:")
    for r in results:
        if "MTD" in r["label"] and sum(r["action_dist"].values()) > 0:
            total = sum(r["action_dist"].values())
            dist  = "  ".join(
                f"T{a}={100*c/total:.0f}%"
                for a, c in r["action_dist"].items()
            )
            print(f"    {r['label']:<32} {dist}")

    # ── Save ──────────────────────────────────────────────────────────
    out_dir   = os.path.dirname(cfg["paths"]["model_save"])
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "attack_success_rate.csv")
    df_out   = pd.DataFrame([{
        "label":         r["label"],
        "asr_pct":       r["asr_pct"],
        "id_switches":   r["id_switches"],
        "lost":          r["lost"],
        "target_frames": r["target_frames"],
    } for r in results])
    df_out.to_csv(csv_path, index=False)

    latex_path = os.path.join(out_dir, "asr_latex.txt")
    with open(latex_path, "w") as f:
        f.write("% ASR Table — paste into paper\n")
        f.write("\\hline\n")
        f.write("Condition & ASR (\\%) & ID-sw & Lost/Total \\\\\n")
        f.write("\\hline\n")
        for r in results:
            f.write(f"{r['label']} & {r['asr_pct']:.1f}\\% & "
                    f"{r['id_switches']} & "
                    f"{r['lost']}/{r['target_frames']} \\\\\n")
        f.write("\\hline\n")

    print(f"\n  CSV   → {csv_path}")
    print(f"  LaTeX → {latex_path}\n")