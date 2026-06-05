# scripts/evaluate_multi_sequence.py
"""
TRACE — Multi-Sequence Evaluation

Usage:
    python scripts\evaluate_multi_sequence.py --model outputs\best_model.zip
    python scripts\evaluate_multi_sequence.py --model outputs\best_model.zip --deterministic
    python scripts\evaluate_multi_sequence.py --model outputs\best_model.zip --cpu

FIX: Auto-detects GPU VRAM. If < 8GB (e.g. RTX 4050 6GB), forces CPU
     evaluation automatically. This prevents the hang caused by:
       - embedder_gpu=True loading MobileNet ReID onto GPU
       - FramePrefetcher pushing full frame tensors to GPU
       - Both competing for 6GB VRAM simultaneously → CUDA stall

     PPO MLP is 3→64→64→4 = 4,500 params. CPU inference is microseconds.
     DeepSORT ReID on CPU takes ~15ms/frame — fine for one-time evaluation.
     Evaluation correctness > evaluation speed.
"""

import sys
import os

# ── VRAM check MUST happen before any src.* imports ──────────────────────────
# src.device, src.mot_env, src.state_extractor all read DEVICE at import time.
# Patching sys.modules["src.device"] here ensures they all see CPU on low-VRAM.

import torch
import types

def _resolve_eval_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb >= 8.0:
            print(f"[eval] GPU: {torch.cuda.get_device_name(0)}  "
                  f"({vram_gb:.1f}GB) — using CUDA")
            return torch.device("cuda:0")
        else:
            print(f"[eval] GPU VRAM: {vram_gb:.1f}GB < 8GB  "
                  f"(RTX 4050 detected) — forcing CPU to prevent hang")
            return torch.device("cpu")
    print("[eval] No CUDA GPU — using CPU")
    return torch.device("cpu")

# Parse --cpu early, before argparse fully runs
_force_cpu = "--cpu" in sys.argv
_EVAL_DEVICE = _resolve_eval_device(_force_cpu)

# Inject patched device module BEFORE any src imports
# This means state_extractor will init with embedder_gpu=False on CPU
# and FramePrefetcher will not push GPU tensors
_dev_mod = types.ModuleType("src.device")
_dev_mod.DEVICE     = _EVAL_DEVICE
_dev_mod.get_device = lambda cfg=None: _EVAL_DEVICE
sys.modules["src.device"] = _dev_mod

# Now safe to add project root and do remaining imports
sys.path.insert(0, os.path.abspath("."))

import argparse
import yaml
import numpy as np
import pandas as pd
from stable_baselines3 import PPO


# ══════════════════════════════════════════════════════════════════════════════
# SELF-CONTAINED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_ground_truth(seq_path: str) -> dict:
    gt_file = os.path.join(seq_path, "gt", "gt.txt")
    if not os.path.exists(gt_file):
        print(f"  [warn] GT not found: {gt_file}")
        return {}
    cols = ["frame", "id", "x", "y", "w", "h", "active", "class", "visibility"]
    df   = pd.read_csv(gt_file, header=None, names=cols)
    df   = df[(df["active"] == 1) & (df["class"] == 1) & (df["visibility"] >= 0.25)]
    gt   = {}
    for frame_no, grp in df.groupby("frame"):
        gt[int(frame_no)] = grp[["x", "y", "w", "h"]].values.tolist()
    return gt


def match_detections(gt_boxes, pred_boxes, iou_thresh=0.5):
    if not gt_boxes or not pred_boxes:
        return [], len(pred_boxes), len(gt_boxes)

    matched_gt = set(); matched_pred = set(); pairs = []
    for i, g in enumerate(gt_boxes):
        for j, p in enumerate(pred_boxes):
            ix1 = max(g[0], p[0]); iy1 = max(g[1], p[1])
            ix2 = min(g[0]+g[2], p[0]+p[2])
            iy2 = min(g[1]+g[3], p[1]+p[3])
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            union = g[2]*g[3] + p[2]*p[3] - inter
            iou   = inter / union if union > 0 else 0.0
            if iou >= iou_thresh:
                pairs.append((iou, i, j))

    pairs.sort(reverse=True)
    matched_ious = []
    for iou, i, j in pairs:
        if i not in matched_gt and j not in matched_pred:
            matched_gt.add(i); matched_pred.add(j)
            matched_ious.append(iou)

    return matched_ious, len(pred_boxes)-len(matched_pred), len(gt_boxes)-len(matched_gt)


def _get_confirmed_tracks(env) -> list:
    try:
        if env._extractor is None:
            return []
        return [t for t in env._extractor.tracker.tracker.tracks
                if t.is_confirmed()]
    except AttributeError:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# PER-SEQUENCE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_sequence(seq_path, agent, label, deterministic=False):
    # Deferred import — src.device is already patched by this point
    from src.mot_env import MOT17Env

    cfg = yaml.safe_load(open("config.yaml"))
    env = MOT17Env(seq_path,
                   w1=cfg["reward"]["w1"],
                   w2=cfg["reward"]["w2"],
                   w3=cfg["reward"]["w3"])

    gt     = load_ground_truth(seq_path)
    obs, _ = env.reset()

    if env._extractor is not None:
        env._extractor.reset()

    s_gt = s_tp = s_fp = s_fn = s_id_sw = 0
    s_iou_sum = 0.0
    s_matched = 0
    frame_no  = 1
    done      = False

    try:
        while not done:
            action, _ = agent.predict(obs, deterministic=deterministic)
            obs, reward, done, _, info = env.step(int(action))

            gt_boxes   = gt.get(frame_no, [])
            tracks     = _get_confirmed_tracks(env)
            pred_boxes = [t.to_tlwh().tolist() for t in tracks]

            matched_ious, fp, fn = match_detections(gt_boxes, pred_boxes)
            tp = len(matched_ious)

            s_gt      += len(gt_boxes)
            s_tp      += tp
            s_fp      += fp
            s_fn      += fn
            s_id_sw   += int(info["id_switches"])
            s_iou_sum += sum(matched_ious)
            s_matched += tp
            frame_no  += 1

    finally:
        env.close()

    mota      = 1.0 - (s_fn + s_fp + s_id_sw) / max(s_gt, 1)
    motp      = s_iou_sum / max(s_matched, 1)
    precision = s_tp / max(s_tp + s_fp, 1)
    recall    = s_tp / max(s_tp + s_fn, 1)
    idf1      = (2 * s_tp) / max(2 * s_tp + s_fp + s_fn, 1)

    return {
        "sequence":  label,
        "MOTA":      round(mota      * 100, 2),
        "MOTP":      round(motp      * 100, 2),
        "IDF1":      round(idf1      * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall":    round(recall    * 100, 2),
        "ID_sw":     s_id_sw,
        "Frames":    frame_no - 1,
        "raw": {
            "gt":      s_gt,  "tp":      s_tp,
            "fp":      s_fp,  "fn":      s_fn,
            "id_sw":   s_id_sw,
            "iou_sum": s_iou_sum,
            "matched": s_matched,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="TRACE — Multi-Sequence Evaluation")
    parser.add_argument("--model", default=None,
                        help="Path to model. Example: --model outputs\\best_model.zip")
    parser.add_argument("--deterministic", action="store_true",
                        help="Argmax policy. Default: stochastic (EOT defense).")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU evaluation regardless of GPU VRAM.")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("config.yaml"))

    # ── Resolve model path ────────────────────────────────────────────
    if args.model:
        model_path = args.model
    else:
        save_dir   = os.path.dirname(cfg["paths"]["model_save"])
        best_path  = os.path.join(save_dir, "best_model.zip")
        final_path = cfg["paths"]["model_save"] + ".zip"
        if os.path.exists(best_path):
            model_path = best_path
            print(f"[eval] Using best_model:  {best_path}")
        elif os.path.exists(final_path):
            model_path = final_path
            print(f"[eval] Using final model: {final_path}")
        else:
            print("[eval] ERROR: No model found. Pass --model <path>")
            sys.exit(1)

    # Load model onto the resolved eval device (CPU on 6GB GPU)
    print(f"[eval] Loading model on {_EVAL_DEVICE}...")
    agent = PPO.load(model_path, device=_EVAL_DEVICE)
    mode  = "deterministic" if args.deterministic else "stochastic (EOT defense)"
    print(f"[eval] Policy mode: {mode}\n")

    # ── Resolve sequences ─────────────────────────────────────────────
    all_seqs = ([cfg["data"]["seq_path"]] +
                cfg["data"].get("extra_sequences", []))

    valid_seqs = []
    for seq in all_seqs:
        if os.path.exists(seq):
            valid_seqs.append(seq)
        else:
            print(f"  [skip] Not found: {seq}")

    if not valid_seqs:
        print("[eval] ERROR: No valid sequences found.")
        sys.exit(1)

    print(f"[eval] {len(valid_seqs)} sequence(s) to evaluate...\n")

    # ── Run evaluation ────────────────────────────────────────────────
    all_results = []
    for seq in valid_seqs:
        label = os.path.basename(seq)
        print(f"  {label}...", end=" ", flush=True)
        r = evaluate_sequence(seq, agent, label, deterministic=args.deterministic)
        all_results.append(r)
        print(f"MOTA={r['MOTA']:.1f}%  IDF1={r['IDF1']:.1f}%  "
              f"P={r['Precision']:.1f}%  R={r['Recall']:.1f}%  "
              f"ID-sw={r['ID_sw']}")

    # ── Micro-aggregate ───────────────────────────────────────────────
    g_gt      = sum(r["raw"]["gt"]      for r in all_results)
    g_tp      = sum(r["raw"]["tp"]      for r in all_results)
    g_fp      = sum(r["raw"]["fp"]      for r in all_results)
    g_fn      = sum(r["raw"]["fn"]      for r in all_results)
    g_id_sw   = sum(r["raw"]["id_sw"]   for r in all_results)
    g_iou_sum = sum(r["raw"]["iou_sum"] for r in all_results)
    g_matched = sum(r["raw"]["matched"] for r in all_results)

    g_mota = (1.0 - (g_fn + g_fp + g_id_sw) / max(g_gt, 1)) * 100
    g_motp = (g_iou_sum / max(g_matched, 1)) * 100
    g_idf1 = ((2 * g_tp) / max(2 * g_tp + g_fp + g_fn, 1)) * 100
    g_prec = (g_tp / max(g_tp + g_fp, 1)) * 100
    g_rec  = (g_tp / max(g_tp + g_fn, 1)) * 100

    df      = pd.DataFrame(all_results)
    metrics = ["MOTA", "MOTP", "IDF1", "Precision", "Recall"]
    means   = df[metrics].mean()
    stds    = df[metrics].std()

    # ── Print table ───────────────────────────────────────────────────
    W = 22
    print("\n" + "═" * 84)
    hdr = f"  {'Sequence':<{W}}" + "".join(f"  {m:>8}" for m in metrics) + f"  {'ID_sw':>8}"
    print(hdr)
    print("─" * 84)

    for r in all_results:
        row = (f"  {r['sequence']:<{W}}" +
               "".join(f"  {r[m]:>8.1f}" for m in metrics) +
               f"  {r['ID_sw']:>8d}")
        print(row)

    print("─" * 84)
    ms = (f"  {'Mean ± Std':<{W}}" +
          "".join(f"  {means[m]:>4.1f}±{stds[m]:<3.1f}" for m in metrics) +
          f"  {int(df['ID_sw'].sum()):>8d}")
    print(ms)

    print("─" * 84)
    ga = (f"  {'Global (micro-agg)':<{W}}" +
          "".join(f"  {v:>8.1f}" for v in [g_mota, g_motp, g_idf1, g_prec, g_rec]) +
          f"  {g_id_sw:>8d}")
    print(ga)
    print("═" * 84)

    # ── Save ──────────────────────────────────────────────────────────
    out_dir = os.path.dirname(cfg["paths"]["model_save"])
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "multi_sequence_results.csv")
    df.to_csv(csv_path, index=False)

    latex_path = os.path.join(out_dir, "multi_sequence_latex.txt")
    with open(latex_path, "w") as f:
        f.write("% Paste into your results table\n")
        f.write("% Metric & MOTA & MOTP & IDF1 & Precision & Recall & ID-sw \\\\\n")
        f.write(f"TRACE (Ours) & {g_mota:.1f}\\% & {g_motp:.1f}\\% & "
                f"{g_idf1:.1f}\\% & {g_prec:.1f}\\% & {g_rec:.1f}\\% & "
                f"{g_id_sw} \\\\\n\n% Per-sequence breakdown\n")
        for r in all_results:
            f.write(f"{r['sequence']} & {r['MOTA']:.1f}\\% & {r['MOTP']:.1f}\\% & "
                    f"{r['IDF1']:.1f}\\% & {r['Precision']:.1f}\\% & "
                    f"{r['Recall']:.1f}\\% & {r['ID_sw']} \\\\\n")

    print(f"\n  CSV   → {csv_path}")
    print(f"  LaTeX → {latex_path}\n")