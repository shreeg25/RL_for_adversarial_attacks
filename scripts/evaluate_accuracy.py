# scripts/evaluate_accuracy.py
import sys
import os
import argparse
import yaml
import numpy as np
import pandas as pd
import types
import torch
from stable_baselines3 import PPO

# ── VRAM check MUST happen before any src.* imports ──────────────────────────
def _resolve_eval_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb >= 8.0:
            print(f"[eval] GPU: {torch.cuda.get_device_name(0)}  ({vram_gb:.1f}GB) — using CUDA")
            return torch.device("cuda:0")
        else:
            print(f"[eval] GPU VRAM: {vram_gb:.1f}GB < 8GB — forcing CPU")
            return torch.device("cpu")
    return torch.device("cpu")

_force_cpu   = "--cpu" in sys.argv
_EVAL_DEVICE = _resolve_eval_device(_force_cpu)

_dev_mod            = types.ModuleType("src.device")
_dev_mod.DEVICE     = _EVAL_DEVICE
_dev_mod.get_device = lambda cfg=None: _EVAL_DEVICE
sys.modules["src.device"] = _dev_mod

sys.path.insert(0, os.path.abspath("."))

# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def load_ground_truth(seq_path: str) -> dict:
    gt_file = os.path.join(seq_path, "gt", "gt.txt")
    if not os.path.exists(gt_file):
        return {}
    cols = ["frame", "id", "x", "y", "w", "h", "active", "class", "visibility"]
    df   = pd.read_csv(gt_file, header=None, names=cols)
    df   = df[(df["active"] == 1) & (df["class"] == 1) & (df["visibility"] >= 0.25)]
    gt   = {}
    for frame_no, grp in df.groupby("frame"):
        gt[int(frame_no)] = grp[["x", "y", "w", "h"]].values.tolist()
    return gt

def bbox_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2 = min(b1[0]+b1[2], b2[0]+b2[2])
    y2 = min(b1[1]+b1[3], b2[1]+b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter / union if union > 0 else 0.0

def match_detections(gt_boxes, pred_boxes, iou_thresh=0.5):
    if not gt_boxes or not pred_boxes:
        return [], len(pred_boxes), len(gt_boxes)
    matched_gt = set(); matched_pred = set(); pairs = []
    for i, g in enumerate(gt_boxes):
        for j, p in enumerate(pred_boxes):
            iou = bbox_iou(g, p)
            if iou >= iou_thresh:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    matched_ious = []
    for iou, i, j in pairs:
        if i not in matched_gt and j not in matched_pred:
            matched_gt.add(i); matched_pred.add(j)
            matched_ious.append(iou)
    fn = len(gt_boxes)  - len(matched_gt)
    fp = len(pred_boxes) - len(matched_pred)
    return matched_ious, fp, fn

def _get_confirmed_tracks(env):
    try:
        if env._extractor is None: return []
        return [t for t in env._extractor.tracker.tracker.tracks if t.is_confirmed()]
    except AttributeError:
        return []

def _compute_metrics(s_gt, s_tp, s_fp, s_fn, s_id_sw, s_iou_sum, s_matched):
    mota      = 1.0 - (s_fn + s_fp + s_id_sw) / max(s_gt, 1)
    motp      = s_iou_sum / max(s_matched, 1)
    precision = s_tp / max(s_tp + s_fp, 1)
    recall    = s_tp / max(s_tp + s_fn, 1)
    idf1      = (2 * s_tp) / max(2 * s_tp + s_fp + s_fn, 1)
    return {
        "MOTA": round(mota * 100, 2), "MOTP": round(motp * 100, 2),
        "IDF1": round(idf1 * 100, 2), "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2), "ID_sw": s_id_sw,
        "raw": {"gt": s_gt, "tp": s_tp, "fp": s_fp, "fn": s_fn, "id_sw": s_id_sw, "iou_sum": s_iou_sum, "matched": s_matched}
    }

# ══════════════════════════════════════════════════════════════════════════════
# CORE RUNNER 
# ══════════════════════════════════════════════════════════════════════════════
def run_sequence(seq_path, agent=None, deterministic=False, output_dir=None, run_label=""):
    from src.mot_env import MOT17Env
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    
    cfg = yaml.safe_load(open("config.yaml"))
    
    # 1. Initialize the raw base environment
    raw_env = MOT17Env(
        seq_path, w_rec=cfg["reward"]["w_rec"], w_fp=cfg["reward"]["w_fp"],
        w_lost=cfg["reward"]["w_lost"], w_cost=cfg["reward"]["w_cost"],
    )

    # 2. Wrap it in the Memory Buffer so the agent gets its 48D state
    vec_env = DummyVecEnv([lambda: raw_env])
    vec_env = VecFrameStack(vec_env, n_stack=4)

    gt = load_ground_truth(seq_path)
    
    # Reset the vectorized environment
    obs = vec_env.reset()
    if raw_env._extractor is not None: raw_env._extractor.reset()

    s_gt = s_tp = s_fp = s_fn = s_id_sw = 0
    s_iou_sum = 0.0; s_matched = 0
    frame_no = 1; done = False
    
    frame_stats = []
    raw_tracks = []
    try:
        while not done:
            # Predict using the 48D stacked observation
            if agent is None:
                action = 0
            else:
                # predict returns a tuple of (actions_array, states). 
                actions, _ = agent.predict(obs, deterministic=deterministic)
                # Extract the actual integer from the first environment's action array
                action = int(actions[0].item())
            
            # Step the vectorized environment (returns arrays/lists)
            obs, reward, done_arr, info_arr = vec_env.step([action])
            
            # Unpack the vectorized outputs
            done = done_arr[0]
            info = info_arr[0]

            # Query the raw environment directly for tracking metrics
            gt_boxes = gt.get(frame_no, [])
            tracks = _get_confirmed_tracks(raw_env)
            pred_boxes = [t.to_tlwh().tolist() for t in tracks]
            for t in tracks:
                bbox = t.to_tlwh() # [x, y, w, h]
                # MOT Format: frame, id, x, y, w, h, conf, -1, -1, -1
                raw_tracks.append([frame_no, t.track_id, bbox[0], bbox[1], bbox[2], bbox[3], 1.0, -1, -1, -1])

            matched_ious, fp, fn = match_detections(gt_boxes, pred_boxes)
            tp = len(matched_ious)
            sw = int(info.get("id_switches", 0))

            s_gt += len(gt_boxes); s_tp += tp; s_fp += fp; s_fn += fn; s_id_sw += sw
            s_iou_sum += sum(matched_ious); s_matched += tp
            
            frame_stats.append({
                "frame": frame_no, "gt_count": len(gt_boxes), "tp": tp, "fp": fp, "fn": fn, "id_sw": sw, "action_taken": action
            })
            frame_no += 1
    finally:
        vec_env.close()

    # Save per-frame CSV
    if output_dir and run_label:
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(frame_stats)
        df.to_csv(os.path.join(output_dir, f"{run_label}_per_frame.csv"), index=False)
        
        track_file = os.path.join(output_dir, f"{run_label}_tracks.txt")
        if len(raw_tracks) > 0:
            np.savetxt(track_file, np.array(raw_tracks), fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d')
        else:
            open(track_file, 'w').close() # Create empty file if no tracks

    return _compute_metrics(s_gt, s_tp, s_fp, s_fn, s_id_sw, s_iou_sum, s_matched)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
METRICS = ["MOTA", "MOTP", "IDF1", "Precision", "Recall"]
COL_W, MET_W = 22, 8

def _print_row(label, r):
    row = f"  {label:<{COL_W}}" + "".join(f"  {r[m]:>{MET_W}.1f}" for m in METRICS) + f"  {r['ID_sw']:>{MET_W}d}"
    print(row)

def _print_global(label, raw_list):
    g = _compute_metrics(
        sum(r["gt"] for r in raw_list), sum(r["tp"] for r in raw_list), sum(r["fp"] for r in raw_list),
        sum(r["fn"] for r in raw_list), sum(r["id_sw"] for r in raw_list), sum(r["iou_sum"] for r in raw_list), sum(r["matched"] for r in raw_list)
    )
    _print_row(label, g)
    return g

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("config.yaml"))
    model_path = args.model if args.model else cfg["paths"]["model_save"] + ".zip"
    
    agent = PPO.load(model_path, device=_EVAL_DEVICE)
    mode = "deterministic" if args.deterministic else "stochastic (EOT defense)"
    out_dir = os.path.dirname(cfg["paths"]["model_save"])

    # EXPLICIT FILTER: Only sequences that DO NOT have Whitebox/Blackbox in the name are Clean
    all_seqs = [cfg["data"]["seq_path"]] + cfg["data"].get("train_sequences", []) + cfg["data"].get("extra_sequences", [])
    valid_clean = list(set([s for s in all_seqs if os.path.exists(s) and "Whitebox" not in s and "Blackbox" not in s]))
    valid_clean.sort()

    print(f"\n[eval] Discovered {len(valid_clean)} explicit CLEAN sequences.")

    # ── COLUMN 1: Clean + T0 ──
    print("\n=== COLUMN 1 — Clean Data | T0-only Baseline ===")
    col1_results = []
    for seq in valid_clean:
        label = os.path.basename(seq)
        print(f"  {label}...", end=" ", flush=True)
        r = run_sequence(seq, agent=None, output_dir=out_dir, run_label=f"col1_{label}")
        r["sequence"] = label
        col1_results.append(r)
        print(f"MOTA={r['MOTA']:.1f}%")
    col1_global = _print_global("Global C1", [r["raw"] for r in col1_results])

    # ── COLUMN 2: Clean + Agent ──
    print("\n=== COLUMN 2 — Clean Data | MTD-PPO Agent ===")
    col2_results = []
    for seq in valid_clean:
        label = os.path.basename(seq)
        print(f"  {label}...", end=" ", flush=True)
        r = run_sequence(seq, agent=agent, deterministic=args.deterministic, output_dir=out_dir, run_label=f"col2_{label}")
        r["sequence"] = label
        col2_results.append(r)
        print(f"MOTA={r['MOTA']:.1f}%")
    col2_global = _print_global("Global C2", [r["raw"] for r in col2_results])

    # ── SUMMARY TEXT BUILDER (Starts Here) ──
    summary_lines = [
        "TRACE — Accuracy Evaluation Summary",
        "=" * 60,
        f"  Policy mode : {mode}",
        f"  Sequences   : {len(valid_clean)}\n",
        "COLUMN 1 — Clean | T0 Baseline",
        *[f"  {m:<12}: {col1_global[m]:.2f}%" for m in METRICS],
        f"  {'ID_sw':<12}: {col1_global['ID_sw']}\n",
        "COLUMN 2 — Clean | MTD-PPO Agent",
        *[f"  {m:<12}: {col2_global[m]:.2f}%" for m in METRICS],
        f"  {'ID_sw':<12}: {col2_global['ID_sw']}\n"
    ]

    # ── COLUMNS 3 & 4: Attacks ──
    for attack in ["Whitebox", "Blackbox"]:
        poisoned_pairs = [(s, f"{s}-{attack}", os.path.basename(s)) for s in valid_clean if os.path.exists(f"{s}-{attack}")]
        if not poisoned_pairs: continue

        print(f"\n=== COLUMN 3 — {attack} Poisoned | T0-only Baseline ===")
        col3_results = []
        for _, p_seq, label in poisoned_pairs:
            print(f"  {label}...", end=" ", flush=True)
            r = run_sequence(p_seq, agent=None, output_dir=out_dir, run_label=f"col3_{attack}_{label}")
            r["sequence"] = label
            col3_results.append(r)
            print(f"MOTA={r['MOTA']:.1f}%")
        col3_global = _print_global(f"Global C3 {attack}", [r["raw"] for r in col3_results])

        print(f"\n=== COLUMN 4 — {attack} Poisoned | MTD-PPO Agent ===")
        col4_results = []
        for _, p_seq, label in poisoned_pairs:
            print(f"  {label}...", end=" ", flush=True)
            r = run_sequence(p_seq, agent=agent, deterministic=args.deterministic, output_dir=out_dir, run_label=f"col4_{attack}_{label}")
            r["sequence"] = label
            col4_results.append(r)
            print(f"MOTA={r['MOTA']:.1f}%")
        col4_global = _print_global(f"Global C4 {attack}", [r["raw"] for r in col4_results])

        summary_lines.extend([
            f"COLUMN 3 — {attack} Poisoned | T0 Baseline",
            *[f"  {m:<12}: {col3_global[m]:.2f}%" for m in METRICS],
            f"  {'ID_sw':<12}: {col3_global['ID_sw']}\n",
            f"COLUMN 4 — {attack} Poisoned | MTD-PPO Agent",
            *[f"  {m:<12}: {col4_global[m]:.2f}%" for m in METRICS],
            f"  {'ID_sw':<12}: {col4_global['ID_sw']}\n",
            f"DEFENSE GAIN ({attack})",
            *[f"  {m:<12}: {(col4_global[m] - col3_global[m]):+.2f}%" for m in METRICS],
            f"  {'ID_sw':<12}: {(col4_global['ID_sw'] - col3_global['ID_sw']):+d}\n"
        ])

    # ── Final Summary Write ──
    txt_path = os.path.join(out_dir, "accuracy_evaluation_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    
    print(f"\n[eval] Done. Per-frame CSVs and full summary saved to {out_dir}")