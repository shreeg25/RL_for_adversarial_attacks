# scripts/evaluate_accuracy.py
"""
Evaluates the trained MTD-PPO agent against MOT17-04 ground truth.
Reports MOTA, MOTP, ID F1, Precision, Recall, and per-action stats.

Metrics follow the MOTChallenge standard:
  MOTA = 1 - (FN + FP + ID_sw) / GT
  MOTP = mean IoU of matched detections
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO
from src.mot_env import MOT17Env


# ─── IoU helper ───────────────────────────────────────────────────────────────

def bbox_iou(b1, b2):
    """b1, b2: [x, y, w, h] → scalar IoU"""
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2 = min(b1[0]+b1[2], b2[0]+b2[2])
    y2 = min(b1[1]+b1[3], b2[1]+b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter / union if union > 0 else 0.0


# ─── Ground truth loader ──────────────────────────────────────────────────────

def load_ground_truth(seq_path: str) -> dict[int, list]:
    """
    Returns dict: frame_no (1-indexed) → list of [x, y, w, h] for
    all active pedestrians. Filters to class=1, visibility>=0.25.
    """
    gt_file = os.path.join(seq_path, "gt", "gt.txt")
    cols = ["frame","id","x","y","w","h","active","class","visibility"]
    df = pd.read_csv(gt_file, header=None, names=cols)
    df = df[(df["active"] == 1) & (df["class"] == 1) & (df["visibility"] >= 0.25)]
    gt = {}
    for frame_no, grp in df.groupby("frame"):
        gt[int(frame_no)] = grp[["x","y","w","h"]].values.tolist()
    return gt


# ─── Hungarian-style greedy matching ─────────────────────────────────────────

def match_detections(gt_boxes: list, pred_boxes: list, iou_thresh=0.5):
    """
    Greedy matching: highest IoU pairs first.
    Returns: matched_ious (list), n_fp (int), n_fn (int)
    """
    if not gt_boxes or not pred_boxes:
        return [], len(pred_boxes), len(gt_boxes)

    matched_gt  = set()
    matched_pred = set()
    pairs = []

    for i, g in enumerate(gt_boxes):
        for j, p in enumerate(pred_boxes):
            iou = bbox_iou(g, p)
            if iou >= iou_thresh:
                pairs.append((iou, i, j))

    pairs.sort(reverse=True)
    matched_ious = []
    for iou, i, j in pairs:
        if i not in matched_gt and j not in matched_pred:
            matched_gt.add(i)
            matched_pred.add(j)
            matched_ious.append(iou)

    fn = len(gt_boxes)  - len(matched_gt)
    fp = len(pred_boxes) - len(matched_pred)
    return matched_ious, fp, fn


# ─── Main evaluation loop ─────────────────────────────────────────────────────

def evaluate(model_path, deterministic=False):
    cfg = yaml.safe_load(open("config.yaml"))
    seq_path = cfg["data"]["seq_path"]

    env   = MOT17Env(seq_path)
    print(f"[eval] Loading TRACE defense from: {model_path}")
    model = PPO.load(model_path)
    gt    = load_ground_truth(seq_path)

    obs, _ = env.reset()

    # Accumulators
    total_gt       = 0
    total_tp       = 0
    total_fp       = 0
    total_fn       = 0
    total_id_sw    = 0
    total_iou_sum  = 0.0
    total_matched  = 0
    action_counts  = {0: 0, 1: 0, 2: 0, 3: 0}
    per_frame_rows = []

    frame_no = 1  # MOT17 is 1-indexed
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        action = int(action)
        obs, reward, done, _, info = env.step(action)
        action_counts[action] += 1

        # Ground truth boxes for this frame
        gt_boxes = gt.get(frame_no, [])

        # Get current tracker predictions from the extractor
        # (we re-use the env's internal extractor state)
        tracker_tracks = [
            t for t in env._extractor.tracker.tracker.tracks
            if t.is_confirmed()
        ]
        pred_boxes = []
        for t in tracker_tracks:
            tlwh = t.to_tlwh()
            pred_boxes.append(tlwh.tolist())

        # Match
        matched_ious, fp, fn = match_detections(gt_boxes, pred_boxes)
        tp = len(matched_ious)

        total_gt      += len(gt_boxes)
        total_tp      += tp
        total_fp      += fp
        total_fn      += fn
        total_id_sw   += info["id_switches"]
        total_iou_sum += sum(matched_ious)
        total_matched += tp

        per_frame_rows.append({
            "frame":       frame_no,
            "gt":          len(gt_boxes),
            "tp":          tp,
            "fp":          fp,
            "fn":          fn,
            "id_switches": info["id_switches"],
            "mean_iou":    np.mean(matched_ious) if matched_ious else 0.0,
            "action":      action,
            "reward":      reward,
        })
        frame_no += 1

    # ─── Compute summary metrics ──────────────────────────────────────
    mota = 1.0 - (total_fn + total_fp + total_id_sw) / max(total_gt, 1)
    motp = total_iou_sum / max(total_matched, 1)

    precision = total_tp / max(total_tp + total_fp, 1)
    recall    = total_tp / max(total_tp + total_fn, 1)

    # ID F1 (IDF1): harmonic mean of ID precision and recall
    # simplified: 2*TP / (2*TP + FP + FN) across the sequence
    idf1 = (2 * total_tp) / max(2 * total_tp + total_fp + total_fn, 1)

    total_frames = frame_no - 1
    action_labels = {0: "T0 clean", 1: "T1 warp", 2: "T2 noise", 3: "T3 cutout"}

    # ─── Print results ────────────────────────────────────────────────
    print("\n" + "═"*52)
    print("  MTD-PPO Evaluation  —  MOT17-04-FRCNN")
    print("═"*52)
    print(f"  Frames evaluated :  {total_frames}")
    print(f"  Total GT objects :  {total_gt}")
    print(f"  Mode             :  {'deterministic' if deterministic else 'stochastic'}")
    print("─"*52)
    print(f"  MOTA  ↑          :  {mota*100:6.2f}%")
    print(f"  MOTP  ↑          :  {motp*100:6.2f}%  (mean matched IoU)")
    print(f"  IDF1  ↑          :  {idf1*100:6.2f}%")
    print(f"  Precision ↑      :  {precision*100:6.2f}%")
    print(f"  Recall    ↑      :  {recall*100:6.2f}%")
    print("─"*52)
    print(f"  True Positives   :  {total_tp}")
    print(f"  False Positives  :  {total_fp}")
    print(f"  False Negatives  :  {total_fn}")
    print(f"  ID Switches  ↓   :  {total_id_sw}")
    print("─"*52)
    print("  Action distribution:")
    for a, count in action_counts.items():
        pct = 100 * count / total_frames
        bar = "█" * int(pct / 2)
        print(f"    {action_labels[a]:12s} {count:5d}  ({pct:5.1f}%)  {bar}")
    print("═"*52)

    # ─── Save per-frame CSV ───────────────────────────────────────────
    df = pd.DataFrame(per_frame_rows)
    out_dir = os.path.dirname(cfg["paths"]["model_save"])
    csv_path = os.path.join(out_dir, "eval_per_frame.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nPer-frame results saved → {csv_path}")

    return {
        "MOTA": mota, "MOTP": motp, "IDF1": idf1,
        "Precision": precision, "Recall": recall,
        "ID_switches": total_id_sw,
    }


# ─── Baseline comparison (T0 only, no agent) ─────────────────────────────────

def evaluate_baseline():
    """
    Runs the tracker with NO defense (always T0).
    Use this as your paper's baseline to show the agent adds value.
    """
    cfg = yaml.safe_load(open("config.yaml"))
    seq_path = cfg["data"]["seq_path"]

    env = MOT17Env(seq_path)
    gt  = load_ground_truth(seq_path)
    obs, _ = env.reset()

    total_gt, total_tp, total_fp, total_fn, total_id_sw = 0, 0, 0, 0, 0
    total_iou_sum, total_matched = 0.0, 0

    frame_no = 1
    done = False
    while not done:
        obs, reward, done, _, info = env.step(0)   # always T0
        gt_boxes = gt.get(frame_no, [])
        tracker_tracks = [
            t for t in env._extractor.tracker.tracker.tracks
            if t.is_confirmed()
        ]
        pred_boxes = [t.to_tlwh().tolist() for t in tracker_tracks]
        matched_ious, fp, fn = match_detections(gt_boxes, pred_boxes)
        tp = len(matched_ious)
        total_gt += len(gt_boxes); total_tp += tp
        total_fp += fp; total_fn += fn
        total_id_sw += info["id_switches"]
        total_iou_sum += sum(matched_ious); total_matched += tp
        frame_no += 1

    mota = 1.0 - (total_fn + total_fp + total_id_sw) / max(total_gt, 1)
    motp = total_iou_sum / max(total_matched, 1)
    idf1 = (2*total_tp) / max(2*total_tp + total_fp + total_fn, 1)

    print("\n" + "─"*52)
    print("  Baseline (T0 only — no defense)")
    print("─"*52)
    print(f"  MOTA  : {mota*100:6.2f}%")
    print(f"  MOTP  : {motp*100:6.2f}%")
    print(f"  IDF1  : {idf1*100:6.2f}%")
    print(f"  ID sw : {total_id_sw}")
    print("─"*52)

    return {"MOTA": mota, "MOTP": motp, "IDF1": idf1, "ID_switches": total_id_sw}


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--deterministic", action="store_true",
                        help="Use argmax policy instead of stochastic sampling")
    parser.add_argument("--baseline", action="store_true",
                        help="Also run T0-only baseline for comparison")
    parser.add_argument("--model", type=str, required=False, help="Path to specific PPO checkpoint")
    args = parser.parse_args()

    # Fallback to config path if no flag is provided
    cfg = yaml.safe_load(open("config.yaml"))
    target_model = args.model if args.model else cfg["paths"]["model_save"]

    agent_metrics = evaluate(model_path=target_model, deterministic=args.deterministic)

    if args.baseline:
        base_metrics = evaluate_baseline()
        print("\n  Delta (agent − baseline):")
        for k in ["MOTA", "MOTP", "IDF1"]:
            delta = (agent_metrics[k] - base_metrics[k]) * 100
            sign = "+" if delta >= 0 else ""
            print(f"    {k:10s}: {sign}{delta:.2f}%")
        sw_delta = agent_metrics["ID_switches"] - base_metrics["ID_switches"]
        print(f"    ID_switches: {sw_delta:+d}")