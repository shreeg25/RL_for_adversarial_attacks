# scripts/evaluate_multi_sequence.py
"""
TRACE — Multi-Sequence Evaluation
Runs the trained MTD-PPO agent across all configured sequences.

Usage:
    python scripts\evaluate_multi_sequence.py --model outputs\best_model.zip
    python scripts\evaluate_multi_sequence.py --model outputs\best_model.zip --deterministic

FIX-1  load_ground_truth + match_detections defined inline — no fragile import
FIX-2  --model CLI argument added (was hardcoded to wrong path)
FIX-3  Duplicate yaml import inside function removed
FIX-4  Sequence existence check before running
FIX-5  Safe tracker access with None guard
FIX-6  Consistent int formatting for ID_sw column
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import argparse
import yaml
import numpy as np
import pandas as pd
from stable_baselines3 import PPO


# ══════════════════════════════════════════════════════════════════════════════
# SELF-CONTAINED HELPERS  (no import from evaluate_accuracy)
# ══════════════════════════════════════════════════════════════════════════════

def load_ground_truth(seq_path: str) -> dict[int, list]:
    """
    Returns dict: frame_no → list of [x, y, w, h] for active pedestrians.
    Filters: active=1, class=1 (person), visibility >= 0.25
    """
    gt_file = os.path.join(seq_path, "gt", "gt.txt")
    if not os.path.exists(gt_file):
        print(f"  [warn] GT file not found: {gt_file}")
        return {}

    cols = ["frame", "id", "x", "y", "w", "h",
            "active", "class", "visibility"]
    df   = pd.read_csv(gt_file, header=None, names=cols)
    df   = df[(df["active"] == 1) &
              (df["class"]  == 1) &
              (df["visibility"] >= 0.25)]

    gt: dict[int, list] = {}
    for frame_no, grp in df.groupby("frame"):
        gt[int(frame_no)] = grp[["x", "y", "w", "h"]].values.tolist()
    return gt


def match_detections(
    gt_boxes:   list,
    pred_boxes: list,
    iou_thresh: float = 0.5,
) -> tuple[list, int, int]:
    """
    Greedy IoU matching (highest IoU first).
    Returns: matched_ious, n_fp, n_fn
    """
    if not gt_boxes or not pred_boxes:
        return [], len(pred_boxes), len(gt_boxes)

    matched_gt   = set()
    matched_pred = set()
    pairs        = []

    for i, g in enumerate(gt_boxes):
        for j, p in enumerate(pred_boxes):
            # IoU of two [x,y,w,h] boxes
            ix1 = max(g[0], p[0]);  iy1 = max(g[1], p[1])
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
            matched_gt.add(i)
            matched_pred.add(j)
            matched_ious.append(iou)

    fn = len(gt_boxes)   - len(matched_gt)
    fp = len(pred_boxes) - len(matched_pred)
    return matched_ious, fp, fn


def _get_confirmed_tracks(env) -> list:
    """
    Safe accessor for confirmed DeepSORT tracks.
    Returns empty list if extractor is None or internal state is unavailable.
    """
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

def evaluate_sequence(
    seq_path:     str,
    agent:        PPO,
    label:        str,
    deterministic: bool = False,
) -> dict:
    """
    Evaluates a single MOT sequence.
    Returns a results dict with metrics and raw counts for global aggregation.
    """
    from src.mot_env import MOT17Env

    cfg = yaml.safe_load(open("config.yaml"))
    env = MOT17Env(
        seq_path,
        w1=cfg["reward"]["w1"],
        w2=cfg["reward"]["w2"],
        w3=cfg["reward"]["w3"],
    )
    gt       = load_ground_truth(seq_path)
    obs, _   = env.reset()

    # Force tracker memory purge — prevents identity leakage across sequences
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
        env.close()   # stops prefetcher threads cleanly

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
            "gt":      s_gt,
            "tp":      s_tp,
            "fp":      s_fp,
            "fn":      s_fn,
            "id_sw":   s_id_sw,
            "iou_sum": s_iou_sum,
            "matched": s_matched,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="TRACE — Multi-Sequence Evaluation"
    )
    parser.add_argument(
        "--model", default=None,
        help="Path to model zip. Defaults to config paths.model_save. "
             "Example: --model outputs\\best_model.zip"
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Use argmax policy instead of stochastic sampling. "
             "Stochastic (default) is the live EOT defense."
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open("config.yaml"))

    # ── Resolve model path ────────────────────────────────────────────
    if args.model:
        model_path = args.model
    else:
        # Try best_model first (saved by EvalCallback), then final save
        save_dir   = os.path.dirname(cfg["paths"]["model_save"])
        best_path  = os.path.join(save_dir, "best_model.zip")
        final_path = cfg["paths"]["model_save"] + ".zip"
        if os.path.exists(best_path):
            model_path = best_path
            print(f"[eval] Using best_model: {best_path}")
        elif os.path.exists(final_path):
            model_path = final_path
            print(f"[eval] Using final model: {final_path}")
        else:
            print(f"[eval] ERROR: No model found. Pass --model <path>")
            sys.exit(1)

    print(f"[eval] Loading model: {model_path}")
    agent = PPO.load(model_path)
    mode  = "deterministic" if args.deterministic else "stochastic (EOT defense)"
    print(f"[eval] Policy mode : {mode}")

    # ── Resolve sequences ─────────────────────────────────────────────
    all_sequences = [cfg["data"]["seq_path"]] + \
                    cfg["data"].get("extra_sequences", [])

    # FIX-4: existence check before running — skip missing without crashing
    valid_sequences = []
    for seq in all_sequences:
        if os.path.exists(seq):
            valid_sequences.append(seq)
        else:
            print(f"  [skip] Sequence not found: {seq}")

    if not valid_sequences:
        print("[eval] ERROR: No valid sequences found. Check config.yaml paths.")
        sys.exit(1)

    print(f"\n[eval] Evaluating {len(valid_sequences)} sequence(s)...\n")

    # ── Per-sequence evaluation ───────────────────────────────────────
    all_results = []
    for seq in valid_sequences:
        label = os.path.basename(seq)
        print(f"  {label}...", end=" ", flush=True)
        r = evaluate_sequence(seq, agent, label,
                              deterministic=args.deterministic)
        all_results.append(r)
        print(f"MOTA={r['MOTA']:.1f}%  IDF1={r['IDF1']:.1f}%  "
              f"P={r['Precision']:.1f}%  R={r['Recall']:.1f}%  "
              f"ID-sw={r['ID_sw']}")

    # ── Micro-aggregate global metrics ────────────────────────────────
    # Sum raw counts across all sequences, then compute metrics once.
    # More rigorous than averaging per-sequence percentages.
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

    # Per-sequence std (for reporting variance, not for the global metric)
    df      = pd.DataFrame(all_results)
    metrics = ["MOTA", "MOTP", "IDF1", "Precision", "Recall"]
    means   = df[metrics].mean()
    stds    = df[metrics].std()

    # ── Print table ───────────────────────────────────────────────────
    COL_W = 22
    print("\n" + "═" * 84)
    header = f"  {'Sequence':<{COL_W}}"
    for m in metrics:
        header += f"  {m:>8}"
    header += f"  {'ID_sw':>8}"
    print(header)
    print("─" * 84)

    for r in all_results:
        row = f"  {r['sequence']:<{COL_W}}"
        for m in metrics:
            row += f"  {r[m]:>8.1f}"
        row += f"  {r['ID_sw']:>8d}"
        print(row)

    print("─" * 84)

    # Mean ± Std row
    ms_row = f"  {'Mean ± Std':<{COL_W}}"
    for m in metrics:
        ms_row += f"  {means[m]:>4.1f}±{stds[m]:<3.1f}"
    ms_row += f"  {int(df['ID_sw'].sum()):>8d}"
    print(ms_row)

    print("─" * 84)

    # Micro-aggregate row
    ga_row = f"  {'Global (micro-agg)':<{COL_W}}"
    for val in [g_mota, g_motp, g_idf1, g_prec, g_rec]:
        ga_row += f"  {val:>8.1f}"
    ga_row += f"  {g_id_sw:>8d}"
    print(ga_row)
    print("═" * 84)

    # ── Save outputs ──────────────────────────────────────────────────
    out_dir = os.path.dirname(cfg["paths"]["model_save"])
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "multi_sequence_results.csv")
    df.to_csv(csv_path, index=False)

    # LaTeX row for paper (micro-aggregated — this is the number to report)
    latex_path = os.path.join(out_dir, "multi_sequence_latex.txt")
    with open(latex_path, "w") as f:
        f.write("% Paste into your results table\n")
        f.write("% Metric & MOTA & MOTP & IDF1 & Precision & Recall & ID-sw \\\\\n")
        f.write(
            f"TRACE (Ours) & {g_mota:.1f}\\% & {g_motp:.1f}\\% & "
            f"{g_idf1:.1f}\\% & {g_prec:.1f}\\% & {g_rec:.1f}\\% & "
            f"{g_id_sw} \\\\\n"
        )
        f.write("\n% Per-sequence breakdown\n")
        for r in all_results:
            f.write(
                f"{r['sequence']} & {r['MOTA']:.1f}\\% & {r['MOTP']:.1f}\\% & "
                f"{r['IDF1']:.1f}\\% & {r['Precision']:.1f}\\% & "
                f"{r['Recall']:.1f}\\% & {r['ID_sw']} \\\\\n"
            )

    print(f"\n  CSV    → {csv_path}")
    print(f"  LaTeX  → {latex_path}")
    print()