# scripts/evaluate_multi_sequence.py
"""
Runs the trained MTD-PPO agent across all configured sequences
and reports mean ± std for each metric.

This directly addresses the "single sequence evaluation" rejection reason.
Minimum for IEEE Transactions: 2 sequences.
Recommended: 3+ sequences.
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))

import yaml
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from scripts.evaluate_accuracy import evaluate   # reuse your existing evaluator

def evaluate_sequence(seq_path: str, agent, label: str) -> dict:
    """Evaluates a single sequence while cleanly isolating tracker state memory."""
    import yaml
    cfg = yaml.safe_load(open("config.yaml"))

    from src.mot_env import MOT17Env
    from scripts.evaluate_accuracy import load_ground_truth, match_detections

    env = MOT17Env(
        seq_path,
        w1=cfg["reward"]["w1"],
        w2=cfg["reward"]["w2"],
        w3=cfg["reward"]["w3"]
    )
    gt = load_ground_truth(seq_path)
    obs, _ = env.reset()
    
    # Force absolute tracker memory purge to prevent identity leakage
    if hasattr(env, "_extractor") and env._extractor is not None:
        env._extractor.reset()

    s_gt, s_tp, s_fp, s_fn, s_id_sw = 0, 0, 0, 0, 0
    s_iou_sum, s_matched = 0.0, 0
    frame_no = 1
    done = False

    try:
        while not done:
            action, _ = agent.predict(obs, deterministic=False)
            obs, reward, done, _, info = env.step(int(action))

            gt_boxes = gt.get(frame_no, [])
            tracks = [t for t in env._extractor.tracker.tracker.tracks if t.is_confirmed()]
            pred_boxes = [t.to_tlwh().tolist() for t in tracks]

            matched_ious, fp, fn = match_detections(gt_boxes, pred_boxes)
            tp = len(matched_ious)

            s_gt      += len(gt_boxes)
            s_tp      += tp
            s_fp      += fp
            s_fn      += fn
            s_id_sw   += info["id_switches"]
            s_iou_sum += sum(matched_ious)
            s_matched += tp
            frame_no  += 1

    finally:
        # CRITICAL PERFORMANCE FIX: Force the background prefetch threads 
        # to kill their queues and close down cleanly before switching files
        env.close()

    mota      = 1.0 - (s_fn + s_fp + s_id_sw) / max(s_gt, 1)
    motp      = s_iou_sum / max(s_matched, 1)
    precision = s_tp / max(s_tp + s_fp, 1)
    recall    = s_tp / max(s_tp + s_fn, 1)
    idf1      = (2 * s_tp) / max(2 * s_tp + s_fp + s_fn, 1)

    return {
        "sequence":  label,
        "MOTA":      round(mota * 100, 2),
        "MOTP":      round(motp * 100, 2),
        "IDF1":      round(idf1 * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall":    round(recall * 100, 2),
        "ID_sw":     s_id_sw,
        "raw": {"gt": s_gt, "tp": s_tp, "fp": s_fp, "fn": s_fn, "id_sw": s_id_sw, "iou_sum": s_iou_sum, "matched": s_matched}
    }


if __name__ == "__main__":
    cfg   = yaml.safe_load(open("config.yaml"))
    agent = PPO.load(cfg["paths"]["model_save"])

    sequences = [cfg["data"]["seq_path"]] + \
                cfg["data"].get("extra_sequences", [])

    print(f"\n[*] Evaluating across {len(sequences)} sequences...\n")

    all_results = []
    for seq in sequences:
        label = os.path.basename(seq)
        print(f"  Sequence: {label}")
        r = evaluate_sequence(seq, agent, label)
        all_results.append(r)
        print(f"    MOTA={r['MOTA']}%  MOTP={r['MOTP']}%  "
              f"IDF1={r['IDF1']}%  ID-sw={r['ID_sw']}")

    # ── Aggregate stats ───────────────────────────────────────────────
    df      = pd.DataFrame(all_results)
    metrics = ["MOTA", "MOTP", "IDF1", "Precision", "Recall", "ID_sw"]

    print("\n" + "═" * 72)
    print(f"  {'Sequence':<28} " +
          "  ".join(f"{m:>9}" for m in metrics))
    print("─" * 72)
    for _, row in df.iterrows():
        print(f"  {row['sequence']:<28} " +
              "  ".join(f"{row[m]:>9.1f}" for m in metrics))
    print("─" * 72)

    means = df[metrics].mean()
    stds  = df[metrics].std()
    print(f"  {'Mean ± Std':<28} " +
          "  ".join(f"{means[m]:>6.1f}±{stds[m]:.1f}" for m in metrics))
    print("═" * 72)

    # ── Save ──────────────────────────────────────────────────────────
    out = os.path.join(
        os.path.dirname(cfg["paths"]["model_save"]),
        "multi_sequence_results.csv"
    )
    df.to_csv(out, index=False)

    # Also save the mean±std row for direct copy-paste into LaTeX
    summary_path = out.replace(".csv", "_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Metric & " + " & ".join(metrics) + " \\\\\n")
        f.write("MTD-PPO & " +
                " & ".join(f"{means[m]:.1f}$\\pm${stds[m]:.1f}"
                           for m in metrics) +
                " \\\\\n")

    print(f"\n  Saved → {out}")
    print(f"  LaTeX row → {summary_path}")