# scripts/evaluate_multi_sequence.py
"""
Runs the trained MTD-PPO agent across all configured sequences
and reports exact accumulated metrics for IEEE Transactions formatting.

Bypasses sequential identity leakage by explicitly forcing DeepSORT resets.
"""
import sys
import os
import yaml
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

<<<<<<< HEAD
sys.path.insert(0, os.path.abspath("."))

from src.mot_env import MOT17Env
from scripts.evaluate_accuracy import load_ground_truth, match_detections

def evaluate_sequence(seq_path: str, agent, label: str) -> dict:
    """Evaluates a single sequence while cleanly isolating tracker state memory."""
    env = MOT17Env(seq_path)
    gt = load_ground_truth(seq_path)
    obs, _ = env.reset()
    
    # Force absolute tracker memory purge to prevent multi-sequence identity leakage
    if hasattr(env, "_extractor") and env._extractor is not None:
        env._extractor.reset()

    # Sequence Accumulators
=======
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

>>>>>>> f7d9a2d00e5750e95f0850b4f95e628c783a4ae5
    s_gt, s_tp, s_fp, s_fn, s_id_sw = 0, 0, 0, 0, 0
    s_iou_sum, s_matched = 0.0, 0
    frame_no = 1
    done = False

    try:
        while not done:
            action, _ = agent.predict(obs, deterministic=False)
            obs, reward, done, _, info = env.step(int(action))

<<<<<<< HEAD
        gt_boxes = gt.get(frame_no, [])
        tracks = [t for t in env._extractor.tracker.tracker.tracks if t.is_confirmed()]
        pred_boxes = [t.to_tlwh().tolist() for t in tracks]
=======
            gt_boxes = gt.get(frame_no, [])
            tracks = [t for t in env._extractor.tracker.tracker.tracks if t.is_confirmed()]
            pred_boxes = [t.to_tlwh().tolist() for t in tracks]
>>>>>>> f7d9a2d00e5750e95f0850b4f95e628c783a4ae5

            matched_ious, fp, fn = match_detections(gt_boxes, pred_boxes)
            tp = len(matched_ious)

<<<<<<< HEAD
        s_gt      += len(gt_boxes)
        s_tp      += tp
        s_fp      += fp
        s_fn      += fn
        s_id_sw   += info["id_switches"]
        s_iou_sum += sum(matched_ious)
        s_matched += tp
        frame_no  += 1

    # Safe sequence evaluation calculations
=======
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

>>>>>>> f7d9a2d00e5750e95f0850b4f95e628c783a4ae5
    mota      = 1.0 - (s_fn + s_fp + s_id_sw) / max(s_gt, 1)
    motp      = s_iou_sum / max(s_matched, 1)
    precision = s_tp / max(s_tp + s_fp, 1)
    recall    = s_tp / max(s_tp + s_fn, 1)
    idf1      = (2 * s_tp) / max(2 * s_tp + s_fp + s_fn, 1)

    return {
        "sequence": label,
        "MOTA": round(mota * 100, 2),
        "MOTP": round(motp * 100, 2),
        "IDF1": round(idf1 * 100, 2),
        "Precision": round(precision * 100, 2),
<<<<<<< HEAD
        "Recall": round(recall * 100, 2),
        "ID_sw": s_id_sw,
=======
        "Recall":    round(recall * 100, 2),
        "ID_sw":     s_id_sw,
>>>>>>> f7d9a2d00e5750e95f0850b4f95e628c783a4ae5
        "raw": {"gt": s_gt, "tp": s_tp, "fp": s_fp, "fn": s_fn, "id_sw": s_id_sw, "iou_sum": s_iou_sum, "matched": s_matched}
    }

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml"))
    agent = PPO.load(cfg["paths"]["model_save"])

    sequences = [cfg["data"]["seq_path"]] + cfg["data"].get("extra_sequences", [])
    print(f"\n[*] Commencing Rigorous Multi-Sequence Evaluation Array ({len(sequences)} tracks)...")

    all_results = []
    for seq in sequences:
        label = os.path.basename(seq)
        r = evaluate_sequence(seq, agent, label)
        all_results.append(r)
        print(f"    -> {label:18s} | MOTA: {r['MOTA']:.1f}% | IDF1: {r['IDF1']:.1f}% | ID-sw: {r['ID_sw']}")

    # ── Rigorous Global Dataset Metric Accumulation ────────────────────────
    df = pd.DataFrame(all_results)
    
    g_gt = sum(r["raw"]["gt"] for r in all_results)
    g_tp = sum(r["raw"]["tp"] for r in all_results)
    g_fp = sum(r["raw"]["fp"] for r in all_results)
    g_fn = sum(r["raw"]["fn"] for r in all_results)
    g_id_sw = sum(r["raw"]["id_sw"] for r in all_results)
    g_iou_sum = sum(r["raw"]["iou_sum"] for r in all_results)
    g_matched = sum(r["raw"]["matched"] for r in all_results)

    global_mota = (1.0 - (g_fn + g_fp + g_id_sw) / max(g_gt, 1)) * 100
    global_motp = (g_iou_sum / max(g_matched, 1)) * 100
    global_idf1 = ((2 * g_tp) / max(2 * g_tp + g_fp + g_fn, 1)) * 100
    global_prec = (g_tp / max(g_tp + g_fp, 1)) * 100
    global_rec  = (g_tp / max(g_tp + g_fn, 1)) * 100

    metrics = ["MOTA", "MOTP", "IDF1", "Precision", "Recall", "ID_sw"]

    print("\n" + "═" * 82)
    print(f"  {'Sequence/Dataset':<24} " + "  ".join(f"{m:>8}" for m in metrics))
    print("─" * 82)
    for _, row in df.iterrows():
        print(f"  {row['sequence']:<24} " + "  ".join(f"{row[m]:>8.1f}" for m in metrics[:-1]) + f"  {int(row['ID_sw']):8d}")
    print("─" * 82)
    print(f"  {'Micro-Overhead Total':<24} {global_mota:8.1f}  {global_motp:8.1f}  {global_idf1:8.1f}  {global_prec:8.1f}  {global_rec:8.1f}  {g_id_sw:8d}")
    print("═" * 82)

    # ── Save Outputs ──────────────────────────────────────────────────
    out_dir = os.path.dirname(cfg["paths"]["model_save"])
    df.to_csv(os.path.join(out_dir, "multi_sequence_results.csv"), index=False)
    
    summary_path = os.path.join(out_dir, "multi_sequence_results_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Metric & MOTA & MOTP & IDF1 & Precision & Recall & ID-sw \\\\\n")
        f.write(f"TRACE (Ours) & {global_mota:.1f}\\% & {global_motp:.1f}\\% & {global_idf1:.1f}\\% & {global_prec:.1f}\\% & {global_rec:.1f}\\% & {g_id_sw} \\\\\n")

    print(f"\n[SUCCESS] Micro-aggregated LaTeX Row saved → {summary_path}")