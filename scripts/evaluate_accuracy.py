# scripts/evaluate_accuracy.py
"""
TRACE — Three-Column Accuracy Evaluation

Produces the core paper result table across every configured sequence:

  Column 1 | Clean  + T0 only (no attack, no defense)   → your "clean tracker" baseline
  Column 2 | Poisoned + T0 only (attack lands, no defense) → shows attack is effective
  Column 3 | Poisoned + MTD-PPO agent (attack + defense)   → shows TRACE works

The gap between column 2 and column 3 is the defense gain your paper reports.

Poisoned sequences are auto-discovered by looking for sibling folders named
  <SEQ_NAME>-Whitebox  and  <SEQ_NAME>-Blackbox
next to each clean sequence. Missing poisoned folders are skipped with a warning.

Usage:
    # Full three-column evaluation (whitebox + blackbox poisoned sequences):
    python scripts/evaluate_accuracy.py --model outputs/best_model.zip

    # Deterministic policy:
    python scripts/evaluate_accuracy.py --model outputs/best_model.zip --deterministic

    # Force CPU (e.g. low-VRAM GPU):
    python scripts/evaluate_accuracy.py --model outputs/best_model.zip --cpu

    # Only run the clean baseline (no poisoned sequences needed):
    python scripts/evaluate_accuracy.py --model outputs/best_model.zip --clean-only

Metrics (MOTChallenge standard):
    MOTA  = 1 - (FN + FP + ID_sw) / GT
    MOTP  = mean IoU of matched detections
    IDF1  = 2*TP / (2*TP + FP + FN)
"""

import sys
import os

# ── VRAM check MUST happen before any src.* imports ──────────────────────────
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
            print(f"[eval] GPU VRAM: {vram_gb:.1f}GB < 8GB — forcing CPU")
            return torch.device("cpu")
    print("[eval] No CUDA GPU — using CPU")
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
# SHARED HELPERS
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
        if env._extractor is None:
            return []
        return [t for t in env._extractor.tracker.tracker.tracks
                if t.is_confirmed()]
    except AttributeError:
        return []


def _compute_metrics(s_gt, s_tp, s_fp, s_fn, s_id_sw, s_iou_sum, s_matched):
    mota      = 1.0 - (s_fn + s_fp + s_id_sw) / max(s_gt, 1)
    motp      = s_iou_sum / max(s_matched, 1)
    precision = s_tp / max(s_tp + s_fp, 1)
    recall    = s_tp / max(s_tp + s_fn, 1)
    idf1      = (2 * s_tp) / max(2 * s_tp + s_fp + s_fn, 1)
    return {
        "MOTA":      round(mota      * 100, 2),
        "MOTP":      round(motp      * 100, 2),
        "IDF1":      round(idf1      * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall":    round(recall    * 100, 2),
        "ID_sw":     s_id_sw,
        "raw": {
            "gt": s_gt, "tp": s_tp, "fp": s_fp, "fn": s_fn,
            "id_sw": s_id_sw, "iou_sum": s_iou_sum, "matched": s_matched,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# CORE RUNNER — one sequence, one policy (agent or T0-always)
# ══════════════════════════════════════════════════════════════════════════════

def run_sequence(seq_path, agent=None, deterministic=False):
    """
    Runs one sequence through the environment.

    agent=None  → T0-always baseline (always action 0, no defense).
    agent=PPO   → use the trained MTD-PPO policy.

    Returns raw accumulator dict for micro-aggregation.
    """
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
    s_iou_sum = 0.0; s_matched = 0
    frame_no = 1; done = False

    try:
        while not done:
            if agent is None:
                action = 0   # T0 always — baseline
            else:
                action, _ = agent.predict(obs, deterministic=deterministic)
                action = int(action)

            obs, reward, done, _, info = env.step(action)

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

    return _compute_metrics(s_gt, s_tp, s_fp, s_fn, s_id_sw, s_iou_sum, s_matched)


# ══════════════════════════════════════════════════════════════════════════════
# POISONED SEQUENCE DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def find_poisoned_sequence(clean_seq_path: str, attack_type: str) -> str | None:
    """
    Looks for  <parent>/<SEQ_NAME>-Whitebox  or  <parent>/<SEQ_NAME>-Blackbox
    next to the clean sequence directory.

    attack_type: "Whitebox" | "Blackbox"
    Returns the full path if it exists, else None.
    """
    seq_name = os.path.basename(clean_seq_path)
    parent   = os.path.dirname(clean_seq_path)
    poisoned_path = os.path.join(parent, f"{seq_name}-{attack_type}")
    if os.path.exists(poisoned_path):
        return poisoned_path
    return None


# ══════════════════════════════════════════════════════════════════════════════
# PRINTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

METRICS = ["MOTA", "MOTP", "IDF1", "Precision", "Recall"]
COL_W   = 22   # sequence name column width
MET_W   = 8    # metric column width


def _print_section_header(title: str):
    width = COL_W + 2 + (MET_W + 2) * len(METRICS) + MET_W + 2
    print("\n" + "═" * width)
    print(f"  {title}")
    print("─" * width)
    hdr = f"  {'Sequence':<{COL_W}}" + "".join(f"  {m:>{MET_W}}" for m in METRICS) + f"  {'ID_sw':>{MET_W}}"
    print(hdr)
    print("─" * width)


def _print_row(label: str, r: dict):
    row = (f"  {label:<{COL_W}}" +
           "".join(f"  {r[m]:>{MET_W}.1f}" for m in METRICS) +
           f"  {r['ID_sw']:>{MET_W}d}")
    print(row)


def _print_global(label: str, raw_list: list):
    g_gt      = sum(r["gt"]      for r in raw_list)
    g_tp      = sum(r["tp"]      for r in raw_list)
    g_fp      = sum(r["fp"]      for r in raw_list)
    g_fn      = sum(r["fn"]      for r in raw_list)
    g_id_sw   = sum(r["id_sw"]   for r in raw_list)
    g_iou_sum = sum(r["iou_sum"] for r in raw_list)
    g_matched = sum(r["matched"] for r in raw_list)
    g = _compute_metrics(g_gt, g_tp, g_fp, g_fn, g_id_sw, g_iou_sum, g_matched)
    _print_row(label, g)
    return g


def _section_footer():
    width = COL_W + 2 + (MET_W + 2) * len(METRICS) + MET_W + 2
    print("═" * width)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="TRACE — Three-Column Accuracy Evaluation"
    )
    parser.add_argument("--model", default=None,
                        help="Path to trained PPO model zip.")
    parser.add_argument("--deterministic", action="store_true",
                        help="Argmax policy. Default: stochastic (EOT defense).")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU evaluation regardless of GPU VRAM.")
    parser.add_argument("--clean-only", action="store_true",
                        help="Only run clean baseline columns — skip poisoned evaluation.")
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
    mode  = "deterministic" if args.deterministic else "stochastic (EOT defense)"
    print(f"[eval] Policy mode: {mode}\n")

    # ── Collect all clean sequences ───────────────────────────────────
    all_clean = ([cfg["data"]["seq_path"]] +
                 cfg["data"].get("extra_sequences", []))
    valid_clean = [s for s in all_clean if os.path.exists(s)]
    if not valid_clean:
        print("[eval] ERROR: No valid sequences found.")
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════════════
    # COLUMN 1: Clean + T0 baseline
    # ═══════════════════════════════════════════════════════════════════
    print(f"[eval] Running COLUMN 1 — Clean sequences, T0 baseline ({len(valid_clean)} seq)...")
    _print_section_header("COLUMN 1 — Clean Data | T0-only Baseline (No Attack, No Defense)")

    col1_results = []
    for seq in valid_clean:
        label = os.path.basename(seq)
        print(f"  {label}...", end=" ", flush=True)
        r = run_sequence(seq, agent=None)
        r["sequence"] = label
        col1_results.append(r)
        print(f"MOTA={r['MOTA']:.1f}%  IDF1={r['IDF1']:.1f}%  ID-sw={r['ID_sw']}")
        _print_row(label, r)

    print("─" + "─" * (COL_W + 1 + (MET_W + 2) * len(METRICS) + MET_W + 1))
    col1_global = _print_global("Global (micro-agg)", [r["raw"] for r in col1_results])
    _section_footer()

    # ═══════════════════════════════════════════════════════════════════
    # COLUMN 2: Clean + MTD-PPO agent
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n[eval] Running COLUMN 2 — Clean sequences, MTD-PPO agent ({len(valid_clean)} seq)...")
    _print_section_header("COLUMN 2 — Clean Data | MTD-PPO Agent (No Attack, With Defense)")

    col2_results = []
    for seq in valid_clean:
        label = os.path.basename(seq)
        print(f"  {label}...", end=" ", flush=True)
        r = run_sequence(seq, agent=agent, deterministic=args.deterministic)
        r["sequence"] = label
        col2_results.append(r)
        print(f"MOTA={r['MOTA']:.1f}%  IDF1={r['IDF1']:.1f}%  ID-sw={r['ID_sw']}")
        _print_row(label, r)

    print("─" + "─" * (COL_W + 1 + (MET_W + 2) * len(METRICS) + MET_W + 1))
    col2_global = _print_global("Global (micro-agg)", [r["raw"] for r in col2_results])
    _section_footer()

    if not args.clean_only:

        for attack_type in ["Whitebox", "Blackbox"]:

            # ── Discover poisoned sequences ───────────────────────────
            poisoned_pairs = []  # list of (clean_path, poisoned_path, label)
            for seq in valid_clean:
                p = find_poisoned_sequence(seq, attack_type)
                if p:
                    poisoned_pairs.append((seq, p, os.path.basename(seq)))
                else:
                    print(f"  [warn] No {attack_type} poisoned folder for "
                          f"{os.path.basename(seq)} — skipping")

            if not poisoned_pairs:
                print(f"\n[eval] No {attack_type} poisoned sequences found. "
                      f"Run generate_{attack_type.lower()}_attack.py first.")
                continue

            n = len(poisoned_pairs)

            # ═══════════════════════════════════════════════════════════
            # COLUMN 3: Poisoned + T0 (attack lands, no defense)
            # ═══════════════════════════════════════════════════════════
            print(f"\n[eval] Running COLUMN 3 ({attack_type}) — Poisoned, T0 baseline ({n} seq)...")
            _print_section_header(
                f"COLUMN 3 ({attack_type}) — Poisoned Data | T0-only (Attack Lands, No Defense)"
            )

            col3_results = []
            for clean_seq, pois_seq, label in poisoned_pairs:
                print(f"  {label}...", end=" ", flush=True)
                r = run_sequence(pois_seq, agent=None)
                r["sequence"] = label
                col3_results.append(r)
                print(f"MOTA={r['MOTA']:.1f}%  IDF1={r['IDF1']:.1f}%  ID-sw={r['ID_sw']}")
                _print_row(label, r)

            print("─" + "─" * (COL_W + 1 + (MET_W + 2) * len(METRICS) + MET_W + 1))
            col3_global = _print_global("Global (micro-agg)", [r["raw"] for r in col3_results])
            _section_footer()

            # ═══════════════════════════════════════════════════════════
            # COLUMN 4: Poisoned + MTD-PPO (attack + defense = paper result)
            # ═══════════════════════════════════════════════════════════
            print(f"\n[eval] Running COLUMN 4 ({attack_type}) — Poisoned, MTD-PPO agent ({n} seq)...")
            _print_section_header(
                f"COLUMN 4 ({attack_type}) — Poisoned Data | MTD-PPO Agent (Attack + Defense)"
            )

            col4_results = []
            for clean_seq, pois_seq, label in poisoned_pairs:
                print(f"  {label}...", end=" ", flush=True)
                r = run_sequence(pois_seq, agent=agent, deterministic=args.deterministic)
                r["sequence"] = label
                col4_results.append(r)
                print(f"MOTA={r['MOTA']:.1f}%  IDF1={r['IDF1']:.1f}%  ID-sw={r['ID_sw']}")
                _print_row(label, r)

            print("─" + "─" * (COL_W + 1 + (MET_W + 2) * len(METRICS) + MET_W + 1))
            col4_global = _print_global("Global (micro-agg)", [r["raw"] for r in col4_results])
            _section_footer()

            # ═══════════════════════════════════════════════════════════
            # DEFENSE GAIN SUMMARY (col4 - col3 = paper delta)
            # ═══════════════════════════════════════════════════════════
            width = COL_W + 2 + (MET_W + 2) * len(METRICS) + MET_W + 2
            print(f"\n  {'─'*width}")
            print(f"  DEFENSE GAIN ({attack_type})  — MTD-PPO vs No-Defense on Poisoned Data")
            print(f"  (Positive = defense recovers metric; Negative ID-sw = fewer switches)")
            print(f"  {'─'*width}")
            for m in METRICS:
                delta = col4_global[m] - col3_global[m]
                sign  = "+" if delta >= 0 else ""
                print(f"    {m:<12}: {sign}{delta:.2f}%")
            sw_delta = col4_global["ID_sw"] - col3_global["ID_sw"]
            print(f"    {'ID_sw':<12}: {sw_delta:+d}")
            print(f"  {'─'*width}")

            # ── Save per-attack CSV ───────────────────────────────────
            out_dir = os.path.dirname(cfg["paths"]["model_save"])
            os.makedirs(out_dir, exist_ok=True)

            rows = []
            for r3, (_, _, label) in zip(col3_results, poisoned_pairs):
                r4 = next(x for x in col4_results if x["sequence"] == label)
                r1 = next(x for x in col1_results if x["sequence"] == label)
                rows.append({
                    "sequence":          label,
                    "attack_type":       attack_type,
                    **{f"clean_baseline_{m}":   r1[m] for m in METRICS},
                    "clean_baseline_ID_sw":      r1["ID_sw"],
                    **{f"poisoned_nodefense_{m}": r3[m] for m in METRICS},
                    "poisoned_nodefense_ID_sw":   r3["ID_sw"],
                    **{f"poisoned_mtdppo_{m}":    r4[m] for m in METRICS},
                    "poisoned_mtdppo_ID_sw":       r4["ID_sw"],
                })
            df = pd.DataFrame(rows)
            csv_path = os.path.join(out_dir,
                                    f"accuracy_{attack_type.lower()}_comparison.csv")
            df.to_csv(csv_path, index=False)
            print(f"\n  CSV saved → {csv_path}")

    # ═══════════════════════════════════════════════════════════════════
    # MASTER SUMMARY TEXT REPORT
    # ═══════════════════════════════════════════════════════════════════
    out_dir = os.path.dirname(cfg["paths"]["model_save"])
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, "accuracy_evaluation_summary.txt")

    lines = []
    lines.append("TRACE — Accuracy Evaluation Summary")
    lines.append("=" * 60)
    lines.append(f"  Policy mode : {mode}")
    lines.append(f"  Sequences   : {len(valid_clean)}")
    lines.append("")
    lines.append("COLUMN 1 — Clean | T0 Baseline (Global micro-agg)")
    for m in METRICS:
        lines.append(f"  {m:<12}: {col1_global[m]:.2f}%")
    lines.append(f"  {'ID_sw':<12}: {col1_global['ID_sw']}")
    lines.append("")
    lines.append("COLUMN 2 — Clean | MTD-PPO Agent (Global micro-agg)")
    for m in METRICS:
        lines.append(f"  {m:<12}: {col2_global[m]:.2f}%")
    lines.append(f"  {'ID_sw':<12}: {col2_global['ID_sw']}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n[eval] Summary saved → {txt_path}")
    print("[eval] Done.\n")