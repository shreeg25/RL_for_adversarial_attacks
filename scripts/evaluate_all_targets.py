# scripts/evaluate_all_targets.py
"""
Master Target-Specific Adversarial Evaluation Suite (4-Column Matrix).
Scans all sequences, auto-detects targets, and calculates the true defense margins 
across Clean (Col 1/2) and Poisoned (Col 3/4) tracking outputs.
"""

import os
import pandas as pd

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
DATA_DIR = "data/MOT17/train"
OUTPUTS_DIR = "outputs" 
# ==========================================

def auto_detect_target(gt_path, min_frames=100, min_visibility=0.8):
    cols = ["frame", "id", "x", "y", "w", "h", "active", "class", "visibility"]
    df = pd.read_csv(gt_path, header=None, names=cols)
    df = df[(df["active"] == 1) & (df["class"] == 1) & (df["visibility"] >= min_visibility)]

    if df.empty: return None

    target_stats = []
    for pid, group in df.groupby("id"):
        frames = group["frame"].sort_values().tolist()
        longest_streak, current_streak = 0, 1
        for i in range(1, len(frames)):
            if frames[i] == frames[i-1] + 1:
                current_streak += 1
            else:
                longest_streak = max(longest_streak, current_streak)
                current_streak = 1
        longest_streak = max(longest_streak, current_streak)

        if longest_streak >= min_frames:
            avg_area = (group["w"] * group["h"]).mean()
            target_stats.append({
                "target_id": int(pid),
                "streak_length": int(longest_streak),
                "avg_area": float(avg_area)
            })

    if not target_stats: return None
    target_stats.sort(key=lambda x: (x["streak_length"], x["avg_area"]), reverse=True)
    return target_stats[0]["target_id"]

def bb_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[0] + boxA[2], boxB[0] + boxB[2]), min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def evaluate_target(gt_path, track_path, target_id):
    if track_path is None or not os.path.exists(track_path):
        return None

    # Handle varying output types from DeepSORT (.csv or .txt)
    try:
        # Check if it has a header row (like standard pandas csv) or MOTChallenge format
        first_line = open(track_path, 'r').readline()
        if "frame" in first_line.lower():
            track_df = pd.read_csv(track_path)
        else:
            track_cols = ["frame", "track_id", "x", "y", "w", "h", "conf", "x1", "y1", "z1"]
            track_df = pd.read_csv(track_path, header=None, names=track_cols)
    except Exception:
        return None

    # Map column names if they differ slightly
    track_df.columns = [c.lower().strip() for c in track_df.columns]
    id_col = "track_id" if "track_id" in track_df.columns else ("id" if "id" in track_df.columns else track_df.columns[1])
    
    cols = ["frame", "id", "x", "y", "w", "h", "conf", "class", "vis"]
    gt_df = pd.read_csv(gt_path, header=None, names=cols)
    target_gt = gt_df[(gt_df["id"] == target_id) & (gt_df["class"] == 1)].sort_values("frame")
    
    frames_present = target_gt["frame"].unique()
    total_target_frames = len(frames_present)
    
    successful_matches = 0
    assigned_track_ids = []
    
    for frame in frames_present:
        gt_row = target_gt[target_gt["frame"] == frame].iloc[0]
        gt_box = [gt_row["x"], gt_row["y"], gt_row["w"], gt_row["h"]]
        frame_tracks = track_df[track_df["frame"] == frame]
        if frame_tracks.empty: continue
            
        best_iou, best_track_id = 0, -1
        for _, trk_row in frame_tracks.iterrows():
            trk_box = [trk_row["x"], trk_row["y"], trk_row["w"], trk_row["h"]]
            iou = bb_iou(gt_box, trk_box)
            if iou > best_iou:
                best_iou, best_track_id = iou, trk_row[id_col]
                
        if best_iou >= 0.3:
            successful_matches += 1
            if not assigned_track_ids or assigned_track_ids[-1] != best_track_id:
                assigned_track_ids.append(best_track_id)

    survival_rate = (successful_matches / total_target_frames) * 100
    return survival_rate

def find_track_file(base_seq, prefix, attack_type=None):
    """
    Hunts through the flat outputs folder for the exact matching tracking file.
    Prefix limits to col1_, col2_, col3_, col4_
    """
    if not os.path.exists(OUTPUTS_DIR):
        return None

    for f in os.listdir(OUTPUTS_DIR):
        if f.startswith(prefix) and base_seq in f and f.endswith(".txt"):
            if attack_type:
                # Cols 3 and 4 should contain the attack type
                if attack_type in f: return os.path.join(OUTPUTS_DIR, f)
            else:
                # Cols 1 and 2 shouldn't be the blackbox/whitebox file
                if "Whitebox" not in f and "Blackbox" not in f:
                    return os.path.join(OUTPUTS_DIR, f)
    return None

if __name__ == "__main__":
    print("\n" + "="*105)
    print(f"{'TARGET-ISOLATED SURVIVAL RATE MATRIX (CLEAN vs. POISONED vs. DEFENDED)':^105}")
    print("="*105)
    print(f"{'Condition':<25} | {'Target ID':<10} | {'Col 1 (Cln Base)':<16} | {'Col 2 (Cln Agt)':<15} | {'Col 3 (Pois Base)':<17} | {'Col 4 (Pois Agt)':<16}")
    print("-" * 105)

    sequences = sorted([d for d in os.listdir(DATA_DIR) if "Whitebox" in d or "Blackbox" in d])
    
    if not sequences:
        print("[!] No Whitebox or Blackbox sequences found in data directory.")
        exit(1)

    for seq in sequences:
        gt_path = os.path.join(DATA_DIR, seq, "gt", "gt.txt")
        if not os.path.exists(gt_path): continue

        target_id = auto_detect_target(gt_path)
        if target_id is None: continue

        # Extract base sequence and attack type
        parts = seq.rsplit("-", 1)
        if len(parts) != 2: continue
        base_seq, attack_type = parts[0], parts[1]

        # Dynamically find all 4 columns
        c1_path = find_track_file(base_seq, "col1_")
        c2_path = find_track_file(base_seq, "col2_")
        c3_path = find_track_file(base_seq, "col3_", attack_type)
        c4_path = find_track_file(base_seq, "col4_", attack_type)

        c1_surv = evaluate_target(gt_path, c1_path, target_id)
        c2_surv = evaluate_target(gt_path, c2_path, target_id)
        c3_surv = evaluate_target(gt_path, c3_path, target_id)
        c4_surv = evaluate_target(gt_path, c4_path, target_id)

        def fmt_surv(val):
            return f"{val:>5.1f}%" if val is not None else "   N/A  "

        condition = f"{base_seq} ({attack_type})"
        print(f"{condition:<25} | {target_id:<10} | {fmt_surv(c1_surv):<16} | {fmt_surv(c2_surv):<15} | {fmt_surv(c3_surv):<17} | {fmt_surv(c4_surv):<16}")

    print("="*105)