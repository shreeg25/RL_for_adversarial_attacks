# adversarial_attack_scripts/target_selector.py
"""
Analyzes MOT17 Ground Truth to find the optimal pedestrian target for an adversarial attack.
Prioritizes sequence length and visibility.
"""
import pandas as pd
import os

def find_optimal_target(seq_path: str, min_frames: int = 100, min_visibility: float = 0.8) -> dict:
    """
    Scans the gt.txt file to find the pedestrian ID with the longest uninterrupted, 
    highly visible presence in the video.
    """
    gt_file = os.path.join(seq_path, "gt", "gt.txt")
    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"Ground truth not found at {gt_file}")

    print(f"[*] Scanning Ground Truth: {gt_file}")
    
    # MOT17 GT format: frame, id, x, y, w, h, active, class, visibility
    cols = ["frame", "id", "x", "y", "w", "h", "active", "class", "visibility"]
    df = pd.read_csv(gt_file, header=None, names=cols)

    # Filter 1: Must be an active pedestrian (class == 1)
    df = df[(df["active"] == 1) & (df["class"] == 1)]

    # Filter 2: Must be highly visible (avoid targets hidden behind objects)
    df = df[df["visibility"] >= min_visibility]

    if df.empty:
        raise ValueError("No targets meet the visibility threshold.")

    # Group by pedestrian ID and calculate their lifespan
    target_stats = []
    for pid, group in df.groupby("id"):
        frames = group["frame"].sort_values().tolist()
        
        # We need continuous frames to run a stable attack
        # Let's find the longest consecutive streak of frames for this ID
        longest_streak = 0
        current_streak = 1
        start_frame = frames[0]
        best_start = frames[0]
        best_end = frames[0]

        for i in range(1, len(frames)):
            if frames[i] == frames[i-1] + 1:
                current_streak += 1
            else:
                if current_streak > longest_streak:
                    longest_streak = current_streak
                    best_start = start_frame
                    best_end = frames[i-1]
                current_streak = 1
                start_frame = frames[i]

        # Catch the final streak
        if current_streak > longest_streak:
            longest_streak = current_streak
            best_start = start_frame
            best_end = frames[-1]

        if longest_streak >= min_frames:
            # Calculate average bounding box area (larger is better for patches)
            avg_area = (group["w"] * group["h"]).mean()
            target_stats.append({
                "target_id": int(pid),
                "streak_length": int(longest_streak),
                "start_frame": int(best_start),
                "end_frame": int(best_end),
                "avg_area": float(avg_area)
            })

    if not target_stats:
        raise ValueError(f"No pedestrians found with >= {min_frames} consecutive visible frames.")

    # Sort candidates: Primary by streak length, Secondary by average bounding box size
    target_stats.sort(key=lambda x: (x["streak_length"], x["avg_area"]), reverse=True)
    
    optimal_target = target_stats[0]
    
    print("\n" + "═"*50)
    print("  Optimal Attack Target Acquired")
    print("═"*50)
    print(f"  Target ID    : {optimal_target['target_id']}")
    print(f"  Frame Window : {optimal_target['start_frame']} to {optimal_target['end_frame']}")
    print(f"  Total Frames : {optimal_target['streak_length']}")
    print("═"*50 + "\n")
    
    return optimal_target

if __name__ == "__main__":
    # Test the function directly
    import yaml
    try:
        cfg = yaml.safe_load(open("config.yaml"))
        seq_path = cfg["data"]["seq_path"]
        find_optimal_target(seq_path)
    except FileNotFoundError:
        print("Run this from the project root: python adversarial_attack_scripts/target_selector.py")