# adversarial_attack_scripts/generate_blackbox_attack.py
"""
Black-Box Physical EOT + PGD Attack Generator.

Attacker assumptions (blackbox):
  - Does NOT know the defense architecture — no BPDA wrapper
  - Does NOT know which action the RL agent will select
  - Only knows a person detector (Faster R-CNN) is present
  - Uses PhysicalRenderer for EOT to simulate real-world variance

This is the ablation counterpart to generate_whitebox_attack.py.
The whitebox vs blackbox ASR gap in evaluate_attack_success.py
directly quantifies how much the BPDA knowledge advantage is worth.

Fixes applied vs whitebox baseline:
  Same FIX-1..8 as whitebox, MINUS the BPDA wrapper and action cycling.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import shutil
import yaml
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
import numpy as np
import pandas as pd

from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from src.mot_env import FramePrefetcher
from adversarial_attack_scripts.target_selector import find_optimal_target
from adversarial_attack_scripts.generate_whitebox_attack import (
    PhysicalRenderer,
    _get_target_score,
)

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_EOT   = 10
EPSILON = 1.0
ALPHA   = 0.05
ITERS   = 40


# ── PGD + EOT (no BPDA, no action knowledge) ─────────────────────────────────

def optimize_patch_blackbox(
    model:    torch.nn.Module,
    frame:    torch.Tensor,        # (1, C, H, W) float32 on device
    box:      list,                # [x1, y1, w, h]
    renderer: PhysicalRenderer,
    epsilon:  float = EPSILON,
    alpha:    float = ALPHA,
    iters:    int   = ITERS,
    n_eot:    int   = N_EOT,
) -> torch.Tensor:
    """
    Blackbox PGD: gradients flow only through the detector.
    No BPDA — the attacker has no knowledge of any defense layer.
    EOT still uses PhysicalRenderer to maintain physical realism.
    """
    device = frame.device
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h

    patch_data = torch.empty(1, 3, h, w, device=device).uniform_(-epsilon, epsilon)

    for iteration in range(iters):
        patch_data.requires_grad_(True)
        accum_grad = torch.zeros_like(patch_data)
        total_loss = 0.0
        n_valid    = 0

        for eot_idx in range(n_eot):

            # Physical rendering — same as whitebox, no defense layer after
            phys_patch = renderer.apply(patch_data, bbox_w=w, bbox_h=h)

            poisoned = frame.clone()
            poisoned[:, :, y1:y2, x1:x2] = torch.clamp(
                poisoned[:, :, y1:y2, x1:x2] + phys_patch, 0.0, 1.0
            )

            # ── No BPDA here — raw detector, no defense knowledge ─────
            preds   = model([poisoned[0]])[0]
            t_score = _get_target_score(preds, x1, y1, x2, y2)

            if t_score is None:
                continue

            t_score.backward(retain_graph=False)

            if patch_data.grad is not None:
                accum_grad = accum_grad + patch_data.grad.detach().clone()
                patch_data.grad.zero_()
                total_loss += t_score.item()
                n_valid    += 1

        if n_valid == 0:
            patch_data = patch_data.detach()
            print(f"    iter {iteration+1:>3d}  target suppressed — early stop")
            break

        avg_grad   = accum_grad / n_valid
        patch_data = patch_data.detach() - alpha * avg_grad.sign()
        patch_data = torch.clamp(patch_data, -epsilon, epsilon)

        if (iteration + 1) % 10 == 0:
            print(f"    iter {iteration+1:>3d}/{iters}  "
                  f"avg_loss={total_loss/n_valid:.4f}  "
                  f"valid={n_valid}/{n_eot}")

    return patch_data.detach()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    cfg      = yaml.safe_load(open("config.yaml"))
    seq_path = cfg["data"]["seq_path"]
    parent   = os.path.dirname(seq_path)

    out_base    = os.path.join(parent, "MOT17-04-Blackbox")
    out_img_dir = os.path.join(out_base, "img1")
    out_gt_dir  = os.path.join(out_base, "gt")
    out_det_dir = os.path.join(out_base, "det")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_gt_dir,  exist_ok=True)
    os.makedirs(out_det_dir, exist_ok=True)

    print("[*] Copying GT and det files...")
    shutil.copy(os.path.join(seq_path, "gt",  "gt.txt"),
                os.path.join(out_gt_dir,  "gt.txt"))
    shutil.copy(os.path.join(seq_path, "det", "det.txt"),
                os.path.join(out_det_dir, "det.txt"))

    target  = find_optimal_target(seq_path, min_frames=100, min_visibility=0.8)
    tid     = target["target_id"]
    s_frame = target["start_frame"]
    e_frame = target["end_frame"]

    print(f"\n[*] Blackbox attack — Target ID={tid}  "
          f"frames {s_frame}→{e_frame}  N_EOT={N_EOT}  (no BPDA)")

    cols   = ["frame","id","x","y","w","h","active","class","visibility"]
    df_gt  = pd.read_csv(os.path.join(seq_path, "gt", "gt.txt"),
                         header=None, names=cols)
    tgt_gt = df_gt[df_gt["id"] == tid]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Loading Faster R-CNN on {device}...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    ).to(device).eval()

    clean_img_dir = os.path.join(seq_path, "img1")
    all_frames    = sorted(os.listdir(clean_img_dir))
    print(f"[*] Copying {len(all_frames)} clean frames as base...")
    for fname in all_frames:
        dst = os.path.join(out_img_dir, fname)
        if not os.path.exists(dst):
            shutil.copy(os.path.join(clean_img_dir, fname), dst)

    prefetcher = FramePrefetcher(
        img_dir=clean_img_dir,
        frame_files=all_frames,
        queue_size=16,
    )
    prefetcher.start(start_idx=s_frame - 1)

    renderer = PhysicalRenderer()
    attacked = 0
    skipped  = 0

    print(f"\n[*] Starting EOT+PGD blackbox attack...\n")

    try:
        for frame_idx in range(s_frame, e_frame + 1):

            tensor = prefetcher.get()
            if tensor is None:
                break

            frame_t = tensor.unsqueeze(0).to(device)

            row = tgt_gt[tgt_gt["frame"] == frame_idx]
            if row.empty:
                skipped += 1
                continue

            x1 = max(0, int(row["x"].values[0]))
            y1 = max(0, int(row["y"].values[0]))
            w  = max(1, int(row["w"].values[0]))
            h  = max(1, int(row["h"].values[0]))

            # Clamp to frame bounds
            _, _, fh, fw = frame_t.shape
            x1 = min(x1, fw - 2); x2 = min(x1 + w, fw)
            y1 = min(y1, fh - 2); y2 = min(y1 + h, fh)
            w  = x2 - x1;         h  = y2 - y1
            if w < 2 or h < 2:
                skipped += 1
                continue

            print(f"  Frame {frame_idx:04d}  bbox=[{x1},{y1},{w},{h}]")

            patch = optimize_patch_blackbox(
                model    = model,
                frame    = frame_t,
                box      = [x1, y1, w, h],
                renderer = renderer,
                epsilon  = EPSILON,
                alpha    = ALPHA,
                iters    = ITERS,
                n_eot    = N_EOT,
            )

            poisoned   = frame_t.clone()
            phys_final = renderer.apply(patch, bbox_w=w, bbox_h=h)
            poisoned[:, :, y1:y1+h, x1:x1+w] = torch.clamp(
                poisoned[:, :, y1:y1+h, x1:x1+w] + phys_final, 0.0, 1.0
            )

            save_path = os.path.join(out_img_dir, f"{frame_idx:06d}.jpg")
            torchvision.utils.save_image(poisoned[0], save_path)
            attacked += 1

    finally:
        prefetcher.stop()

    print(f"\n[*] Done.  Attacked={attacked}  Skipped={skipped}")
    print(f"[*] Blackbox sequence → {out_base}")
    print("[*] Next step: run generate_poisoned_detections.py "
          "(point it at MOT17-04-Blackbox)")