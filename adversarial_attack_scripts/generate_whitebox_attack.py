# adversarial_attack_scripts/generate_whitebox_attack.py
"""
White-Box Adaptive Physical EOT + BPDA + PGD Attack Generator.

Attacker assumptions (whitebox):
  - Knows the full defense architecture (all 4 actions T0..T3)
  - Uses BPDA to pass gradients through non-differentiable transforms
  - Averages gradients over N_EOT physical renderings AND all 4 actions
  - Models physical patch deployment via PhysicalRenderer

Fixes applied vs original:
  FIX-1  EOT now averages over N_EOT samples per action per PGD step
  FIX-2  Gradient accumulation rewritten — no None-grad crash after detach()
  FIX-3  rl_action cycles ALL 4 defense actions (true whitebox)
  FIX-4  Missing target no longer silently breaks loop — skips frame
  FIX-5  align_corners=False on all F.interpolate calls
  FIX-6  Non-attacked clean frames copied to poisoned folder before loop
  FIX-7  BPDA warp action now uses differentiable affine grid (was a no-op)
  FIX-8  PhysicalRenderer replaces naive eot_transform
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import shutil
import yaml
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import pandas as pd

from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from src.mot_env import FramePrefetcher
from adversarial_attack_scripts.target_selector import find_optimal_target

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_EOT    = 10     # EOT samples averaged per action per PGD step
EPSILON  = 1.0    # Full contrast — physical patch can use any colour
ALPHA    = 0.05   # PGD step size
ITERS    = 40     # PGD iterations per frame
# Whitebox: attacker knows ALL 4 defense actions and averages over them
WHITEBOX_ACTIONS = [0, 1, 2, 3]


# ── 1. Physical Renderer ──────────────────────────────────────────────────────

class PhysicalRenderer:
    """
    Simulates real-world patch deployment before injection into a frame.

    Models four physical degradation sources:
      1. Print gamut compression  (inkjet/laser cannot reproduce full sRGB)
      2. Lighting variation       (50 lux indoor → 50 klux outdoor)
      3. Perspective distortion   (patch is on tilted clothing/surface)
      4. Distance scaling         (target at 3m–25m from camera)

    Every patch MUST pass through this before pixel injection.
    This closes the "digital-only injection" gap that causes desk rejections.
    """

    def __init__(self):
        self.rng = np.random.default_rng()

    def apply(
        self,
        patch: torch.Tensor,    # (1, C, H, W) float32 [0,1]
        bbox_w: int,
        bbox_h: int,
    ) -> torch.Tensor:
        """Returns a physically-rendered patch at size (1, C, bbox_h, bbox_w)."""
        if patch.dim() == 3:
            patch = patch.unsqueeze(0)

        device = patch.device
        p = patch.cpu().float()

        # 1. Print gamut compression
        gamut = float(self.rng.uniform(0.72, 0.92))
        shift = float(self.rng.uniform(-0.04, 0.04))
        p = torch.clamp(p * gamut + shift, 0.0, 1.0)

        # 2. Lighting variation
        p = TF.adjust_brightness(p, float(self.rng.uniform(0.55, 1.45)))
        p = TF.adjust_contrast(p,   float(self.rng.uniform(0.75, 1.35)))
        p = TF.adjust_saturation(p, float(self.rng.uniform(0.70, 1.30)))
        p = TF.adjust_hue(p,        float(self.rng.uniform(-0.08, 0.08)))
        p = torch.clamp(p, 0.0, 1.0)

        # 3. Perspective distortion (clothing tilt / camera angle)
        tilt_x = float(self.rng.uniform(-0.12, 0.12))
        tilt_y = float(self.rng.uniform(-0.08, 0.08))
        shear  = float(self.rng.uniform(-0.06, 0.06))
        theta  = torch.tensor([[
            [1.0 + tilt_x, shear,         tilt_x * 0.5],
            [shear * 0.3,  1.0 + tilt_y,  tilt_y * 0.5],
        ]], dtype=torch.float32)
        grid = F.affine_grid(theta, p.size(), align_corners=False)
        p    = F.grid_sample(p, grid, align_corners=False,
                             mode="bilinear", padding_mode="border")

        # 4. Distance scaling (3m–25m CCTV range)
        d     = float(self.rng.uniform(3.0, 25.0))
        scale = max(0.20, min(1.0, 5.0 / d))
        if scale < 0.95:
            sh = max(4, int(p.shape[2] * scale))
            sw = max(4, int(p.shape[3] * scale))
            p  = F.interpolate(p, size=(sh, sw), mode="bilinear",
                               align_corners=False)

        # 5. Resize to exact bbox and add mild blur (lens + print dot gain)
        p = F.interpolate(p, size=(bbox_h, bbox_w), mode="bilinear",
                          align_corners=False)
        sigma = float(self.rng.uniform(0.3, 1.2))
        p = TF.gaussian_blur(p, kernel_size=[3, 3], sigma=[sigma, sigma])

        return p.to(device)


# ── 2. BPDA Wrapper ───────────────────────────────────────────────────────────

class BPDA(torch.autograd.Function):
    """
    Backward Pass Differentiable Approximation.

    Forward : applies the REAL non-differentiable defense transform
    Backward: identity — passes gradient straight through as if defense = f(x) = x

    This lets PGD optimise through transforms that have no true gradient.
    FIX-7: T1 warp is now a real differentiable affine (was a no-op pass).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, action: int) -> torch.Tensor:
        ctx.save_for_backward(x)
        out = x.clone()

        if action == 1:
            # T1: Spatial warp — differentiable affine approximation
            # Matches the perspective jitter in transformations.py
            theta = torch.tensor([[
                [1.0,  0.02, 0.01],
                [0.02, 1.0,  0.01],
            ]], device=x.device, dtype=torch.float32)
            grid = F.affine_grid(theta, x.size(), align_corners=False)
            out  = F.grid_sample(x, grid, align_corners=False,
                                 mode="bilinear", padding_mode="reflection")

        elif action == 2:
            # T2: Gaussian noise — σ matches transformations.py (15/255)
            out = torch.clamp(x + torch.randn_like(x) * (15.0 / 255.0), 0, 1)

        elif action == 3:
            # T3: Block cutout — random sector masked to 0.5
            _, _, h, w = x.shape
            bh = h // 5
            bw = w // 5
            y0 = torch.randint(0, max(1, h - bh), (1,)).item()
            x0 = torch.randint(0, max(1, w - bw), (1,)).item()
            out[:, :, y0:y0+bh, x0:x0+bw] = 0.5

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        # Identity: pretend defense = f(x) = x
        return grad_out, None


# ── 3. Target score helper ────────────────────────────────────────────────────

def _get_target_score(
    preds: dict,
    x1: int, y1: int, x2: int, y2: int,
) -> torch.Tensor | None:
    """
    Returns the detection confidence for the box whose centre lies
    inside [x1, y1, x2, y2]. Returns None if target is already suppressed.
    """
    for box, score in zip(preds["boxes"], preds["scores"]):
        bx1, by1, bx2, by2 = box
        cx = (bx1 + bx2) / 2
        cy = (by1 + by2) / 2
        if x1 < cx < x2 and y1 < cy < y2:
            return score
    return None


# ── 4. PGD + EOT + BPDA optimisation ─────────────────────────────────────────

def optimize_patch(
    model:    torch.nn.Module,
    frame:    torch.Tensor,        # (1, C, H, W) float32 on device
    box:      list,                # [x1, y1, w, h] ints
    actions:  list[int],           # WHITEBOX: [0,1,2,3]
    renderer: PhysicalRenderer,
    epsilon:  float = EPSILON,
    alpha:    float = ALPHA,
    iters:    int   = ITERS,
    n_eot:    int   = N_EOT,
) -> torch.Tensor:
    """
    Optimises a physical adversarial patch using PGD with:
      - True EOT: gradient averaged over n_eot physical renderings
      - True whitebox: gradient averaged over all defense actions
      - BPDA: identity backward through non-differentiable transforms

    Returns optimised patch at original patch spatial size.
    """
    device = frame.device
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h

    # Initialise patch as uniform noise in [-ε, ε]
    # Shape: (1, 3, h, w) — same spatial size as the target bounding box
    patch_data = torch.empty(1, 3, h, w, device=device).uniform_(-epsilon, epsilon)

    for iteration in range(iters):
        # We accumulate gradients manually across all (action × eot_sample) combos
        patch_data.requires_grad_(True)
        accum_grad = torch.zeros_like(patch_data)
        total_loss = 0.0
        n_valid    = 0

        for action in actions:
            for eot_idx in range(n_eot):

                # ── Physical rendering (FIX-8) ────────────────────────
                # renderer.apply returns a new tensor with grad_fn intact
                phys_patch = renderer.apply(patch_data, bbox_w=w, bbox_h=h)

                # ── Inject into frame ─────────────────────────────────
                poisoned = frame.clone()
                region   = poisoned[:, :, y1:y2, x1:x2]
                poisoned[:, :, y1:y2, x1:x2] = torch.clamp(
                    region + phys_patch, 0.0, 1.0
                )

                # ── BPDA forward (real defense, identity backward) ────
                defended = BPDA.apply(poisoned, action)

                # ── Detector forward pass ─────────────────────────────
                preds    = model([defended[0]])[0]
                t_score  = _get_target_score(preds, x1, y1, x2, y2)

                # Target already suppressed for this sample — skip
                if t_score is None:
                    continue

                # ── Backward ─────────────────────────────────────────
                # FIX-2: we accumulate into accum_grad and zero manually
                # so that patch_data.grad is never None after detach
                t_score.backward(retain_graph=False)

                if patch_data.grad is not None:
                    accum_grad = accum_grad + patch_data.grad.detach().clone()
                    patch_data.grad.zero_()
                    total_loss += t_score.item()
                    n_valid    += 1

        # All samples suppressed — patch already works, stop early
        if n_valid == 0:
            patch_data = patch_data.detach()
            print(f"    iter {iteration+1:>3d}  target suppressed — early stop")
            break

        # ── PGD step (FIX-1: averaged over all samples) ───────────────
        avg_grad   = accum_grad / n_valid
        patch_data = patch_data.detach() - alpha * avg_grad.sign()
        patch_data = torch.clamp(patch_data, -epsilon, epsilon)

        if (iteration + 1) % 10 == 0:
            print(f"    iter {iteration+1:>3d}/{iters}  "
                  f"avg_loss={total_loss/n_valid:.4f}  "
                  f"valid={n_valid}/{len(actions)*n_eot}")

    return patch_data.detach()


# ── 5. Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    cfg      = yaml.safe_load(open("config.yaml"))
    seq_path = cfg["data"]["seq_path"]
    parent   = os.path.dirname(seq_path)

    # Output directories for the poisoned (whitebox) sequence
    out_base    = os.path.join(parent, "MOT17-04-Poisoned")
    out_img_dir = os.path.join(out_base, "img1")
    out_gt_dir  = os.path.join(out_base, "gt")
    out_det_dir = os.path.join(out_base, "det")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_gt_dir,  exist_ok=True)
    os.makedirs(out_det_dir, exist_ok=True)

    # ── FIX-6: Copy GT and clean det immediately ──────────────────────
    # Non-attacked frames must exist so the sequence is valid MOT17 format
    print("[*] Copying GT and det files...")
    shutil.copy(os.path.join(seq_path, "gt",  "gt.txt"),
                os.path.join(out_gt_dir,  "gt.txt"))
    shutil.copy(os.path.join(seq_path, "det", "det.txt"),
                os.path.join(out_det_dir, "det.txt"))

    # ── Find optimal target ───────────────────────────────────────────
    target    = find_optimal_target(seq_path, min_frames=100, min_visibility=0.8)
    tid       = target["target_id"]
    s_frame   = target["start_frame"]
    e_frame   = target["end_frame"]

    print(f"\n[*] Whitebox attack — Target ID={tid}  "
          f"frames {s_frame}→{e_frame}  "
          f"actions={WHITEBOX_ACTIONS}  N_EOT={N_EOT}")

    # ── Load GT for dynamic bbox tracking ────────────────────────────
    cols   = ["frame","id","x","y","w","h","active","class","visibility"]
    df_gt  = pd.read_csv(os.path.join(seq_path, "gt", "gt.txt"),
                         header=None, names=cols)
    tgt_gt = df_gt[df_gt["id"] == tid]

    # ── Load Faster R-CNN on GPU ──────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Loading Faster R-CNN on {device}...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    ).to(device).eval()

    # ── FIX-6: Copy all clean frames as base before overwriting ───────
    clean_img_dir = os.path.join(seq_path, "img1")
    all_frames    = sorted(os.listdir(clean_img_dir))
    print(f"[*] Copying {len(all_frames)} clean frames as base...")
    for fname in all_frames:
        dst = os.path.join(out_img_dir, fname)
        if not os.path.exists(dst):
            shutil.copy(os.path.join(clean_img_dir, fname), dst)

    # ── Prefetcher starts at attack window ────────────────────────────
    prefetcher = FramePrefetcher(
        img_dir=clean_img_dir,
        frame_files=all_frames,
        queue_size=16,
    )
    prefetcher.start(start_idx=s_frame - 1)

    renderer = PhysicalRenderer()
    attacked = 0
    skipped  = 0

    print(f"\n[*] Starting EOT+BPDA+PGD whitebox attack...\n")

    try:
        for frame_idx in range(s_frame, e_frame + 1):

            tensor = prefetcher.get()
            if tensor is None:
                break

            # Prefetcher returns (C,H,W) — add batch dim
            # Convert the NumPy array frame safely to a torch tensor
            if isinstance(tensor, np.ndarray):
                # original shape: (H, W, C) -> unsqueeze(0) makes it (1, H, W, C)
                # permute(0, 3, 1, 2) reorders it to standard PyTorch batch: (1, C, H, W)
                frame_t = torch.from_numpy(tensor).unsqueeze(0).permute(0, 3, 1, 2).to(device).float()
            else:
                if tensor.ndim == 4 and tensor.shape[-1] == 3:
                    frame_t = tensor.permute(0, 3, 1, 2).to(device).float()
                else:
                    frame_t = tensor.to(device).float()
            row = tgt_gt[tgt_gt["frame"] == frame_idx]
            if row.empty:
                # FIX-4: skip cleanly instead of breaking
                skipped += 1
                continue

            x1 = max(0, int(row["x"].values[0]))
            y1 = max(0, int(row["y"].values[0]))
            w  = max(1, int(row["w"].values[0]))
            h  = max(1, int(row["h"].values[0]))

            # Clamp bbox to frame dimensions
            _, _, fh, fw = frame_t.shape
            x1 = min(x1, fw - 2); x2 = min(x1 + w, fw)
            y1 = min(y1, fh - 2); y2 = min(y1 + h, fh)
            w  = x2 - x1;         h  = y2 - y1
            if w < 2 or h < 2:
                skipped += 1
                continue

            print(f"  Frame {frame_idx:04d}  bbox=[{x1},{y1},{w},{h}]")

            # ── Optimise patch (whitebox: all 4 actions) ──────────────
            patch = optimize_patch(
                model    = model,
                frame    = frame_t,
                box      = [x1, y1, w, h],
                actions  = WHITEBOX_ACTIONS,
                renderer = renderer,
                epsilon  = EPSILON,
                alpha    = ALPHA,
                iters    = ITERS,
                n_eot    = N_EOT,
            )

            # ── Final physical injection ──────────────────────────────
            # Apply one final physical rendering at exact bbox size
            poisoned   = frame_t.clone()
            phys_final = renderer.apply(patch, bbox_w=w, bbox_h=h)
            poisoned[:, :, y1:y1+h, x1:x1+w] = torch.clamp(
                poisoned[:, :, y1:y1+h, x1:x1+w] + phys_final, 0.0, 1.0
            )

            # ── Overwrite the clean copy with poisoned version ─────────
            save_path = os.path.join(out_img_dir, f"{frame_idx:06d}.jpg")
            torchvision.utils.save_image(poisoned[0], save_path)
            attacked += 1

    finally:
        prefetcher.stop()

    print(f"\n[*] Done.  Attacked={attacked}  Skipped={skipped}")
    print(f"[*] Poisoned sequence → {out_base}")
    print("[*] Next step: run generate_poisoned_detections.py")