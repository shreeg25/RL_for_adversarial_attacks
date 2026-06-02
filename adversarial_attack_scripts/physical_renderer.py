# adversarial_attack_scripts/physical_renderer.py
"""
Simulates physical-world patch deployment constraints.

Models four real-world degradation sources:
  1. Perspective distortion   — patch is on a tilted/angled surface
  2. Lighting variation       — indoor/outdoor brightness & colour shift
  3. Print colour loss        — inkjet/laser gamut compression (sRGB → print gamut)
  4. Distance scaling         — target moves toward/away from camera

Every patch injected into a frame MUST pass through PhysicalRenderer.apply()
before being added to the pixel buffer. This satisfies the physical threat
model requirement for IEEE Transactions on IFS / CVPR reviewers.

Usage:
    renderer = PhysicalRenderer()
    patch_physical = renderer.apply(patch_tensor, bbox_area_px, frame_distance_hint)
"""
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np


class PhysicalRenderer:
    """
    Applies physically-motivated transformations to an adversarial patch
    before it is injected into a surveillance frame.

    All parameters are sampled stochastically to simulate the variance
    of a real deployment (different lighting, angles, distances).
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def apply(
        self,
        patch: torch.Tensor,          # (1, C, H, W) or (C, H, W) float32 [0,1]
        bbox_w: int,                  # target bbox width  in pixels
        bbox_h: int,                  # target bbox height in pixels
        distance_m: float | None = None,  # estimated distance in metres (optional)
    ) -> torch.Tensor:
        """
        Returns a physically-rendered patch of spatial size (bbox_h, bbox_w).
        """
        if patch.dim() == 3:
            patch = patch.unsqueeze(0)   # → (1, C, H, W)

        device = patch.device
        p = patch.cpu().float()

        # ── 1. Print Colour Gamut Compression ────────────────────────
        # Inkjet/laser printers cannot reproduce saturated RGB values
        # faithfully. Empirically, the effective gamut is ~75–90% of sRGB.
        gamut_scale = float(self.rng.uniform(0.72, 0.92))
        gamut_shift = float(self.rng.uniform(-0.04, 0.04))
        p = torch.clamp(p * gamut_scale + gamut_shift, 0.0, 1.0)

        # ── 2. Lighting Variation ─────────────────────────────────────
        # Simulates scene illuminance (50 lux indoor → 50 klux outdoor)
        # mapped to brightness [0.55, 1.45] and colour temperature shift
        brightness = float(self.rng.uniform(0.55, 1.45))
        contrast   = float(self.rng.uniform(0.75, 1.35))
        saturation = float(self.rng.uniform(0.70, 1.30))
        hue        = float(self.rng.uniform(-0.08, 0.08))
        p = TF.adjust_brightness(p, brightness)
        p = TF.adjust_contrast(p,   contrast)
        p = TF.adjust_saturation(p, saturation)
        p = TF.adjust_hue(p,        hue)
        p = torch.clamp(p, 0.0, 1.0)

        # ── 3. Perspective Distortion ────────────────────────────────
        # A patch worn on clothing is never perfectly fronto-parallel.
        # We apply a random thin-plate-spline-approximated perspective warp.
        _, C, H, W = p.shape
        tilt_x = float(self.rng.uniform(-0.12, 0.12))   # horizontal tilt
        tilt_y = float(self.rng.uniform(-0.08, 0.08))   # vertical tilt (less)
        shear  = float(self.rng.uniform(-0.06, 0.06))

        # Build perspective transform matrix
        theta = torch.tensor([[
            [1.0 + tilt_x, shear,         tilt_x * 0.5],
            [shear * 0.3,  1.0 + tilt_y,  tilt_y * 0.5],
        ]], dtype=torch.float32)

        grid = F.affine_grid(theta, p.size(), align_corners=False)
        p    = F.grid_sample(p, grid, align_corners=False,
                             mode="bilinear", padding_mode="border")

        # ── 4. Distance Scaling ──────────────────────────────────────
        # As target moves away from camera, effective patch resolution
        # drops. We simulate this by downsampling then upsampling.
        if distance_m is not None:
            # Rough model: a 0.3m patch at 5m fills ~60px, at 20m ~15px
            scale = max(0.20, min(1.0, 5.0 / max(distance_m, 1.0)))
        else:
            # Sample from realistic CCTV distance range (3m – 25m)
            d     = float(self.rng.uniform(3.0, 25.0))
            scale = max(0.20, min(1.0, 5.0 / d))

        if scale < 0.95:
            # Downsample to simulate distance, then upsample to bbox size
            small_h = max(4, int(H * scale))
            small_w = max(4, int(W * scale))
            p = F.interpolate(p, size=(small_h, small_w),
                              mode="bilinear", align_corners=False)

        # ── 5. Resize to exact bbox dimensions ───────────────────────
        p = F.interpolate(p, size=(bbox_h, bbox_w),
                          mode="bilinear", align_corners=False)

        # ── 6. Mild Gaussian blur (lens + print dot gain) ────────────
        kernel_size = 3
        sigma = float(self.rng.uniform(0.3, 1.2))
        p = TF.gaussian_blur(p, kernel_size=[kernel_size, kernel_size],
                             sigma=[sigma, sigma])

        return p.to(device)

    def apply_batch(
        self,
        patch: torch.Tensor,
        bbox_w: int,
        bbox_h: int,
        n: int = 10,
    ) -> torch.Tensor:
        """
        Returns a batch of n independently-rendered versions of the patch.
        Used for EOT averaging in the attack optimiser.
        Shape: (n, C, bbox_h, bbox_w)
        """
        samples = [self.apply(patch, bbox_w, bbox_h) for _ in range(n)]
        return torch.cat(samples, dim=0)   # (n, C, H, W)