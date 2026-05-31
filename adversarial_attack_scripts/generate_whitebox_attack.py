# adversarial_attack_scripts/generate_whitebox_attack.py
"""
White-Box Adaptive EOT + BPDA + PGD Attack Generator
Synthesizes adversarial patches designed specifically to bypass the TRACE architecture.
"""
import sys
import os

# Crucial: This ensures Python can find your 'src' folder when running from the root directory
sys.path.insert(0, os.path.abspath("."))

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import cv2

# ─── 1. BPDA Wrapper (The Mathematical Bypass) ──────────────────────────────

class TRACE_BPDA_Wrapper(torch.autograd.Function):
    """
    Backward Pass Differentiable Approximation.
    Allows PGD gradients to flow through the RL agent's discrete, 
    non-differentiable image transformations.
    """
    @staticmethod
    def forward(ctx, image_tensor, action_idx):
        # Save the original image for the backward pass (Identity approximation)
        ctx.save_for_backward(image_tensor)
        
        # Apply the actual, non-differentiable TRACE defense
        # 0: Clean, 1: Warp, 2: Noise, 3: Cutout
        defended_image = image_tensor.clone()
        
        if action_idx == 1:
            # Simulate Spatial Warp (Perspective)
            # In actual implementation, hook to your cv2 warp logic here
            pass 
        elif action_idx == 2:
            # Simulate Gaussian Noise
            noise = torch.randn_like(defended_image) * 0.1
            defended_image = torch.clamp(defended_image + noise, 0, 1)
        elif action_idx == 3:
            # Simulate Random Block Cutout
            h, w = defended_image.shape[2:]
            y = torch.randint(0, h - 20, (1,))
            x = torch.randint(0, w - 20, (1,))
            defended_image[:, :, y:y+20, x:x+20] = 0.0

        return defended_image

    @staticmethod
    def backward(ctx, grad_output):
        # The core BPDA logic: Identity mapping.
        # We pretend the defense was f(x) = x, passing the gradient straight through.
        image_tensor, = ctx.saved_tensors
        return grad_output, None  # None for action_idx since it's an integer


# ─── 2. Expectation Over Transformation (EOT) ───────────────────────────────

def apply_eot(patch_tensor):
    """
    Applies random physical transformations to the patch to simulate 
    real-world camera shifts, rotations, and lighting changes.
    """
    angle = (torch.rand(1).item() - 0.5) * 30  # -15 to +15 degrees
    scale = 0.9 + torch.rand(1).item() * 0.2   # 0.9x to 1.1x scale
    
    transform = T.Compose([
        T.RandomAffine(degrees=angle, scale=(scale, scale)),
        T.ColorJitter(brightness=0.2, contrast=0.2)
    ])
    
    return transform(patch_tensor)


# ─── 3. PGD Optimization Loop ───────────────────────────────────────────────

def optimize_adversarial_patch(model_frcnn, base_image, target_box, rl_action, epsilon=8/255, alpha=2/255, iters=20):
    """
    Runs the PGD attack through the BPDA wrapper to suppress the Faster R-CNN logit.
    """
    device = base_image.device
    
    # Initialize a random patch over the target bounding box
    x1, y1, w, h = target_box
    x2, y2 = x1 + w, y1 + h
    
    patch = torch.zeros_like(base_image[:, :, y1:y2, x1:x2]).uniform_(-epsilon, epsilon)
    patch.requires_grad = True

    model_frcnn.eval()

    print(f"[Attack] Optimizing EOT+BPDA patch over {iters} iterations...")
    
    for i in range(iters):
        # 1. Apply EOT to the patch
        eot_patch = apply_eot(patch)
        
        # 2. Inject patch into the image
        poisoned_image = base_image.clone()
        poisoned_image[:, :, y1:y2, x1:x2] += eot_patch
        poisoned_image = torch.clamp(poisoned_image, 0, 1)

        # 3. Apply the RL Defense via BPDA
        # If the RL agent picks T3 (Cutout), the forward pass cuts the image, 
        # but the backward pass will still pull gradients through it.
        defended_image = TRACE_BPDA_Wrapper.apply(poisoned_image, rl_action)

        # 4. Forward pass through Faster R-CNN
        # (Assuming your Faster R-CNN wrapper outputs logits/scores)
        predictions = model_frcnn([defended_image[0]])
        
        # Find the prediction overlapping our target box and extract its confidence score
        # For this script, we assume a simplified loss: minimize the max objectness score
        scores = predictions[0]['scores']
        
        if len(scores) == 0:
            break # Attack successfully suppressed all boxes
            
        loss = scores.max() # We want to minimize this confidence

        # 5. BPDA Backward Pass
        model_frcnn.zero_grad()
        loss.backward()

        # 6. PGD Update Step (Gradient Ascent on the Attack Loss)
        with torch.no_grad():
            # Standard Fast Gradient Sign Method (FGSM) step
            patch -= alpha * patch.grad.sign()
            # Project back to epsilon ball
            patch = torch.clamp(patch, -epsilon, epsilon)
            patch.grad.zero_()

        if i % 5 == 0:
            print(f"  Step {i:02d} | Target Confidence: {loss.item():.4f}")

    print("[Attack] Patch optimization complete.")
    return patch.detach()

if __name__ == "__main__":
    import yaml
    from adversarial_attack_scripts.target_selector import find_optimal_target

    cfg = yaml.safe_load(open("config.yaml"))
    seq_path = cfg["data"]["seq_path"]

    # 1. Automatically find the best victim
    target_info = find_optimal_target(seq_path, min_frames=100, min_visibility=0.8)
    
    target_id = target_info["target_id"]
    start_frame = target_info["start_frame"]
    end_frame = target_info["end_frame"]

    # 2. Load the specific frame sequence for that target
    # 3. Pass the target's bounding box into the optimize_adversarial_patch() function
    # 4. Render the poisoned frames to a new directory