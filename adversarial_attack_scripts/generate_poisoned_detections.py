# adversarial_attack_scripts/generate_poisoned_detections.py
"""
Runs Faster R-CNN on the poisoned image folder to regenerate det.txt.
MUST be run AFTER generate_whitebox_attack.py (or generate_blackbox_attack.py).

FIX: Added pre-flight check so it fails loudly if the poisoned folder is empty.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))

import yaml
import torch
import torchvision
from torchvision.transforms import functional as TF
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from PIL import Image


def generate_poisoned_det(poisoned_seq_path: str):

    img_dir = os.path.join(poisoned_seq_path, "img1")
    det_dir = os.path.join(poisoned_seq_path, "det")
    os.makedirs(det_dir, exist_ok=True)

    # ── Pre-flight check ────────────────────────────────────────────
    img_names = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    if not img_names:
        raise RuntimeError(
            f"No images found in {img_dir}\n"
            "Run generate_whitebox_attack.py first to populate the poisoned folder."
        )
    print(f"[*] Found {len(img_names)} poisoned frames.")

    # ── Load model ───────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Loading Faster R-CNN on {device}...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    ).to(device).eval()

    det_file = os.path.join(det_dir, "det.txt")
    written  = 0

    print(f"[*] Generating detections → {det_file}")
    with open(det_file, "w") as f:
        for img_name in img_names:
            frame_id  = int(img_name.split(".")[0])
            img_path  = os.path.join(img_dir, img_name)

            img    = Image.open(img_path).convert("RGB")
            tensor = TF.to_tensor(img).unsqueeze(0).to(device)

            with torch.no_grad():
                preds = model(tensor)[0]

            for box, score, label in zip(
                preds["boxes"].cpu().numpy(),
                preds["scores"].cpu().numpy(),
                preds["labels"].cpu().numpy(),
            ):
                if label == 1 and score > 0.5:
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    f.write(
                        f"{frame_id},-1,{x1:.2f},{y1:.2f},"
                        f"{w:.2f},{h:.2f},{score:.4f},-1,-1,-1\n"
                    )
                    written += 1

            if frame_id % 100 == 0:
                print(f"  Frame {frame_id:04d}  detections so far: {written}")

    print(f"[*] Done. {written} detections written to {det_file}")


if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml"))
    parent        = os.path.dirname(cfg["data"]["seq_path"])
    poisoned_path = os.path.join(parent, "MOT17-04-FRCNN-Whitebox")
    generate_poisoned_det(poisoned_path)