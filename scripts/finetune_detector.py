# scripts/finetune_detector.py
import os
import torch
import pandas as pd
import cv2
import numpy as np
import yaml
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from src.device import DEVICE

class MOT17Dataset(Dataset):
    def __init__(self, seq_dirs):
        self.samples = []
        for seq in seq_dirs:
            img_dir = os.path.join(seq, "img1")
            if not os.path.exists(img_dir): continue
            
            # Load Ground Truth
            gt_file = os.path.join(seq, "gt", "gt.txt")
            df = pd.read_csv(gt_file, header=None, names=["frame", "id", "x", "y", "w", "h", "active", "class", "vis"])
            # Filter for active, visible pedestrians
            df = df[(df["active"] == 1) & (df["class"] == 1) & (df["vis"] >= 0.25)]
            
            for frame_no, grp in df.groupby("frame"):
                img_path = os.path.join(img_dir, f"{int(frame_no):06d}.jpg")
                boxes = grp[["x", "y", "w", "h"]].values
                # Convert xywh to xyxy for torchvision
                boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
                boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
                self.samples.append((img_path, boxes.astype(np.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, boxes = self.samples[idx]
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        # Convert to PyTorch format [C, H, W] and normalize to [0, 1]
        tensor_img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        
        # Format targets
        target = {}
        target["boxes"] = torch.from_numpy(boxes)
        target["labels"] = torch.ones((len(boxes),), dtype=torch.int64) # Label 1 is Pedestrian
        
        return tensor_img, target

def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml"))
    
    # We only train on Clean data. Never feed the adversarial Whitebox data to the baseline detector.
    all_seqs = [cfg["data"]["seq_path"]] + cfg["data"].get("train_sequences", [])
    clean_seqs = list(set([s for s in all_seqs if os.path.exists(s) and "Whitebox" not in s and "Blackbox" not in s]))
    
    print(f"[FINETUNE] Assembling dataset from {len(clean_seqs)} clean sequences...")
    dataset = MOT17Dataset(clean_seqs)
    
    # Use batch size 2 to protect your 8GB VRAM
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=0)
    
    print("[FINETUNE] Downloading base COCO weights and injecting 2-Class Head...")
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    
    model.to(DEVICE)
    
    # Standard Optimizer configuration for fine-tuning
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    EPOCHS = 2  # 2 epochs is enough to adapt the feature pyramid to MOT17
    
    print("[FINETUNE] Initiating Domain Adaptation...")
    model.train()
    
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for i, (images, targets) in enumerate(loader):
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            if i % 50 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {i}/{len(loader)} | Loss: {losses.item():.4f}")
                
        print(f"=== Epoch {epoch+1} Complete | Average Loss: {epoch_loss/len(loader):.4f} ===")
        
    os.makedirs("weights", exist_ok=True)
    save_path = "weights/faster_rcnn_mot17.pth"
    torch.save(model.state_dict(), save_path)
    print(f"[FINETUNE] Success. MOT17 domain weights locked and saved to {save_path}")