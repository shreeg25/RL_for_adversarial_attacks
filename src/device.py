# src/device.py
"""
Central device resolver. Import `DEVICE` from here everywhere.
Never hardcode 'cuda' or 'cpu' in other files.
"""
import torch
import yaml

def get_device(cfg: dict | None = None) -> torch.device:
    if cfg is None:
        cfg = yaml.safe_load(open("config.yaml"))
    if cfg["device"]["use_gpu"] and torch.cuda.is_available():
        dev = torch.device(f"cuda:{cfg['device']['gpu_id']}")
        name = torch.cuda.get_device_name(dev)
        vram = torch.cuda.get_device_properties(dev).total_memory / 1e9
        print(f"[device] Using GPU: {name}  ({vram:.1f} GB VRAM)")
        return dev
    print("[device] GPU not available or disabled — using CPU")
    return torch.device("cpu")

DEVICE = get_device()