#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from PIL import Image

MASK_DIR = Path("data/masks_manual")

ys_min = []
ys_max = []
areas = []
heights = []
widths = []

for p in MASK_DIR.glob("*.png"):
    mask = Image.open(p).convert("L")
    m = (np.array(mask) > 127).astype(np.uint8)
    H, W = m.shape
    ys, xs = np.where(m == 1)
    if ys.size == 0:
        continue
    ys_min.append(ys.min() / H)   # relative positions
    ys_max.append(ys.max() / H)
    areas.append(ys.size / (H * W))
    heights.append(H)
    widths.append(W)

print(f"Num masks used: {len(ys_min)}")
print(f"rel_y_min: mean={np.mean(ys_min):.3f}, min={np.min(ys_min):.3f}, max={np.max(ys_min):.3f}")
print(f"rel_y_max: mean={np.mean(ys_max):.3f}, min={np.min(ys_max):.3f}, max={np.max(ys_max):.3f}")
print(f"area_frac: mean={np.mean(areas):.4f}, min={np.min(areas):.4f}, max={np.max(areas):.4f}")
