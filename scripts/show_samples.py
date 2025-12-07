import random
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

ROOT = Path(".")
IM_DIR = ROOT / "data" / "images"
MASK_DIR = ROOT / "data" / "masks_manual"

def list_files(d):
    exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff")
    files = []
    for e in exts:
        files += list(d.glob(e))
    return files

ims = list_files(IM_DIR)
masks = list_files(MASK_DIR)
ims_by_stem = {p.stem: p for p in ims}
masks_by_stem = {p.stem: p for p in masks}
matched = list(set(ims_by_stem.keys()).intersection(masks_by_stem.keys()))
if not matched:
    print("No matched pairs found. Run check_pairs.py first.")
    raise SystemExit(1)

sample = random.sample(matched, min(6, len(matched)))
for s in sample:
    im = cv2.imread(str(ims_by_stem[s]), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(masks_by_stem[s]), cv2.IMREAD_GRAYSCALE)
    if im is None or mask is None:
        print("Failed to read", s)
        continue
    mask_bin = (mask > 127).astype('uint8')
    color = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    color[mask_bin==1] = [0,0,255]
    plt.figure(figsize=(4,6))
    plt.imshow(color[..., ::-1])
    plt.title(s)
    plt.axis('off')
plt.show()
