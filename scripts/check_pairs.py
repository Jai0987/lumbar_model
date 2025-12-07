import os
from pathlib import Path
from PIL import Image
import numpy as np

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

print(f"Images found: {len(ims)}")
print(f"Masks found:  {len(masks)}")
print()

matched = set(ims_by_stem.keys()).intersection(masks_by_stem.keys())
only_images = sorted(set(ims_by_stem.keys()) - set(masks_by_stem.keys()))
only_masks  = sorted(set(masks_by_stem.keys()) - set(ims_by_stem.keys()))

print(f"Matched pairs: {len(matched)}")
print(f"Images with NO mask: {len(only_images)} (showing up to 20)")
for k in only_images[:20]:
    print("  -", ims_by_stem[k].name)
print()
print(f"Masks with NO image: {len(only_masks)} (showing up to 20)")
for k in only_masks[:20]:
    print("  -", masks_by_stem[k].name)

# check masks readability and not-empty
bad_masks = []
for stem in matched:
    mp = masks_by_stem[stem]
    try:
        img = Image.open(mp).convert("L")
        arr = np.asarray(img).astype("uint8")
        if arr.max() == 0:
            bad_masks.append((mp.name, "all-zero"))
    except Exception as e:
        bad_masks.append((mp.name, f"error:{e}"))

if bad_masks:
    print()
    print("Bad masks (empty or unreadable):")
    for b in bad_masks:
        print(" -", b)
else:
    print()
    print("All matched masks readable and non-empty (pass).")
