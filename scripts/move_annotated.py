import os
import shutil
import re

IMAGES_DIR = "data/images"
MASKS_DIR = "data/masks_manual"
RAW_DIR = "data/raw_unlabelled"

os.makedirs(RAW_DIR, exist_ok=True)

# extract num ID before _mask
mask_ids = set()

mask_pattern = re.compile(r"(\d+)_mask")

for fname in os.listdir(MASKS_DIR):
    match = mask_pattern.match(fname)
    if match:
        mask_ids.add(match.group(1))

print("Detected mask IDs:", sorted(mask_ids))

count_raw = 0

for fname in os.listdir(IMAGES_DIR):
    img_id = os.path.splitext(fname)[0]

    # if the image filename is exactly a mask ID (num match)
    if img_id in mask_ids:
        print(f"Annotated image: {fname}")
    else:
        print(f"Moving RAW image: {fname}")
        shutil.move(os.path.join(IMAGES_DIR, fname),
                    os.path.join(RAW_DIR, fname))
        count_raw += 1

print(f"\nMoved {count_raw} raw images to data/raw_unlabelled/")
