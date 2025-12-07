import os
import shutil
from PIL import Image
import numpy as np

RAW_DIR = "data/raw_unlabelled"
OUT_DIR = "data/to_label"
os.makedirs(OUT_DIR, exist_ok=True)

scores = []

for fname in sorted(os.listdir(RAW_DIR)):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(RAW_DIR, fname)

    try:
        img = Image.open(path).convert("L")  # grayscale
    except Exception as e:
        print(f"Skipping {fname}: {e}")
        continue

    img_np = np.array(img, dtype=np.float32)
    h, w = img_np.shape

    # depth band where bone usually is (roughly 30–60% of height)
    y1 = int(0.30 * h)
    y2 = int(0.60 * h)
    roi = img_np[y1:y2, :]

    if roi.size == 0:
        score = 0.0
    else:
        flat = roi.flatten()
        k = max(1, int(0.05 * flat.size))  # top 5% brightest pixels
        topk = np.partition(flat, -k)[-k:]
        score = float(topk.mean())

    scores.append((score, fname))

# sort by brightness score (descending)
scores.sort(reverse=True)

# how many frames to label
N = 120  # can change this number
# could also be a really useful number to get a good sample of the dataset

selected = scores[:N]

print(f"Scored {len(scores)} frames. Top {N} examples:")
for s, f in selected[:10]:
    print(f"  {f}: score={s:.2f}")

for _, fname in selected:
    src = os.path.join(RAW_DIR, fname)
    dst = os.path.join(OUT_DIR, fname)
    shutil.copy2(src, dst)

print(f"Selected {len(selected)} frames into {OUT_DIR}")
