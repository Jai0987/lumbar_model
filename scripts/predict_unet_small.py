#!/usr/bin/env python3
from pathlib import Path
import csv

import numpy as np
from PIL import Image
import torch

from simple_unet import SimpleUNet

DATA_ROOT = Path("data")
RAW_DIR = DATA_ROOT / "raw_unlabelled"
OUT_MASK_DIR = DATA_ROOT / "masks_pseudo_small"
OUT_OVERLAY_DIR = DATA_ROOT / "overlays_pseudo_small"
MODEL_PATH = Path("models/unet_small_pseudo.pth")

OUT_MASK_DIR.mkdir(parents=True, exist_ok=True)
OUT_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    model = SimpleUNet(in_channels=1, out_channels=1, base_ch=32)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def predict_mask(model, img_pil):
    """Run model on a single PIL grayscale image, return 0/1 mask in original size."""
    orig_w, orig_h = img_pil.size

    # resize to training res
    img_resized = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0   # [h,w]

    inp = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float().to(DEVICE)  # [1,1,h,w]

    with torch.no_grad():
        logits = model(inp)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mask_small = (probs > 0.5).astype(np.uint8)  # [h,w] 0/1

    # back to original res
    mask_pil = Image.fromarray(mask_small * 255).resize((orig_w, orig_h), Image.NEAREST)
    mask_np = (np.array(mask_pil) > 127).astype(np.uint8)

    return mask_np


def make_overlay(img_pil, mask_np, alpha=0.4):
    """Overlay red mask on top of grayscale image; return rgb pil image."""
    img_np = np.array(img_pil, dtype=np.float32) / 255.0
    h, w = img_np.shape
    base = np.stack([img_np, img_np, img_np], axis=-1)  # hxwx3
    red = np.zeros_like(base)
    red[..., 0] = 1.0

    mask = mask_np.astype(np.float32)[..., None]
    overlay = base * (1 - alpha * mask) + red * (alpha * mask)
    overlay = (overlay * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def main():
    model = load_model()

    img_files = sorted(
        [p for p in RAW_DIR.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]]
    )
    print(f"Found {len(img_files)} raw images in {RAW_DIR}")

    stats_path = DATA_ROOT / "pseudo_stats_small.csv"
    with open(stats_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "area_frac", "cy_rel", "cx_rel", "plausible"])

        for i, img_path in enumerate(img_files, 1):
            img_pil = Image.open(img_path).convert("L")
            mask_np = predict_mask(model, img_pil)

            H, W = mask_np.shape
            area = mask_np.sum()
            area_frac = area / (H * W + 1e-8)

            if area > 0:
                ys, xs = np.where(mask_np == 1)
                cy_rel = float(ys.mean() / H)
                cx_rel = float(xs.mean() / W)
            else:
                cy_rel = -1.0
                cx_rel = -1.0

            # simple plausibility flag based on stats from 50 masks
            plausible = (0.015 <= area_frac <= 0.13)

            mask_out = OUT_MASK_DIR / f"{img_path.stem}.png"
            Image.fromarray(mask_np * 255).save(mask_out)
            overlay = make_overlay(img_pil, mask_np)
            overlay_out = OUT_OVERLAY_DIR / f"{img_path.stem}_overlay.png"
            overlay.save(overlay_out)

            writer.writerow([img_path.name, f"{area_frac:.5f}", f"{cy_rel:.3f}", f"{cx_rel:.3f}", int(plausible)])

            if i % 20 == 0 or i == len(img_files): # helps with debug logging incase processing got stuck
                print(f"Processed {i}/{len(img_files)} images")

    print("Done.")
    print("Pseudo-masks in:", OUT_MASK_DIR)
    print("Overlays in:", OUT_OVERLAY_DIR)
    print("Stats CSV:", stats_path)


if __name__ == "__main__":
    main()
