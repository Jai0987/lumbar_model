#!/usr/bin/env python3
import os
from pathlib import Path
from scipy import ndimage as ndi

import numpy as np
import torch
from PIL import Image

import segmentation_models_pytorch as smp

IMG_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = Path("data")
RAW_DIR = DATA_ROOT / "raw_unlabelled"
OUT_MASK_DIR = DATA_ROOT / "masks_pred"
OUT_OVERLAY_DIR = DATA_ROOT / "overlays_pred"

MODEL_PATH = Path("models/unet_best.pth")

OUT_MASK_DIR.mkdir(parents=True, exist_ok=True)
OUT_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

def postprocess_mask(mask_np,
                     y_low=0.10,
                     y_high=0.90,
                     min_area_frac=0.003):
    """
    mask_np: binary (0/1) numpy array at original res.
    - keep only vertical band [y_low, y_high] of the image height.
    - remove connected components smaller than min_area_frac of total pixels.
    """
    H, W = mask_np.shape

    # vertical band (had to relax a lot vs before)
    y0 = int(H * y_low)
    y1 = int(H * y_high)
    band = np.zeros_like(mask_np, dtype=np.uint8)
    band[y0:y1, :] = 1
    mask = mask_np * band

    # remove tiny CCs
    labeled, num = ndi.label(mask)
    if num == 0:
        return mask

    sizes = ndi.sum(mask, labeled, index=range(1, num + 1))
    min_area = int(min_area_frac * H * W)

    clean = np.zeros_like(mask, dtype=np.uint8)
    for label_id, size in enumerate(sizes, start=1):
        if size >= min_area:
            clean[labeled == label_id] = 1

    return clean

def keep_largest_component(mask_np, min_area_frac=0.001):
    """
    keep only the largest connected component of mask_np.
    if even that is smaller than min_area_frac of the image, return original mask_np.
    """
    H, W = mask_np.shape
    labeled, num = ndi.label(mask_np)
    if num == 0:
        return mask_np

    sizes = ndi.sum(mask_np, labeled, index=range(1, num + 1))
    max_idx = int(np.argmax(sizes)) + 1  # label ids start at 1
    max_size = sizes[max_idx - 1]
    min_area = int(min_area_frac * H * W)

    if max_size < min_area:
        # too small, just keep original (probably very uncertain frame)
        return mask_np

    clean = np.zeros_like(mask_np, dtype=np.uint8)
    clean[labeled == max_idx] = 1
    return clean


def load_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
        classes=1,
    )
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def predict_mask(model, img_pil):
    """img_pil: pil grayscale image"""
    orig_w, orig_h = img_pil.size

    # resize to training res
    img_resized = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0

    inp = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(inp)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    pred_small = (prob > 0.5).astype(np.uint8)  # 0/1 at 256x256 res

    mask_pil = Image.fromarray(pred_small * 255).resize((orig_w, orig_h), Image.NEAREST)
    mask_np0 = (np.array(mask_pil) > 127).astype(np.uint8)   # original pred at full res

    # gentle geometric + size filt
    mask_np = postprocess_mask(mask_np0,
                               y_low=0.10,
                               y_high=0.90,
                               min_area_frac=0.003)

    # fllback: if everything got wiped out, keep the largest blob
    if mask_np.sum() == 0 and mask_np0.sum() > 0:
        mask_np = keep_largest_component(mask_np0, min_area_frac=0.001)

    return mask_np


# needs fixing
def make_overlay(img_pil, mask_np, alpha=0.4):
    """Return rgb overlay image (pil)"""
    img_np = np.array(img_pil, dtype=np.float32) / 255.0  # hxw
    h, w = img_np.shape
    base = np.stack([img_np, img_np, img_np], axis=-1)    # hxwx3 gray
    red = np.zeros_like(base)
    red[..., 0] = 1.0
    mask = mask_np.astype(np.float32)[..., None]          # hxwx1
    overlay = base * (1 - alpha * mask) + red * (alpha * mask)
    overlay = (overlay * 255).clip(0,255).astype(np.uint8)
    return Image.fromarray(overlay)

def main():
    model = load_model()
    img_files = [f for f in RAW_DIR.iterdir() if f.suffix.lower() in [".jpg",".png",".jpeg",".bmp",".tif",".tiff"]]
    img_files = sorted(img_files)

    print(f"Found {len(img_files)} raw images in {RAW_DIR}")

    for i, img_path in enumerate(img_files, 1):
        img_pil = Image.open(img_path).convert("L")
        mask_np = predict_mask(model, img_pil)
        mask_out_path = OUT_MASK_DIR / f"{img_path.stem}.png"
        Image.fromarray(mask_np * 255).save(mask_out_path) # save mask
        overlay = make_overlay(img_pil, mask_np)
        overlay_out_path = OUT_OVERLAY_DIR / f"{img_path.stem}_overlay.png"
        overlay.save(overlay_out_path) # save overlay

        if i % 20 == 0 or i == len(img_files):
            print(f"Processed {i}/{len(img_files)} images")

    print("Done! Masks in:", OUT_MASK_DIR)
    print("Overlays in:", OUT_OVERLAY_DIR)

if __name__ == "__main__":
    main()
