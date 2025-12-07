from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

import segmentation_models_pytorch as smp
from dataset import UltrasoundBoneDataset

DATA_ROOT = Path("data")
IMG_DIR = DATA_ROOT / "images"
MASK_DIR = DATA_ROOT / "masks_manual"
SPLIT_DIR = DATA_ROOT / "splits"
MODEL_PATH = Path("models/unet_best.pth")

IMG_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = smp.Unet(
        encoder_name="resnet34", # i was thinking if we should use resnet34 / resnet50 or keep tuning u-net (or in fact, we could then take att-unet with a trimmed loss function?)
        encoder_weights=None,
        in_channels=1,
        classes=1,
    )
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def predict_single(model, img_path):
    img = Image.open(img_path).convert("L") # load image (grayscale)
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img_np = np.array(img, dtype=np.float32) / 255.0   # [h,w] 0..1

    inp = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1,1,h,w]
    with torch.no_grad():
        logits = model(inp)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()  # [h,w]

    pred_mask = (prob > 0.5).astype(np.uint8)  # binary 0/1 [h,w]
    return img_np, pred_mask

def main():
    # pick a few val + test images
    val_files = (SPLIT_DIR / "val.txt").read_text().strip().splitlines()
    test_files = (SPLIT_DIR / "test.txt").read_text().strip().splitlines()
    sample_files = val_files[:3] + test_files[:3]

    print("Visualizing predictions for:", sample_files)

    model = load_model()

    for fname in sample_files:
        img_path = IMG_DIR / fname
        mask_path = MASK_DIR / fname.replace(".jpg", ".png")

        # ground truth mask
        gt_mask = Image.open(mask_path).convert("L")
        gt_mask = gt_mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        gt_np = (np.array(gt_mask) > 127).astype(np.uint8)

        img_np, pred_mask = predict_single(model, img_path)

        plt.figure(figsize=(10,4))
        plt.suptitle(fname)

        plt.subplot(1,4,1)
        plt.title("Image")
        plt.imshow(img_np, cmap="gray")
        plt.axis("off")

        plt.subplot(1,4,2)
        plt.title("GT Mask")
        plt.imshow(gt_np, cmap="gray")
        plt.axis("off")

        plt.subplot(1,4,3)
        plt.title("Pred Mask")
        plt.imshow(pred_mask, cmap="gray")
        plt.axis("off")

        plt.subplot(1,4,4)
        plt.title("Overlay")
        
        # overlay prediction in red
        overlay = np.stack([img_np, img_np, img_np], axis=-1)  # gray → RGB
        red = np.zeros_like(overlay)
        red[..., 0] = 1.0  # pure red
        alpha = 0.4
        overlay = overlay * (1 - alpha * pred_mask[..., None]) + red * (alpha * pred_mask[..., None])
        plt.imshow(overlay)
        plt.axis("off")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
