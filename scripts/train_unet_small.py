from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from dataset import UltrasoundBoneDataset
from simple_unet import SimpleUNet
from losses import BCEDiceLoss

# config
DATA_ROOT = Path("data")
IMG_DIR = DATA_ROOT / "images"
MASK_DIR = DATA_ROOT / "masks_manual"
SPLIT_DIR = DATA_ROOT / "splits"
BATCH_SIZE = 4
IMG_SIZE = 256
EPOCHS = 80
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "unet_small_pseudo.pth"


def dice_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()


def main():
    # Datasets
    train_list = SPLIT_DIR / "train.txt"
    val_list   = SPLIT_DIR / "val.txt"

    train_set = UltrasoundBoneDataset(
        img_dir=str(IMG_DIR),
        mask_dir=str(MASK_DIR),
        file_list_path=str(train_list),
        img_size=IMG_SIZE,
    )
    val_set = UltrasoundBoneDataset(
        img_dir=str(IMG_DIR),
        mask_dir=str(MASK_DIR),
        file_list_path=str(val_list),
        img_size=IMG_SIZE,
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {len(train_set)}  Val samples: {len(val_set)}")

    # model + loss + optimizer
    model = SimpleUNet(in_channels=1, out_channels=1, base_ch=32).to(DEVICE)
    criterion = BCEDiceLoss(alpha=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_val_dice = 0.0

    for epoch in range(1, EPOCHS + 1):
        # train
        model.train()
        train_losses, train_dices = [], []

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(imgs)
            loss = criterion(logits, masks)
            dice = dice_from_logits(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_dices.append(dice.item())

        avg_train_loss = float(np.mean(train_losses))
        avg_train_dice = float(np.mean(train_dices))

        # val
        model.eval()
        val_losses, val_dices = [], []

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                imgs = imgs.to(DEVICE)
                masks = masks.to(DEVICE)

                logits = model(imgs)
                loss = criterion(logits, masks)
                dice = dice_from_logits(logits, masks)

                val_losses.append(loss.item())
                val_dices.append(dice.item())

        avg_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        avg_val_dice = float(np.mean(val_dices)) if val_dices else 0.0

        print(
            f"Epoch {epoch:03d} | "
            f"TrainLoss: {avg_train_loss:.4f}  TrainDice: {avg_train_dice:.4f} | "
            f"ValLoss: {avg_val_loss:.4f}  ValDice: {avg_val_dice:.4f}"
        )

        # save best
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"New best model saved to {MODEL_PATH} (ValDice={best_val_dice:.4f})")

    print("Training complete. Best Val Dice:", best_val_dice)


if __name__ == "__main__":
    main()
