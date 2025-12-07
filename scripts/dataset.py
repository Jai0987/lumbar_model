import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class UltrasoundBoneDataset(Dataset):
    def __init__(self, img_dir, mask_dir, file_list_path, img_size=256, transform=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.img_size = img_size

        with open(file_list_path, "r") as f:
            self.files = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]

        # Paths
        img_path = self.img_dir / filename
        # masks are .png with same stem as image
        stem = os.path.splitext(filename)[0]
        mask_path = self.mask_dir / f"{stem}.png"

        img = Image.open(img_path).convert("L")   # grayscale
        mask = Image.open(mask_path).convert("L") # grayscale, but will binarize

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # Using NEAREST for mask to keep it clean binary
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        img = np.array(img, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.uint8)
        mask = (mask > 127).astype(np.float32)

        if self.transform is not None:
            img, mask = self.transform(img, mask)
        img = torch.from_numpy(img).float().unsqueeze(0)    # [1, H, W]
        mask = torch.from_numpy(mask).float().unsqueeze(0)  # [1, H, W]

        return img, mask
