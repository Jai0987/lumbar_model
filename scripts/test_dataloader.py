from torch.utils.data import DataLoader
from dataset import UltrasoundBoneDataset

DATA_ROOT = "data"

train_set = UltrasoundBoneDataset(
    img_dir=f"{DATA_ROOT}/images",
    mask_dir=f"{DATA_ROOT}/masks_manual",
    file_list_path=f"{DATA_ROOT}/splits/train.txt"
)

loader = DataLoader(train_set, batch_size=4, shuffle=True)

print("Dataset size:", len(train_set))

for imgs, masks in loader:
    print("Batch images shape:", imgs.shape)
    print("Batch masks shape: ", masks.shape)
    break
