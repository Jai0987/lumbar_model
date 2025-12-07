import os
import random

IMG_DIR = "data/images"
MASK_DIR = "data/masks_manual"
SPLIT_DIR = "data/splits"

def main():
    os.makedirs(SPLIT_DIR, exist_ok=True)

    # Get all labeled images (those that have a matching mask)
    images = sorted([
        f for f in os.listdir(IMG_DIR)
        if os.path.exists(os.path.join(MASK_DIR, f.replace('.jpg', '.png')))
    ])

    print(f"Found {len(images)} labelled samples.")

    random.shuffle(images)

    n = len(images)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    n_test  = n - n_train - n_val  # remainder

    train_files = images[:n_train]
    val_files = images[n_train:n_train + n_val]
    test_files = images[n_train + n_val:]

    # Save lists
    with open(os.path.join(SPLIT_DIR, "train.txt"), "w") as f:
        f.write("\n".join(train_files))

    with open(os.path.join(SPLIT_DIR, "val.txt"), "w") as f:
        f.write("\n".join(val_files))

    with open(os.path.join(SPLIT_DIR, "test.txt"), "w") as f:
        f.write("\n".join(test_files))

    print("Splits written to data/splits/")
    print(f"Train: {len(train_files)}  Val: {len(val_files)}  Test: {len(test_files)}")

if __name__ == "__main__":
    main()
