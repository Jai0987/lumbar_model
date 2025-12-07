import os
import random
import cv2
import matplotlib.pyplot as plt

IMG_DIR = "data/images"
MASK_DIR = "data/masks_manual"

def load(img_path, mask_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return img, mask

def main():
    images = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.jpg') or f.endswith('.png')])
    print(f"Found {len(images)} images")
    # sample 5 rand pairs
    sample_imgs = random.sample(images, 5)

    for img_name in sample_imgs:
        img_path = os.path.join(IMG_DIR, img_name)
        mask_name = img_name.replace('.jpg', '.png').replace('.jpeg','.png')
        mask_path = os.path.join(MASK_DIR, mask_name)

        img, mask = load(img_path, mask_path)

        print(f"Displaying pair: {img_name} + {mask_name}")

        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.title("Image")
        plt.imshow(img, cmap='gray')
        plt.axis('off')

        plt.subplot(1,3,2)
        plt.title("Mask")
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

        plt.subplot(1,3,3)
        plt.title("Overlay")
        overlay = cv2.addWeighted(img, 0.7, mask, 0.3, 0)
        plt.imshow(overlay, cmap='gray')
        plt.axis('off')

        plt.show()

if __name__ == '__main__':
    main()
