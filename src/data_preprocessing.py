import os
import cv2
import numpy as np
from skimage.filters import threshold_otsu

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')

def get_label_from_path(image_path):
    """Return 1 for genuine/original images, 0 for forgery/forged images."""
    path_lower = image_path.lower()
    if "genuine" in path_lower or "original" in path_lower:
        return 1
    elif "forgery" in path_lower or "forged" in path_lower:
        return 0
    return None

def load_images_from_folder(dataset_folder, img_size=(128, 128)):
    images = []
    labels = []
    paths = []

    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Dataset folder '{dataset_folder}' not found!")

    print(f"Scanning dataset folder: {dataset_folder}")
    for root, _, files in os.walk(dataset_folder):
        for file in files:
            if file.lower().endswith(VALID_EXTENSIONS):
                full_path = os.path.join(root, file)
                label = get_label_from_path(full_path)
                if label is None:
                    continue
                try:
                    img = preprocess_image(full_path, size=img_size)
                    images.append(img)
                    labels.append(label)
                    paths.append(full_path)
                except Exception as e:
                    print(f"Error processing {full_path}: {e}")

    print(f"Total images loaded: {len(images)}")
    if len(images) == 0:
        raise ValueError("No images loaded. Check dataset path and structure.")
    print(f"Label distribution: { {lbl: labels.count(lbl) for lbl in set(labels)} }")
    return np.array(images), np.array(labels), paths

def preprocess_image(image_path, size=(128, 128)):
    """Load image in grayscale, resize, apply Gaussian blur, and OTSU threshold."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")
    img = cv2.resize(img, size)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    thresh_val = threshold_otsu(img)
    img = (img > thresh_val).astype(np.uint8) * 255
    return img

if __name__ == "__main__":
    root = os.path.join("..", "data")
    imgs, labs, paths = load_images_from_folder(root)
    print(f"Loaded {len(imgs)} images with labels: {set(labs)}")
