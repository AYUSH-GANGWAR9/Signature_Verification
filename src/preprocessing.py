"""
Preprocessing functions for signature images.
"""

import cv2
import numpy as np
from skimage.transform import resize

def load_image(image_path):
    """Load an image in grayscale."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return img

def preprocess_image(img, target_size=(150, 150)):
    """Preprocess image for feature extraction."""
    if img is None:
        return None
    
    # Resize the image
    img_resized = resize(img, target_size, anti_aliasing=True, preserve_range=True)
    
    # Normalize to [0, 1]
    img_normalized = img_resized / 255.0
    
    # Enhance contrast using histogram equalization
    img_enhanced = cv2.equalizeHist(img_resized.astype(np.uint8)) / 255.0
    
    return img_enhanced

def load_and_preprocess_image(image_path, target_size=(150, 150)):
    """Load and preprocess an image."""
    img = load_image(image_path)
    return preprocess_image(img, target_size)

def augment_data(img):
    """Apply simple data augmentation techniques."""
    augmented_images = []
    
    # Original image
    augmented_images.append(img)
    
    # Rotation
    rows, cols = img.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 10, 1)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (cols, rows))
    augmented_images.append(rotated_img)
    
    # Slight shear/affine transform
    pts1 = np.float32([[0, 0], [cols-1, 0], [0, rows-1]])
    pts2 = np.float32([[0, 0], [cols-1, 0], [int(0.1*cols), rows-1]])
    shear_matrix = cv2.getAffineTransform(pts1, pts2)
    sheared_img = cv2.warpAffine(img, shear_matrix, (cols, rows))
    augmented_images.append(sheared_img)
    
    return augmented_images