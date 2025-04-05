import numpy as np
import cv2
from skimage.feature import hog
import tensorflow as tf
from tensorflow.keras import layers, Sequential, Input
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from sklearn.preprocessing import StandardScaler

def extract_hog_features(image, pixels_per_cell=(4, 4), cells_per_block=(2, 2)):
    """Extracts HOG features (expected length ~34596) from the image."""
    try:
        features, _ = hog(image,
                          orientations=9,
                          pixels_per_cell=pixels_per_cell,
                          cells_per_block=cells_per_block,
                          block_norm='L2-Hys',
                          visualize=True)
        return features
    except Exception as e:
        print("Error extracting HOG features:", e)
        return np.array([])

def build_cnn_feature_extractor(input_shape=(128, 128, 1)):
    """
    Builds a custom CNN based on the paper:
      - 3 convolutional layers (32, 64, 128 filters) with ReLU and max pooling.
      - A flatten layer and a 512-unit dense layer.
    """
    try:
        model = Sequential([
            Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation='relu')
        ])
        return model
    except Exception as e:
        print("Error building custom CNN model:", e)
        return None

def build_vgg16_feature_extractor(input_shape=(128, 128, 3)):
    """
    Builds a VGG16-based feature extractor (top removed) followed by flattening and a 512-unit dense layer.
    """
    try:
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
        base_model.trainable = False
        model = Sequential([
            base_model,
            layers.Flatten(),
            layers.Dense(512, activation='relu')
        ])
        return model
    except Exception as e:
        print("Error building VGG16 model:", e)
        return None

def build_resnet_feature_extractor(input_shape=(128, 128, 3)):
    """
    Builds a ResNet50-based feature extractor.
    Uses a pretrained ResNet50 with the top removed,
    followed by global average pooling and a 512-unit dense layer.
    """
    try:
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
        base_model.trainable = False
        model = Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu')
        ])
        return model
    except Exception as e:
        print("Error building ResNet50 model:", e)
        return None

def extract_cnn_features(model, image):
    """
    Extracts CNN features from the image using the provided model.
    Converts grayscale to 3-channel if needed for pretrained models.
    """
    try:
        image = image.astype("float32") / 255.0
        if len(image.shape) == 2:
            if model.name.lower().find("vgg16") != -1 or model.name.lower().find("resnet") != -1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        if model.name.lower().find("vgg16") != -1:
            image = vgg16_preprocess(image)
        elif model.name.lower().find("resnet") != -1:
            image = resnet_preprocess(image)
        features = model.predict(image)
        return features.flatten()
    except Exception as e:
        print("Error extracting CNN features:", e)
        return np.array([])

def extract_hybrid_features(image, cnn_model):
    """
    Extracts HOG and CNN features, normalizes them separately,
    and concatenates them to form the hybrid feature vector.
    """
    hog_feats = extract_hog_features(image)
    cnn_feats = extract_cnn_features(cnn_model, image)
    if hog_feats.size == 0 or cnn_feats.size == 0:
        print("Warning: One of the feature extractions returned an empty array.")
    scaler_hog = StandardScaler()
    scaler_cnn = StandardScaler()
    hog_norm = scaler_hog.fit_transform(hog_feats.reshape(-1, 1)).flatten()
    cnn_norm = scaler_cnn.fit_transform(cnn_feats.reshape(-1, 1)).flatten()
    combined_features = np.concatenate((hog_norm, cnn_norm))
    return combined_features

if __name__ == "__main__":
    import os
    test_path = os.path.join("..", "data", "UTSig", "Genuine", "1", "1.tif")
    img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (128, 128))
        # To use custom CNN:
        # cnn_model = build_cnn_feature_extractor()
        # To use VGG16:
        # cnn_model = build_vgg16_feature_extractor()
        # To use ResNet50:
        cnn_model = build_resnet_feature_extractor()
        if cnn_model is not None:
            features = extract_hybrid_features(img, cnn_model)
            print("Hybrid feature vector length:", len(features))
        else:
            print("CNN model creation failed.")
    else:
        print("Test image not found.")
