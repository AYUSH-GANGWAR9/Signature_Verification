"""
Feature extraction using HOG and CNN.
"""

import numpy as np
import tensorflow as tf
from skimage.feature import hog
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16

def extract_hog_features(img, cell_size=4, orientations=9):
    """Extract HOG features from an image."""
    # Calculate HOG features
    features, _ = hog(
        img, 
        orientations=orientations,
        pixels_per_cell=(cell_size, cell_size),
        cells_per_block=(2, 2),
        visualize=True,
        feature_vector=True
    )
    return features

def build_cnn_model(input_shape=(150, 150, 1)):
    """Build a CNN model for feature extraction."""
    inputs = Input(shape=input_shape)
    
    # First convolutional block
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Second convolutional block
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Third convolutional block
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Flatten the output and add fully connected layers
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Output layer for binary classification (genuine vs forged)
    output = Dense(1, activation='sigmoid')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=output)
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def create_feature_extractor(model):
    """Create a feature extractor from a trained CNN model."""
    # Get the output of the last Dense layer before the classification layer
    feature_extractor = Model(
        inputs=model.input,
        outputs=model.layers[-3].output  # Feature layer before the final Dense layer
    )
    return feature_extractor

def extract_cnn_features(img, feature_extractor):
    """Extract features using a pre-trained CNN model."""
    # Reshape image to match CNN input shape
    img_reshaped = np.expand_dims(np.expand_dims(img, axis=-1), axis=0)
    
    # Extract features
    features = feature_extractor.predict(img_reshaped)
    return features.flatten()

def extract_features_from_image(img, cnn_feature_extractor=None):
    """Extract both HOG and CNN features from an image."""
    # Extract HOG features
    hog_features = extract_hog_features(img)
    
    # Extract CNN features if a feature extractor is provided
    cnn_features = np.array([])
    if cnn_feature_extractor is not None:
        cnn_features = extract_cnn_features(img, cnn_feature_extractor)
    
    return hog_features, cnn_features