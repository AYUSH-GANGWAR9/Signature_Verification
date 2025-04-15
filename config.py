"""
Configuration parameters for signature verification system.
"""

# Paths
UTSIG_PATH = "./data/UTSig/"
CEDAR_PATH = "./data/CEDAR/"
MODELS_PATH = "./models/"

# HOG parameters
HOG_CELL_SIZE = 4  # Cell size [4x4] as per paper
HOG_ORIENTATIONS = 9  # Default HOG orientations

# CNN parameters
CNN_INPUT_SHAPE = (150, 150, 1)  # Input shape for signature images
CNN_BATCH_SIZE = 32
CNN_EPOCHS = 20

# Feature selection
FEATURE_SELECTION_THRESHOLD = 0.001  # Threshold for feature importance

# Training parameters
TRAIN_TEST_SPLIT = 0.2  # 20% for testing
RANDOM_SEED = 42

# Classification parameters
K_NEIGHBORS = 5  # K value for KNN classifier