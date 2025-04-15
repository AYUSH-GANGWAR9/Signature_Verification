"""
Feature integration module to combine HOG and CNN features.
"""

import numpy as np

def integrate_features(hog_features, cnn_features):
    """
    Integrate HOG and CNN features into a single feature vector.
    
    Parameters:
    -----------
    hog_features : array-like, shape = [n_samples, n_hog_features]
        The HOG features.
    cnn_features : array-like, shape = [n_samples, n_cnn_features]
        The CNN features.
        
    Returns:
    --------
    integrated_features : array-like, shape = [n_samples, n_hog_features + n_cnn_features]
        The integrated features.
    """
    # Check if both feature sets are available
    if hog_features.size == 0:
        return cnn_features
    elif cnn_features.size == 0:
        return hog_features
    
    # Ensure the number of samples is the same
    assert hog_features.shape[0] == cnn_features.shape[0], "Number of samples must be the same"
    
    # Concatenate features along the feature axis
    integrated_features = np.hstack((hog_features, cnn_features))
    
    return integrated_features