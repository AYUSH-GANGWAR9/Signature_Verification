"""
Classifier implementations: LSTM, SVM, and KNN.
"""

import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape, TimeDistributed
from tensorflow.keras.layers import Input

def create_lstm_model(input_shape, lstm_units=64, dropout_rate=0.2):
    feature_dim = input_shape[0]

    # Find a sequence length that divides the feature dimension
    for seq_length in range(10, 1, -1):
        if feature_dim % seq_length == 0:
            break
    else:
        raise ValueError(f"Cannot reshape input of size {feature_dim} into a valid (seq_len, features) shape.")

    n_features = feature_dim // seq_length

    model = Sequential([
        Input(shape=(feature_dim,)),
        Reshape((seq_length, n_features)),
        LSTM(lstm_units, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(32),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_svm_classifier(kernel='rbf', C=1.0, gamma='scale'):
    """
    Create an SVM classifier for signature verification.
    
    Parameters:
    -----------
    kernel : str
        The kernel type to be used in the algorithm.
    C : float
        Regularization parameter.
    gamma : str or float
        Kernel coefficient.
        
    Returns:
    --------
    classifier : SVC
        The SVM classifier.
    """
    return SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        probability=True,
        random_state=42
    )

def create_knn_classifier(n_neighbors=5, weights='uniform', algorithm='auto'):
    """
    Create a KNN classifier for signature verification.
    
    Parameters:
    -----------
    n_neighbors : int
        Number of neighbors.
    weights : str
        Weight function used in prediction.
    algorithm : str
        Algorithm used to compute the nearest neighbors.
        
    Returns:
    --------
    classifier : KNeighborsClassifier
        The KNN classifier.
    """
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm
    )