"""
Feature selection using Decision Trees.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def select_important_features(X, y, threshold=0.001):
    """
    Select important features using Random Forest feature importance.
    
    Parameters:
    -----------
    X : array-like, shape = [n_samples, n_features]
        The training input samples.
    y : array-like, shape = [n_samples]
        The target values.
    threshold : float
        The threshold for feature importance.
        
    Returns:
    --------
    important_indices : array-like
        Indices of important features.
    feature_importance : array-like
        Feature importance values.
    """
    # Train a random forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    
    # Get feature importances
    feature_importance = clf.feature_importances_
    
    # Select features above threshold
    important_indices = np.where(feature_importance > threshold)[0]
    
    return important_indices, feature_importance

def visualize_feature_importance(feature_importance, title="Feature Importance"):
    """Visualize feature importance."""
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.title(title)
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.close()

def select_and_visualize_features(X, y, feature_type, threshold=0.001):
    """Select important features and visualize their importance."""
    important_indices, feature_importance = select_important_features(X, y, threshold)
    
    # Visualize feature importance
    visualize_feature_importance(feature_importance, f"{feature_type} Feature Importance")
    
    # Return the selected features
    X_selected = X[:, important_indices]
    
    print(f"Selected {len(important_indices)} important {feature_type} features out of {X.shape[1]}")
    
    return X_selected, important_indices