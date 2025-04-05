import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA

def select_features_decision_tree(X, y, threshold="median"):
    """
    Uses a RandomForest (embedded decision tree) to select the most important features.
    """
    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        rf.fit(X, y)
        selector = SelectFromModel(rf, threshold=threshold, prefit=True)
        X_selected = selector.transform(X)
        return X_selected, selector
    except Exception as e:
        print("Error during decision-tree based feature selection:", e)
        return X, None

def select_features_pca(X, n_components=0.95):
    """
    Reduces dimensionality using PCA while preserving 95% of the variance.
    """
    try:
        pca = PCA(n_components=n_components, random_state=42)
        X_reduced = pca.fit_transform(X)
        return X_reduced, pca
    except Exception as e:
        print("Error during PCA feature selection:", e)
        return X, None

if __name__ == "__main__":
    X_dummy = np.random.rand(100, 35000)
    y_dummy = np.random.randint(0, 2, 100)
    X_sel, selector = select_features_decision_tree(X_dummy, y_dummy)
    print("Decision-tree reduced shape:", X_sel.shape)
    X_pca, pca_model = select_features_pca(X_dummy)
    print("PCA reduced shape:", X_pca.shape)
