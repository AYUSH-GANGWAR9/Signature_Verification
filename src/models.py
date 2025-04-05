from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout

def train_svm(X_train, y_train, grid_search=False):
    """
    Trains an SVM with an RBF (Gaussian) kernel and balanced class weights.
    Optionally performs grid search for hyperparameter tuning.
    """
    from sklearn.model_selection import GridSearchCV
    parameters = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf'],
        'gamma': ['scale', 'auto']
    }
    base_estimator = SVC(probability=True, class_weight="balanced", random_state=42)
    if grid_search:
        clf = GridSearchCV(base_estimator, parameters, cv=3, scoring='roc_auc')
        clf.fit(X_train, y_train)
        print("Best SVM params:", clf.best_params_)
        return clf.best_estimator_
    else:
        base_estimator.fit(X_train, y_train)
        return base_estimator

def train_knn(X_train, y_train, grid_search=False):
    """
    Trains a KNN classifier.
    Optionally performs grid search for hyperparameter tuning.
    """
    from sklearn.model_selection import GridSearchCV
    parameters = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    }
    base_estimator = KNeighborsClassifier()
    if grid_search:
        clf = GridSearchCV(base_estimator, parameters, cv=3, scoring='roc_auc')
        clf.fit(X_train, y_train)
        print("Best KNN params:", clf.best_params_)
        return clf.best_estimator_
    else:
        base_estimator.fit(X_train, y_train)
        return base_estimator

def build_lstm_model(input_shape):
    """
    Builds and compiles an LSTM model to process the (reshaped) feature vector.
    """
    try:
        model = Sequential([
            Input(shape=input_shape),
            LSTM(64, return_sequences=True),
            Dropout(0.5),
            LSTM(32),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        print("Error building LSTM model:", e)
        return None

if __name__ == "__main__":
    import numpy as np
    X_dummy = np.random.rand(100, 200)
    y_dummy = np.random.randint(0, 2, 100)
    svm_model = train_svm(X_dummy, y_dummy)
    knn_model = train_knn(X_dummy, y_dummy)
    if svm_model is not None and knn_model is not None:
        print("SVM and KNN models trained successfully.")
