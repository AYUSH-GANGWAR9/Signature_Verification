import os
import argparse
import numpy as np
import joblib
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from tensorflow.keras.models import load_model
from sklearn.utils.class_weight import compute_class_weight
from src.data_preprocessing import load_images_from_folder
from src.feature_extraction import build_cnn_feature_extractor, build_vgg16_feature_extractor, build_resnet_feature_extractor, extract_hybrid_features
from src.feature_selection import select_features_decision_tree, select_features_pca
from src.models import train_svm, train_knn, build_lstm_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def prepare_dataset(dataset_folder, img_size=(128, 128)):
    images, labels, _ = load_images_from_folder(dataset_folder, img_size)
    return images, labels

def extract_features_from_dataset(images, cnn_model, use_hybrid=True):
    features = []
    for img in images:
        feat = extract_hybrid_features(img, cnn_model) if use_hybrid else img.flatten()
        features.append(feat)
    return np.array(features)

def stratified_subset(X, y, sample_size):
    sss = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=42)
    indices = next(sss.split(X, y))[0]
    return X[indices], y[indices]

def train_classifiers(X, y, epochs=5, sample_size=None, selection_method="decision_tree", use_grid_search=False):
    if sample_size is not None and sample_size < len(X):
        X, y = stratified_subset(X, y, sample_size)
        print(f"Training on a stratified subset of {sample_size} samples.")

    if len(set(y)) < 2:
        raise ValueError("Insufficient classes for training.")

    if selection_method.lower() == "decision_tree":
        X_selected, selector = select_features_decision_tree(X, y)
    elif selection_method.lower() == "pca":
        X_selected, selector = select_features_pca(X)
    else:
        print("Unknown selection method; proceeding without feature selection.")
        X_selected, selector = X, None

    svm_model = train_svm(X_selected, y, grid_search=use_grid_search)
    if svm_model is None:
        raise ValueError("SVM training failed.")
    knn_model = train_knn(X_selected, y, grid_search=use_grid_search)
    if knn_model is None:
        raise ValueError("KNN training failed.")

    X_lstm = X_selected.reshape((X_selected.shape[0], X_selected.shape[1], 1))
    lstm_model = build_lstm_model(input_shape=(X_lstm.shape[1], 1))
    if lstm_model is None:
        raise ValueError("LSTM model building failed.")

    classes = np.unique(y)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    class_weight_dict = {int(c): w for c, w in zip(classes, class_weights)}
    print("Class weights for LSTM:", class_weight_dict)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint_path = os.path.join("saved_models", "lstm_best_model.h5")
    model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)

    print(f"Training LSTM for up to {epochs} epochs...")
    lstm_model.fit(
        X_lstm, y,
        epochs=epochs,
        batch_size=16,
        verbose=1,
        validation_split=0.2,
        class_weight=class_weight_dict,
        callbacks=[early_stopping, model_checkpoint]
    )

    lstm_model = load_model(checkpoint_path)

    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(svm_model, os.path.join("saved_models", "svm_model.joblib"))
    joblib.dump(knn_model, os.path.join("saved_models", "knn_model.joblib"))
    joblib.dump(selector, os.path.join("saved_models", "selector.joblib"))
    lstm_model.save(os.path.join("saved_models", "lstm_model.h5"))
    print("Models saved in 'saved_models/' folder.")
    return svm_model, knn_model, lstm_model, selector

def load_trained_models():
    svm_path = os.path.join("saved_models", "svm_model.joblib")
    knn_path = os.path.join("saved_models", "knn_model.joblib")
    selector_path = os.path.join("saved_models", "selector.joblib")
    lstm_path = os.path.join("saved_models", "lstm_model.h5")
    if all(os.path.exists(p) for p in [svm_path, knn_path, selector_path, lstm_path]):
        svm_model = joblib.load(svm_path)
        knn_model = joblib.load(knn_path)
        selector = joblib.load(selector_path)
        lstm_model = load_model(lstm_path)
        print("Trained models loaded from 'saved_models/'.")
        return svm_model, knn_model, lstm_model, selector
    else:
        print("Saved models not found.")
        return None, None, None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classifiers for signature verification.")
    parser.add_argument("--dataset_folder", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--epochs", type=int, default=45, help="Number of epochs for LSTM training")
    parser.add_argument("--sample_size", type=int, default=1000, help="Optional stratified sample size")
    parser.add_argument("--selection_method", type=str, default="decision_tree", help="Feature selection: decision_tree or pca")
    parser.add_argument("--feature_extractor", type=str, default="hybrid", help="Feature extractor: hybrid (default), vgg16, or resnet")
    parser.add_argument("--use_grid_search", action="store_true", help="Use grid search for hyperparameter tuning")
    args = parser.parse_args()

    images, labels = prepare_dataset(args.dataset_folder)
    if args.feature_extractor.lower() == "vgg16":
        cnn_model = build_vgg16_feature_extractor()
    elif args.feature_extractor.lower() == "resnet":
        cnn_model = build_resnet_feature_extractor()
    else:
        cnn_model = build_cnn_feature_extractor()

    if cnn_model is None:
        print("CNN model build failed. Exiting.")
    else:
        X = extract_features_from_dataset(images, cnn_model, use_hybrid=True)
        try:
            svm_model, knn_model, lstm_model, selector = train_classifiers(
                X, labels, epochs=args.epochs, sample_size=args.sample_size,
                selection_method=args.selection_method, use_grid_search=args.use_grid_search)
            print("Models trained and saved successfully.")
        except Exception as e:
            print("Error during classifier training:", e)
