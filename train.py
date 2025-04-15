"""
Training script for signature verification system.
"""

import os
import numpy as np
import tensorflow as tf
import pickle
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import time

# Import local modules
from config import *
from src.preprocessing import load_and_preprocess_image, augment_data
from src.feature_extraction import extract_hog_features, build_cnn_model, create_feature_extractor
from src.feature_selection import select_and_visualize_features
from src.feature_integration import integrate_features
from src.classifiers import create_lstm_model, create_svm_classifier, create_knn_classifier
from src.evaluation import calculate_metrics, print_classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split


def load_dataset(dataset_path, dataset_name, max_samples=None):
    """Load and preprocess the dataset."""
    print(f"Loading {dataset_name} dataset...")

    genuine_samples = []
    forged_samples = []
    labels = []

    if dataset_name == "UTSig":
        # Adjusted path structure
        genuine_dirs = sorted(glob(os.path.join(dataset_path, "Genuine", "*", "*")))
        skilled_dirs = sorted(glob(os.path.join(dataset_path, "Forgery", "Skilled", "*", "*")))

        if max_samples:
            genuine_dirs = genuine_dirs[:max_samples]
            skilled_dirs = skilled_dirs[:max_samples]

        for path in tqdm(genuine_dirs, desc="Loading genuine signatures"):
            img = load_and_preprocess_image(path)
            if img is not None:
                genuine_samples.append(img)
                labels.append(1)

        for path in tqdm(skilled_dirs, desc="Loading skilled forgeries"):
            img = load_and_preprocess_image(path)
            if img is not None:
                forged_samples.append(img)
                labels.append(0)

    elif dataset_name == "CEDAR":
        genuine_dirs = sorted(glob(os.path.join(dataset_path, "full_org/*")))
        skilled_dirs = sorted(glob(os.path.join(dataset_path, "full_forg/*")))

        if max_samples:
            genuine_dirs = genuine_dirs[:max_samples]
            skilled_dirs = skilled_dirs[:max_samples]

        for path in tqdm(genuine_dirs, desc="Loading genuine signatures"):
            img = load_and_preprocess_image(path)
            if img is not None:
                genuine_samples.append(img)
                labels.append(1)

        for path in tqdm(skilled_dirs, desc="Loading forged signatures"):
            img = load_and_preprocess_image(path)
            if img is not None:
                forged_samples.append(img)
                labels.append(0)

    print(f"Loaded {len(genuine_samples)} genuine and {len(forged_samples)} forged signatures.")

    samples = genuine_samples + forged_samples
    labels = np.array(labels)

    return samples, labels


def extract_all_features(samples, cnn_model=None):
    """Extract HOG and CNN features from all samples."""
    print("Extracting features...")
    
    hog_features = []
    cnn_features = []
    
    # Create feature extractor from CNN if model is provided
    feature_extractor = None
    if cnn_model is not None:
        feature_extractor = create_feature_extractor(cnn_model)
    
    # Extract features for each sample
    for img in tqdm(samples, desc="Extracting features"):
        # Extract HOG features
        hog_feat = extract_hog_features(img)
        hog_features.append(hog_feat)
        
        # Extract CNN features if feature extractor is available
        if feature_extractor is not None:
            # Reshape image for CNN input
            img_cnn = np.expand_dims(np.expand_dims(img, axis=-1), axis=0)
            cnn_feat = feature_extractor.predict(img_cnn).flatten()
            cnn_features.append(cnn_feat)
    
    hog_features = np.array(hog_features)
    cnn_features = np.array(cnn_features) if cnn_features else np.array([])
    
    return hog_features, cnn_features

def train_cnn_model(samples, labels):
    """Train a CNN model for feature extraction."""
    print("Training CNN model for feature extraction...")
    
    # Prepare data for CNN
    X = np.array([np.expand_dims(sample, axis=-1) for sample in samples])
    y = labels
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    # Build and train the CNN model
    model = build_cnn_model(input_shape=CNN_INPUT_SHAPE)
    
    history = model.fit(
        X_train, y_train,
        batch_size=CNN_BATCH_SIZE,
        epochs=CNN_EPOCHS,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig("cnn_training_history.png")
    plt.close()
    
    # Save the trained model
    model.save(os.path.join(MODELS_PATH, "cnn_feature_extractor.h5"))
    
    return model

def main():
    """Main function to train the signature verification system."""
    # Create necessary directories
    os.makedirs(MODELS_PATH, exist_ok=True)
    
    # Select dataset (UTSig or CEDAR)
    dataset_name = "UTSig"  # Change to "CEDAR" if using CEDAR dataset
    dataset_path = UTSIG_PATH if dataset_name == "UTSig" else CEDAR_PATH
    
    # Load dataset
    samples, labels = load_dataset(dataset_path, dataset_name)
    
    # Train CNN model for feature extraction
    cnn_model = train_cnn_model(samples, labels)
    
    # Extract HOG and CNN features
    hog_features, cnn_features = extract_all_features(samples, cnn_model)
    
    # Select important features
    print("Selecting important features...")
    hog_features_selected, hog_indices = select_and_visualize_features(
        hog_features, labels, "HOG", FEATURE_SELECTION_THRESHOLD
    )
    
    cnn_features_selected, cnn_indices = select_and_visualize_features(
        cnn_features, labels, "CNN", FEATURE_SELECTION_THRESHOLD
    )
    
    # Integrate features
    integrated_features = integrate_features(hog_features_selected, cnn_features_selected)
    
    # --- Pad to make feature vector divisible for LSTM ---
    feature_dim = integrated_features.shape[1]

    for seq_length in range(10, 1, -1):
        if feature_dim % seq_length == 0:
            break
    else:
        # If not divisible, pad with zeros to make it divisible
        for seq_length in range(10, 1, -1):
            remainder = feature_dim % seq_length
            if remainder > 0:
                pad_size = seq_length - remainder
                integrated_features = np.pad(integrated_features, ((0, 0), (0, pad_size)), mode='constant')
                feature_dim = integrated_features.shape[1]
                break

    #     # --- Trim to make feature vector divisible for LSTM ---
    # feature_dim = integrated_features.shape[1]

    # # Find nearest smaller divisible shape
    # for seq_length in range(10, 1, -1):
    #     if feature_dim % seq_length == 0:
    #         break
    # else:
    #     for seq_length in range(10, 1, -1):
    #         if (feature_dim - feature_dim % seq_length) % seq_length == 0:
    #             feature_dim = feature_dim - (feature_dim % seq_length)
    #             break

    # # Trim the feature vectors to match
    # integrated_features = integrated_features[:, :feature_dim]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        integrated_features, labels, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_SEED
    )
    
    # Train and evaluate classifiers
    classifiers = {
        "LSTM": create_lstm_model(input_shape=(integrated_features.shape[1],)),
        "SVM": create_svm_classifier(),
        "KNN": create_knn_classifier(n_neighbors=K_NEIGHBORS)
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        print(f"\nTraining {name} classifier...")
        start_time = time.time()
        
        if name == "LSTM":
            # Train LSTM model
            history = clf.fit(
                X_train, y_train,
                batch_size=32,
                epochs=20,
                validation_split=0.2,
                verbose=1
            )
            
            # Predict probabilities
            y_pred_proba = clf.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Save the model
            clf.save(os.path.join(MODELS_PATH, f"{name.lower()}_model.h5"))
            
        else:
            # Train SVM or KNN
            clf.fit(X_train, y_train)
            
            # Predict
            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
            
            # Save the model
            with open(os.path.join(MODELS_PATH, f"{name.lower()}_model.pkl"), 'wb') as file:
                pickle.dump(clf, file)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        metrics['training_time'] = training_time
        
        # Plot ROC curve if probability scores are available
        if y_pred_proba is not None:
            metrics['roc_auc'] = plot_roc_curve(y_test, y_pred_proba, f"{name} ROC Curve")
        
        # Print classification report
        print(f"\n{name} Classification Report:")
        print_classification_report(y_test, y_pred, target_names=["Forged", "Genuine"])
        
        # Store results
        results[name] = metrics
        
        print(f"{name} Results:")
        for key, value in metrics.items():
            print(f"{key}: {value}")
    
    # Save feature selection indices for later use
    with open(os.path.join(MODELS_PATH, "feature_indices.pkl"), 'wb') as file:
        pickle.dump({
            'hog_indices': hog_indices,
            'cnn_indices': cnn_indices
        }, file)
    
    # Print final results
    print("\nFinal Results:")
    for name, metrics in results.items():
        print(f"\n{name}:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
    print(f"Number of samples to extract features from: {len(samples)}")
    if len(samples) == 0:
        raise ValueError("No samples loaded. Check dataset path and structure.")
