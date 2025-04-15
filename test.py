"""
Interactive testing script for signature verification system.
"""

import os
import numpy as np
import tensorflow as tf
import pickle
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import cv2

# Import local modules
from config import *
from src.preprocessing import load_and_preprocess_image, load_image, preprocess_image
from src.feature_extraction import extract_hog_features, create_feature_extractor
from src.feature_integration import integrate_features
from src.evaluation import calculate_metrics, calculate_equal_error_rate, calculate_far_frr
from src.evaluation import print_classification_report, plot_roc_curve


def is_valid_image_file(path):
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    return path.lower().endswith(valid_extensions) and "Zone.Identifier" not in path


def load_test_samples(test_path, dataset_name, max_samples=None):
    """Load test samples."""
    print(f"Loading {dataset_name} test samples...")

    genuine_samples = []
    forged_samples = []
    labels = []

    if dataset_name == "UTSig":
        genuine_dirs = sorted([
            f for f in glob(os.path.join(test_path, "Genuine", "*", "*")) if is_valid_image_file(f)
        ])
        skilled_dirs = sorted([
            f for f in glob(os.path.join(test_path, "Forgery", "Skilled", "*", "*")) if is_valid_image_file(f)
        ])

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
        genuine_dirs = sorted(glob(os.path.join(test_path, "full_org/*")))
        skilled_dirs = sorted(glob(os.path.join(test_path, "full_forg/*")))

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


def extract_features(samples, cnn_model, hog_indices=None, cnn_indices=None):
    """Extract HOG and CNN features from samples."""
    print("Extracting features...")

    hog_features = []
    cnn_features = []
    feature_extractor = create_feature_extractor(cnn_model)

    for img in tqdm(samples, desc="Extracting features"):
        if img is None:
            continue

        try:
            hog_feat = extract_hog_features(img)
            if hog_feat is not None and len(hog_feat) > 0:
                hog_features.append(np.array(hog_feat).flatten())

            img_cnn = np.expand_dims(np.expand_dims(img, axis=-1), axis=0)
            cnn_feat = feature_extractor.predict(img_cnn, verbose=0).flatten()
            cnn_features.append(cnn_feat)

        except Exception as e:
            print(f"[Warning] Skipping a sample due to error: {e}")

    hog_features = np.array(hog_features)
    cnn_features = np.array(cnn_features)

    print("HOG feature shape:", hog_features.shape)
    print("CNN feature shape:", cnn_features.shape)

    if hog_features.size == 0 or cnn_features.size == 0:
        raise ValueError("No valid HOG or CNN features extracted. Check sample images or feature functions.")

    if hog_indices is not None and hog_features.ndim == 2:
        hog_features_selected = hog_features[:, hog_indices]
    else:
        hog_features_selected = hog_features

    if cnn_indices is not None and cnn_features.ndim == 2:
        cnn_features_selected = cnn_features[:, cnn_indices]
    else:
        cnn_features_selected = cnn_features

    return hog_features_selected, cnn_features_selected


def load_models_and_indices():
    """Load trained models and feature indices."""
    models = {}

    cnn_model = tf.keras.models.load_model(os.path.join(MODELS_PATH, "cnn_feature_extractor.h5"))
    models["CNN"] = cnn_model

    with open(os.path.join(MODELS_PATH, "feature_indices.pkl"), 'rb') as file:
        feature_indices = pickle.load(file)
    hog_indices = feature_indices['hog_indices']
    cnn_indices = feature_indices['cnn_indices']

    try:
        lstm_model = tf.keras.models.load_model(os.path.join(MODELS_PATH, "lstm_model.h5"))
        models["LSTM"] = lstm_model
    except:
        print("LSTM model not found.")

    try:
        with open(os.path.join(MODELS_PATH, "svm_model.pkl"), 'rb') as file:
            svm_model = pickle.load(file)
        models["SVM"] = svm_model
    except:
        print("SVM model not found.")

    try:
        with open(os.path.join(MODELS_PATH, "knn_model.pkl"), 'rb') as file:
            knn_model = pickle.load(file)
        models["KNN"] = knn_model
    except:
        print("KNN model not found.")

    return models, hog_indices, cnn_indices


def verify_signature(test_signature_path, reference_signature_path, models, hog_indices, cnn_indices):
    """Verify if a test signature matches with a reference signature."""
    test_img = load_and_preprocess_image(test_signature_path)
    reference_img = load_and_preprocess_image(reference_signature_path)

    if test_img is None or reference_img is None:
        return "Error: Unable to load or preprocess one or both of the signature images."

    samples = [test_img, reference_img]
    hog_features, cnn_features = extract_features(samples, models["CNN"], hog_indices, cnn_indices)
    integrated_features = integrate_features(hog_features, cnn_features)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(test_img, cmap='gray')
    plt.title("Test Signature")
    plt.subplot(1, 2, 2)
    plt.imshow(reference_img, cmap='gray')
    plt.title("Reference Signature")
    plt.tight_layout()
    plt.savefig("signature_comparison.png")
    plt.show()

    results = {}

    for name, clf in models.items():
        if name == "CNN":
            continue

        if name == "LSTM":
            score = clf.predict(integrated_features, verbose=0)[0][0]
            prediction = "Genuine" if score > 0.5 else "Forged"
            confidence = score if score > 0.5 else 1 - score
        else:
            pred = clf.predict([integrated_features[0]])[0]
            prediction = "Genuine" if pred == 1 else "Forged"
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba([integrated_features[0]])[0]
                confidence = proba[1] if pred == 1 else proba[0]
            else:
                confidence = None

        results[name] = {
            "prediction": prediction,
            "confidence": confidence,
            "score": score if name == "LSTM" else (proba[1] if hasattr(clf, "predict_proba") else None)
        }

    return results


def batch_test(dataset_name, models, hog_indices, cnn_indices):
    dataset_path = UTSIG_PATH if dataset_name == "UTSig" else CEDAR_PATH
    samples, labels = load_test_samples(dataset_path, dataset_name)
    hog_features, cnn_features = extract_features(samples, models["CNN"], hog_indices, cnn_indices)
    integrated_features = integrate_features(hog_features, cnn_features)

    results = {}
    thresholds = {}

    for name, clf in models.items():
        if name == "CNN":
            continue

        print(f"\nTesting {name} classifier...")

        if name == "LSTM":
            y_pred_proba = clf.predict(integrated_features, verbose=0)
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = clf.predict(integrated_features)
            y_pred_proba = clf.predict_proba(integrated_features)[:, 1] if hasattr(clf, "predict_proba") else None

        metrics = calculate_metrics(labels, y_pred, y_pred_proba)

        if y_pred_proba is not None:
            eer, threshold = calculate_equal_error_rate(labels, y_pred_proba)
            metrics['eer'] = eer
            thresholds[name] = threshold
            far, frr = calculate_far_frr(labels, y_pred_proba, threshold)
            metrics['far_at_eer'] = far
            metrics['frr_at_eer'] = frr
            metrics['roc_auc'] = plot_roc_curve(labels, y_pred_proba, f"{name} ROC Curve (Test)")

        print(f"\n{name} Classification Report:")
        print_classification_report(labels, y_pred, target_names=["Forged", "Genuine"])

        results[name] = metrics
        print(f"{name} Results:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

    if thresholds:
        plt.figure(figsize=(10, 8))
        for name, clf in models.items():
            if name == "CNN" or name not in thresholds:
                continue

            if name == "LSTM":
                y_scores = clf.predict(integrated_features, verbose=0)
            else:
                y_scores = clf.predict_proba(integrated_features)[:, 1] if hasattr(clf, "predict_proba") else None

            if y_scores is not None:
                thresholds_range = np.linspace(0, 1, 100)
                fars, frrs = [], []

                for threshold in thresholds_range:
                    far, frr = calculate_far_frr(labels, y_scores, threshold)
                    fars.append(far)
                    frrs.append(frr)

                plt.plot(thresholds_range, fars, label=f"{name} FAR")
                plt.plot(thresholds_range, frrs, linestyle='--', label=f"{name} FRR")
                plt.axvline(x=thresholds[name], color='gray', linestyle=':', label=f"{name} EER Threshold")

        plt.xlabel("Threshold")
        plt.ylabel("Error Rate")
        plt.title("FAR vs FRR Curves")
        plt.legend()
        plt.grid(True)
        plt.savefig("far_frr_curves.png")
        plt.close()

    return results


def main():
    parser = argparse.ArgumentParser(description="Signature Verification System")
    parser.add_argument("--mode", type=str, choices=["batch", "interactive"], default="interactive",
                        help="Testing mode: batch or interactive")
    parser.add_argument("--dataset", type=str, choices=["UTSig", "CEDAR"], default="UTSig",
                        help="Dataset to use for batch testing")
    parser.add_argument("--test", type=str, default=None,
                        help="Path to test signature image for interactive mode")
    parser.add_argument("--reference", type=str, default=None,
                        help="Path to reference signature image for interactive mode")

    args = parser.parse_args()

    print("Loading models and feature indices...")
    models, hog_indices, cnn_indices = load_models_and_indices()

    if args.mode == "batch":
        print(f"\nRunning batch testing on {args.dataset} dataset...")
        batch_test(args.dataset, models, hog_indices, cnn_indices)

    elif args.mode == "interactive":
        if args.test and args.reference:
            results = verify_signature(args.test, args.reference, models, hog_indices, cnn_indices)
            print("\nVerification Results:")
            for model_name, result in results.items():
                print(f"\n{model_name} classifier:")
                print(f"  Prediction: {result['prediction']}")
                if result['confidence']:
                    print(f"  Confidence: {result['confidence']:.4f}")
                if result['score']:
                    print(f"  Score: {result['score']:.4f}")
        else:
            while True:
                print("\nSignature Verification System - Interactive Mode")
                print("1. Verify a signature")
                print("2. Run batch testing")
                print("3. Exit")

                choice = input("\nEnter your choice (1-3): ")

                if choice == "1":
                    test_path = input("Enter path to test signature image: ")
                    ref_path = input("Enter path to reference signature image: ")
                    if os.path.exists(test_path) and os.path.exists(ref_path):
                        results = verify_signature(test_path, ref_path, models, hog_indices, cnn_indices)
                        print("\nVerification Results:")
                        for model_name, result in results.items():
                            print(f"\n{model_name} classifier:")
                            print(f"  Prediction: {result['prediction']}")
                            if result['confidence']:
                                print(f"  Confidence: {result['confidence']:.4f}")
                            if result['score']:
                                print(f"  Score: {result['score']:.4f}")
                    else:
                        print("Error: One or both of the specified files does not exist.")

                elif choice == "2":
                    dataset = input("Enter dataset name (UTSig/CEDAR): ")
                    if dataset in ["UTSig", "CEDAR"]:
                        batch_test(dataset, models, hog_indices, cnn_indices)
                    else:
                        print("Invalid dataset name. Please enter UTSig or CEDAR.")

                elif choice == "3":
                    print("Exiting...")
                    break

                else:
                    print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
