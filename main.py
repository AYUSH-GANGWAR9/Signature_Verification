import os
import argparse
import numpy as np
import tensorflow as tf

# GPU configuration: ensure CUDA-enabled GPU is used
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Detected: {gpus}")
    except RuntimeError as e:
        print("⚠️ Error setting GPU memory growth:", e)
else:
    print("❌ No GPU found. Running on CPU.")

from src.data_preprocessing import load_images_from_folder, preprocess_image
from src.feature_extraction import build_cnn_feature_extractor, build_vgg16_feature_extractor, build_resnet_feature_extractor, extract_hybrid_features
from src.training import prepare_dataset, extract_features_from_dataset, train_classifiers, load_trained_models
from src.evaluation import evaluate_model
from src.visualization import plot_confusion_matrix, plot_roc_curve

def main(args):
    svm_model, knn_model, lstm_model, selector = load_trained_models()
    if args.feature_extractor.lower() == "vgg16":
        cnn_model = build_vgg16_feature_extractor()
    elif args.feature_extractor.lower() == "resnet":
        cnn_model = build_resnet_feature_extractor()
    else:
        cnn_model = build_cnn_feature_extractor()

    if cnn_model is None:
        print("CNN model could not be built. Exiting.")
        return

    if any(model is None for model in [svm_model, knn_model, lstm_model, selector]):
        print("Saved models not found/incomplete. Training models...")
        images, labels = prepare_dataset(args.dataset_folder)
        if len(images) == 0:
            print("No images found. Exiting.")
            return
        X = extract_features_from_dataset(images, cnn_model, use_hybrid=True)
        try:
            svm_model, knn_model, lstm_model, selector = train_classifiers(
                X, labels, epochs=args.epochs, sample_size=args.sample_size, selection_method=args.selection_method)
        except Exception as e:
            print("Error during classifier training:", e)
            return

    images, labels = prepare_dataset(args.dataset_folder)
    X = extract_features_from_dataset(images, cnn_model, use_hybrid=True)
    split = int(0.8 * len(X))
    X_test = X[split:]
    y_test = labels[split:]
    try:
        X_test_selected = selector.transform(X_test)
    except Exception as e:
        print("Error applying feature selector:", e)
        return
    X_test_lstm = X_test_selected.reshape((X_test_selected.shape[0], X_test_selected.shape[1], 1))

    print("Evaluating SVM on test set...")
    svm_scores, svm_y_scores, _ = evaluate_model(svm_model, X_test_selected, y_test, model_type="svm")
    print("SVM Evaluation:", svm_scores)
    plot_confusion_matrix(svm_scores["confusion_matrix"], title="SVM Confusion Matrix")
    if svm_scores["roc_auc"] is not None:
        plot_roc_curve(y_test, svm_y_scores, title="SVM ROC Curve")

    print("Evaluating KNN on test set...")
    knn_scores, knn_y_scores, _ = evaluate_model(knn_model, X_test_selected, y_test, model_type="knn")
    print("KNN Evaluation:", knn_scores)
    plot_confusion_matrix(knn_scores["confusion_matrix"], title="KNN Confusion Matrix")
    if knn_scores["roc_auc"] is not None:
        plot_roc_curve(y_test, knn_y_scores, title="KNN ROC Curve")

    print("Evaluating LSTM on test set...")
    lstm_scores, lstm_y_scores, _ = evaluate_model(lstm_model, X_test_lstm, y_test, model_type="lstm")
    print("LSTM Evaluation:", lstm_scores)
    plot_confusion_matrix(lstm_scores["confusion_matrix"], title="LSTM Confusion Matrix")
    if lstm_scores["roc_auc"] is not None:
        plot_roc_curve(y_test, lstm_y_scores, title="LSTM ROC Curve")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Signature Verification Training & Evaluation")
    parser.add_argument("--dataset_folder", type=str, required=True, help="Path to dataset folder (e.g., ./data)")
    parser.add_argument("--epochs", type=int, default=45, help="Number of epochs for LSTM training")
    parser.add_argument("--sample_size", type=int, default=1000, help="Optional stratified sample size")
    parser.add_argument("--selection_method", type=str, default="decision_tree", help="Feature selection: decision_tree or pca")
    parser.add_argument("--feature_extractor", type=str, default="hybrid", help="Feature extractor: hybrid (default), vgg16, or resnet")
    parser.add_argument("--use_grid_search", action="store_true", help="Use grid search for hyperparameter tuning")
    args = parser.parse_args()
    main(args)
