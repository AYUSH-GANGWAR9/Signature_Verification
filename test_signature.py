import os
import argparse
import numpy as np
from src.data_preprocessing import preprocess_image
from src.feature_extraction import build_cnn_feature_extractor, build_vgg16_feature_extractor, build_resnet_feature_extractor, extract_hybrid_features
from src.training import load_trained_models

def predict_signature(model, features, model_type="svm", threshold=0.5):
    """
    For a single sample, predicts the class and probability.
    Uses predict_proba/decision_function for sklearn models and predict for LSTM.
    """
    if model_type in ["svm", "knn"]:
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(features)[0][1]
        else:
            score = model.decision_function(features)[0]
            prob = (score - score.min()) / (score.max() - score.min() + 1e-6)
        pred = int(prob >= threshold)
    else:
        prob = model.predict(features).flatten()[0]
        pred = int(prob >= threshold)
    return {"predicted_label": pred, "probability": prob}

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
        print("Saved models not found. Please run training first.")
        return

    try:
        img = preprocess_image(args.image_path, size=(128, 128))
    except Exception as e:
        print("Error processing image:", e)
        return

    features = extract_hybrid_features(img, cnn_model)
    if features.size == 0:
        print("Feature extraction failed.")
        return
    try:
        features_selected = selector.transform(np.array([features]))
    except Exception as e:
        print("Error applying feature selector:", e)
        return
    features_lstm = features_selected.reshape((features_selected.shape[0], features_selected.shape[1], 1))

    svm_result = predict_signature(svm_model, features_selected, model_type="svm")
    print("SVM prediction:", svm_result)

    knn_result = predict_signature(knn_model, features_selected, model_type="knn")
    print("KNN prediction:", knn_result)

    lstm_result = predict_signature(lstm_model, features_lstm, model_type="lstm")
    print("LSTM prediction:", lstm_result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test custom signature input for verification.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the signature image file")
    parser.add_argument("--feature_extractor", type=str, default="hybrid", help="Feature extractor: hybrid, vgg16, or resnet")
    args = parser.parse_args()
    main(args)
