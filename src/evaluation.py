import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

def evaluate_model(model, X_test, y_test, model_type="sklearn", threshold=0.5):
    """
    Evaluates the model on test data.
    Uses predict_proba/decision_function for sklearn models and predict for LSTM.
    Returns a dictionary of metrics and prediction scores.
    """
    try:
        if model is None:
            raise ValueError("Model is None.")
        if model_type in ["svm", "knn"]:
            if hasattr(model, "predict_proba"):
                y_scores = model.predict_proba(X_test)[:, 1]
            else:
                y_scores = model.decision_function(X_test)
                y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-6)
            y_pred = (y_scores >= threshold).astype("int32")
        else:
            y_scores = model.predict(X_test).flatten()
            y_pred = (y_scores >= threshold).astype("int32")
        scores = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }
        if len(np.unique(y_test)) == 2:
            scores["roc_auc"] = roc_auc_score(y_test, y_scores)
        else:
            scores["roc_auc"] = None
            print("Warning: Only one class present in y_test. ROC AUC is undefined.")
        return scores, y_scores, y_pred
    except Exception as e:
        print("Error during evaluation:", e)
        return {}, None, None

if __name__ == "__main__":
    import numpy as np
    y_true = np.random.randint(0, 2, 20)
    y_pred_dummy = np.random.randint(0, 2, 20)
    scores, _, _ = evaluate_model(lambda x: y_pred_dummy, None, y_true, model_type="svm")
    print(scores)
