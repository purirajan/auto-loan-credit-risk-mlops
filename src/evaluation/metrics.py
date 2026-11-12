"""Model performance metrics shared by local and MLflow training."""

from sklearn.metrics import roc_auc_score


def evaluate_binary_classifier(model, X_test, y_test):
    """Return AUC, Gini, and predicted PDs for a fitted classifier."""
    predicted_pd = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, predicted_pd)
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "predicted_pd": predicted_pd,
    }
