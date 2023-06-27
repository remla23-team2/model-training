"""
Module for evaluating trained sentiment model.
"""

import os
import json
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

def _load_artifacts():
    """
    Function to load model artifacts from file
    """
    classifier = joblib.load("output/train/sentiment_model")
    with open("output/preprocess/X_test.pkl", "rb") as f:
        X_test = pickle.load(f)
    with open("output/preprocess/y_test.pkl", "rb") as f:
        y_test = pickle.load(f)
    return classifier, X_test, y_test

def evaluate_model(classifier=None, X_test=None, y_test=None):
    """
    Function to evaluate model performance
    """
    if classifier is None or X_test is None or y_test is None:
        classifier, X_test, y_test = _load_artifacts()
    y_pred = classifier.predict(X_test)

    confusion_mat = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    with open("output/evaluate/cm.pkl", "wb") as f:
        pickle.dump(confusion_mat, f)
    with open("output/metrics/metrics.json", "w", encoding='utf-8') as f:
        json.dump({"accuracy": acc}, f)
    return acc, confusion_mat

def main():
    """
    Main function to call model evaluation
    """
    evaluate_model()

if __name__ == "__main__":
    os.makedirs("output/evaluate", exist_ok=True)
    os.makedirs("output/metrics", exist_ok=True)
    main()

# import json
# from sklearn.metrics import confusion_matrix, accuracy_score
# import joblib
# import pickle
# import os


# def _load_artifacts():
#     classifier = joblib.load("output/train/sentiment_model")
#     X_test = pickle.load(open("output/preprocess/X_test.pkl", "rb"))
#     y_test = pickle.load(open("output/preprocess/y_test.pkl", "rb"))
#     return classifier, X_test, y_test


# def evaluate_model(classifier=None, X_test=None, y_test=None):
#     if classifier is None or X_test is None or y_test is None:
#         classifier, X_test, y_test = _load_artifacts()
#     y_pred = classifier.predict(X_test)

#     cm = confusion_matrix(y_test, y_pred)
#     acc = accuracy_score(y_test, y_pred)
#     pickle.dump(cm, open("output/evaluate/cm.pkl", "wb"))
#     json.dump({"accuracy": acc}, open("output/metrics/metrics.json", "w"))
#     return acc, cm


# def main():
#     evaluate_model()


# if __name__ == "__main__":
#     os.makedirs("output/evaluate", exist_ok=True)
#     os.makedirs("output/metrics", exist_ok=True)
#     main()
