"""
Evaluating trained sentiment model.
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
    with open("output/preprocess/X_test.pkl", "rb") as file_x_test:
        X_test = pickle.load(file_x_test)
    with open("output/preprocess/y_test.pkl", "rb") as file_y_test:
        y_test = pickle.load(file_y_test)
    return classifier, X_test, y_test

def evaluate_model(classifier=None, X_test=None, y_test=None):
    """
    Function to evaluate model performance
    """
    if classifier is None or X_test is None or y_test is None:
        classifier, X_test, y_test = _load_artifacts()
    y_pred = classifier.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    return acc, conf_matrix

def save_metrics(acc, conf_matrix):
    """
    Save the accuracy and confusion matrix to file

    Args:
        acc (float): Accuracy score of the model
        conf_matrix (np.array): Confusion matrix of the model
    """
    with open("output/evaluate/cm.pkl", "wb") as file:
        pickle.dump(conf_matrix, file)
    with open("output/metrics/metrics.json", "w", encoding="utf-8") as file:
        json.dump({"accuracy": acc}, file)

def main():
    """
    Main function to evaluate the model and save the metrics
    """
    acc, conf_matrix = evaluate_model()
    save_metrics(acc, conf_matrix)

if __name__ == "__main__":
    os.makedirs("output/evaluate", exist_ok=True)
    os.makedirs("output/metrics", exist_ok=True)
    main()
