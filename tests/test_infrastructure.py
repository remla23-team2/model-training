"""
Test the 'ML Infrastructure' angle
"""

import os
import pandas as pd

from src.getdata import get_dataset
from src.preprocessing import pre_process
from src.train import train
from src.evaluate import evaluate_model, save_metrics


def test_pipeline_integration():
    """
    Ingration test of ML pipeline
    """
    # Get the data
    get_dataset()
    assert os.path.exists("output/getdata/data.tsv")
    data = pd.read_csv("output/getdata/data.tsv", delimiter="\t", quoting=3,
                       dtype={'Review': str, 'Liked': int})
    data = data[['Review', 'Liked']]

    # Preprocess the data
    X_train, X_test, y_train, y_test = pre_process(data)
    assert os.path.exists("output/preprocess/X_test.pkl")
    assert os.path.exists("output/preprocess/y_test.pkl")
    assert os.path.exists("output/preprocess/X_train.pkl")
    assert os.path.exists("output/preprocess/y_train.pkl")
    assert os.path.exists("output/preprocess/cv.pkl")

    # Train the model
    classifier = train(X_train, y_train)
    assert os.path.exists("output/train/sentiment_model")

    # Evaluate the model
    acc, confusion_matrix = evaluate_model(classifier, X_test, y_test)
    # Save the metrics
    save_metrics(acc, confusion_matrix)
    assert os.path.exists("output/evaluate/cm.pkl")
    assert os.path.exists("output/metrics/metrics.json")
