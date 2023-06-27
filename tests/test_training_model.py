"""
This module contains tests for training the model of the project.
"""
import pytest 
import joblib
from sklearn.naive_bayes import GaussianNB
from src.preprocessing import pre_process
from src.evaluate import evaluate_model

@pytest.fixture()
def trained_model():
    """
    This fixture loads a trained model for testing.
    """
    trained_model = joblib.load('data/models/c2_Classifier_Sentiment_Model')
    yield trained_model

def test_model_robustness(trained_model):
    """
    Test the robustness of the model by comparing 
    the performances with different seeds.
    """
    _, X_test, _, y_test = pre_process(10)
    acc_origin, _ = evaluate_model()
    for seed in [1, 2]:
        X_train, X_test, y_train, y_test = pre_process(seed)
        classifier = GaussianNB()
        Classifier = classifier.fit(X_train, y_train)
        acc, _ = evaluate_model()
        assert abs(acc_origin - acc) <= 0.1

