"""
Test the 'Model Development' angle
"""
import joblib
import pytest
from sklearn.naive_bayes import GaussianNB

from src.preprocessing import pre_process
from src.evaluate import evaluate_model

@pytest.fixture()
def baseline_model():
    """
    Fixture that returns the baseline model
    Yields:
        GaussianNB: the baseline model
    """
    bl_model = GaussianNB()
    yield bl_model

@pytest.fixture()
def our_model():
    """
    Fixture that returns our model
    Yields:
        LogisticRegression: our model
    """
    with open("output/train/sentiment_model", "rb") as file:
        model = joblib.load(file)
    yield model

def test_against_baseline(baseline_model, our_model):
    """
    Test that our model is better than the baseline

    Args:
        baseline_model (GaussianNB): baseline model
        our_model (LogisticRegression): model to test against the baseline
    """
    X_train, X_test, y_train, y_test = pre_process()

    # Fit the baseline model
    baseline_model = baseline_model.fit(X_train, y_train)

    # Test the models
    baseline_accuracy, _ = evaluate_model(baseline_model, X_test, y_test)
    our_accuracy, _ = evaluate_model(our_model, X_test, y_test)

    # Assert that our model is better than the baseline
    assert our_accuracy > baseline_accuracy
