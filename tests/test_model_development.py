import joblib
import pytest
from sklearn.naive_bayes import GaussianNB

from src.preprocessing import pre_process
from src.evaluate import evaluate_model

@pytest.fixture()
def baseline_model():
    baseline_model = GaussianNB()
    yield baseline_model

@pytest.fixture()
def our_model():
    our_model = joblib.load(open("output/train/sentiment_model", "rb"))
    yield our_model

def test_against_baseline(baseline_model, our_model):
    X_train, X_test, y_train, y_test = pre_process()
    
    # Fit the baseline model
    baseline_model = baseline_model.fit(X_train, y_train)
    
    # Test the models
    baseline_accuracy, _ = evaluate_model(baseline_model, X_test, y_test)
    our_accuracy, _ = evaluate_model(our_model, X_test, y_test)
    
    # Assert that our model is better than the baseline
    assert our_accuracy > baseline_accuracy
    
    