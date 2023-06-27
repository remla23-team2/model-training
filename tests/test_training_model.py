import pytest 
import joblib

from src.getdata import get_dataset
from src.preprocessing import pre_process
from src.evaluate import evaluate_model
from sklearn.naive_bayes import GaussianNB

def test_model_robustness():
    _, X_test, _, y_test = pre_process()
    acc_origin, _ = evaluate_model()
    for seed in [1, 2]:
        X_train, X_test, y_train, y_test = pre_process(seed=seed)
        classifier = GaussianNB()
        classifier = classifier.fit(X_train, y_train)
        acc, _ = evaluate_model(X_test=X_test, y_test=y_test)
        assert abs(acc_origin - acc) <= 0.15

