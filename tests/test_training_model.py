import pytest 
import joblib

from src.getdata import get_dataset
from src.preprocessing import pre_process
from src.evaluate import evaluate_model
from sklearn.naive_bayes import GaussianNB

@pytest.fixture()
def trained_model():
    trained_model = joblib.load('data/models/c2_Classifier_Sentiment_Model')
    yield trained_model

def test_model_robustness(trained_model):
    dataset = get_dataset('data/input/a1_RestaurantReviews_HistoricDump.tsv')
    _, X_test, _, y_test = pre_process(dataset, 10)
    acc_origin, _ = evaluate_model(trained_model, X_test, y_test)
    for seed in [1, 2]:
        X_train, X_test, y_train, y_test = pre_process(dataset, seed)
        classifier = GaussianNB()
        Classifier = classifier.fit(X_train, y_train)
        acc, _ = evaluate_model(Classifier, X_test, y_test)
        assert abs(acc_origin - acc) <= 0.1

