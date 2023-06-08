import pytest
import sys
import pandas as pd

sys.path.append('./')
from src.preprocessing import pre_process
from src.evaluate import evaluate_model
from sklearn.naive_bayes import GaussianNB

seed = 10
classifier = GaussianNB()

@pytest.fixture
def preprocess_fixture():
    dataset = pd.read_csv("data/input/a1_RestaurantReviews_HistoricDump.tsv", delimiter="\t", quoting=3)
    # sliced_dataset = dataset[dataset['Review'].apply(lambda x: len(x.split()) <= 5)]
    sliced_dataset = dataset[dataset['Review'].apply(lambda x: len(x.split()) <= 5)].reset_index(drop=True)
    X_train, X_test, y_train, y_test = pre_process(dataset, seed)
    X_sliced_train, X_sliced_test, y_sliced_train, y_sliced_test = pre_process(sliced_dataset, seed)

    return X_train, X_test, y_train, y_test, X_sliced_train, X_sliced_test, y_sliced_train, y_sliced_test

def test_preprocess(preprocess_fixture):
    # The preprocessing has already been done by the fixture
    # You can write your test assertions here -> I copied the code below from ProffesorGPT
    X_train, X_test, y_train, y_test, _, _, _, _ = preprocess_fixture
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0

def test_model_performance(preprocess_fixture):
    # Test performance of model on full data
    X_train, X_test, y_train, y_test, _, X_sliced_test, _, y_sliced_test = preprocess_fixture
    classifier_fulldata = classifier.fit(X_train, y_train)
    
    acc_full_data, _ = evaluate_model(classifier_fulldata, X_test, y_test)
    acc_data_slice, _ = evaluate_model(classifier_fulldata, X_sliced_test, y_sliced_test)

    # Assert that the difference in performance is less than 0.1
    assert abs(acc_full_data - acc_data_slice) <= 0.1

