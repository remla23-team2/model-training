import pytest
from pathlib import Path
from src.preprocessing import pre_process
import pandas as pd

seed = 10

@pytest.fixture
def preprocess_fixture():
    dataset = pd.read_csv("data/input/a1_RestaurantReviews_HistoricDump.tsv", delimiter="\t", quoting=3)
    return pre_process(dataset,seed)

def test_preprocess(preprocess_fixture):
    # The preprocessing has already been done by the fixture
    # You can write your test assertions here -> I copied the code below from ProffesorGPT
    X_train, X_test, y_train, y_test = preprocess_fixture
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0