import pytest
import sys

import pandas as pd

sys.path.append("./")
from src.preprocessing import pre_process
from src.evaluate import evaluate_model

@pytest.fixture()
def dataset():
    dataset = pd.read_csv("output/getdata/data.tsv", delimiter="\t", quoting=3)
    yield dataset

def test_data_slices(dataset):
    # Preprocess the full dataset
    _, X_test, _, y_test = pre_process(dataset=dataset)
    
    # Create sliced dataset
    sliced_dataset = dataset[dataset["Review"].apply(lambda x: len(x.split()) <= 5)].reset_index(drop=True)
    
    # Preprocess the sliced dataset
    _, X_test_sliced, _, y_test_sliced = pre_process(dataset=sliced_dataset)
    
    acc_full_data, _ = evaluate_model(X_test=X_test, y_test=y_test)
    acc_data_slice, _ = evaluate_model(X_test=X_test_sliced, y_test=y_test_sliced)
    assert (acc_full_data - acc_data_slice) <= 0.15
