from getdata import get_dataset
import os

def test_get_data():
    get_dataset()
    assert os.path.exists("output/getdata/data.tsv") == True