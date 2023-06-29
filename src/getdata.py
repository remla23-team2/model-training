"""
Fetch and save the dataset
"""

import os
import pandas as pd
import gdown

def get_dataset():
    """
    Get the dataset and save it to output/getdata/data.tsv
    """
    url = 'https://drive.google.com/uc?id=15Ud_ABNmAjZLK3MPMCzp96KxbnVKx5uY'
    output = 'output/getdata/data.tsv'
    gdown.download(url, output, quiet=False)


if __name__ == "__main__":
    os.makedirs("output/getdata", exist_ok=True)
    get_dataset()
