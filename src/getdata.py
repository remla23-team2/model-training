"""
Fetch and save the dataset
"""

import os
import pandas as pd


def get_dataset():
    """
    Get the dataset and save it to output/getdata/data.tsv
    """
    dataset = pd.read_csv(
        "data/input/a1_RestaurantReviews_HistoricDump.tsv", delimiter="\t",
        quoting=3, dtype={'Review': 'str', 'Liked': 'int'})
    dataset = dataset[['Review', 'Liked']]
    dataset.to_csv("output/getdata/data.tsv", sep="\t", quoting=3)


if __name__ == "__main__":
    os.makedirs("output/getdata", exist_ok=True)
    get_dataset()
