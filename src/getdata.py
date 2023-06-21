import pandas as pd
import os
    
def get_dataset(filename):
    dataset = pd.read_csv('data/input/a1_RestaurantReviews_HistoricDump.tsv', delimiter='\t', quoting=3)
    dataset.to_csv('output/getdata/data.tsv', sep='\t', quoting=3)
    return dataset

if __name__ == '__main__':
    os.makedirs("output/getdata", exist_ok=True)
    get_dataset()