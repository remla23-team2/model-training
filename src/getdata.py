import pandas as pd

def get_dataset():
    dataset = pd.read_csv('data/input/a1_RestaurantReviews_HistoricDump.tsv', delimiter='\t', quoting=3)
    dataset.to_csv('output/getdata/data.tsv', sep='\t', quoting=3)

if __name__ == '__main__':
    get_dataset()