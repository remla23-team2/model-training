# this code is only for testing!

import pandas as pd
import numpy as np
import sys

sys.path.append("./")
from src.preprocessing import pre_process
from src.evaluate import evaluate_model
from sklearn.naive_bayes import GaussianNB

seed = 10
classifier = GaussianNB()

dataset = pd.read_csv(
    "data/input/a1_RestaurantReviews_HistoricDump.tsv", delimiter="\t", quoting=3
)
sliced_dataset = dataset[
    dataset["Review"].apply(lambda x: len(x.split()) <= 5)
].reset_index(drop=True)
# short_reviews = dataset[dataset['Review'].apply(lambda x: len(x.split()) <= 5)]

X_train, X_test, y_train, y_test = pre_process(seed)
X_sliced_train, X_sliced_test, y_sliced_train, y_sliced_test = pre_process(
    seed, dataset=sliced_dataset
)

classifier_fulldata = classifier.fit(X_train, y_train)

print(
    dataset,
    "\n\n\n",
    dataset["Review"],
    "\n\n\n",
    sliced_dataset,
    "\n\n\n",
    len(dataset),
    len(sliced_dataset),
    np.shape(y_test),
    np.shape(X_test),
    np.shape(X_sliced_test),
    "\n\n\n",
    X_test,
    X_sliced_test,
)

acc_full_data, _ = evaluate_model(
    classifier=classifier_fulldata, X_test=X_test, y_test=y_test
)
acc_data_slice, _ = evaluate_model(
    classifier=classifier_fulldata, X_test=X_test, y_test=y_test
)
