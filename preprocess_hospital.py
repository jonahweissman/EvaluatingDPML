import csv
import numpy as np
import pickle
import random
from sklearn.preprocessing import normalize

DATA_DIR = 'texas/100'
DATASET_NAME = 'hospital'
NUM_ROWS = 60_000
verbose = True


def normalizeDataset(X):
    mods = np.linalg.norm(X, axis=1)
    return X / mods[:, np.newaxis]

if __name__ == '__main__':
    if verbose: print("Opening feats file")
    with open('%s/feats' % DATA_DIR) as csv_file:
        csv_reader = csv.reader(csv_file)
        features = np.array(list(csv_reader), dtype=int)
    if verbose: print("Opening labels file")
    with open('%s/labels' % DATA_DIR) as csv_file:
        csv_reader = csv.reader(csv_file)
        labels_list = list(csv_reader)
        labels = np.squeeze(np.array(labels_list, dtype=int)) - 1

    if verbose: print("Randomly selecting %i rows" % NUM_ROWS)
    assert NUM_ROWS <= len(features), \
        "Cannot choose more rows than exist in dataset"
    sampled_rows = np.random.choice(
            len(features),
            size=NUM_ROWS,
            replace=False
        )

    if verbose: print("Saving data to pickle files")
    pickle.dump(normalizeDataset(features[sampled_rows]),
            open(f'{DATASET_NAME}_features.p', 'wb'))
    pickle.dump(labels[sampled_rows],
            open(f'{DATASET_NAME}_labels.p', 'wb'))
