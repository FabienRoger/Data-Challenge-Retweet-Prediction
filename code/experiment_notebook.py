#%%
# Imports
import csv
import json
from datetime import datetime
from math import log10

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split
from tensorflow import keras
from utils.mlp import get_trained_model
from utils.tree import Tree
from utils.data_processing import (
    extract_bucketized_features,
    extract_continuous_features,
)

#%%
# Load data
all_df = pd.read_csv("data/train.csv")

train_df, val_df = train_test_split(all_df, test_size=0.1, random_state=1)

#%%
# Data preprocessing for tree-based models

X_train, buckets_borders, cols = extract_bucketized_features(train_df)
y_train = train_df["retweets_count"].values
X_val, _, _ = extract_bucketized_features(val_df)
y_val = val_df["retweets_count"].values

nb_cat = [X_train[:, i].max() + 1 for i in range(len(cols))]

#%%
#%%
# train tree
tree = Tree(min_bucket_size=300, nb_cat=nb_cat, cols_n=list(range(len(cols)))).fit(
    X_train, y_train
)
#%%
# Run tree on new data
val_preds = tree(X_val)
#%%
# Try variants of the MLP model

attempts = [(100, 100)]
scores = []
for thresh, eval_tresh in attempts:
    print()
    print(thresh, eval_tresh)

    xltdf = train_df[train_df["favorites_count"] > thresh].copy()
    X_train, y_train, ms = extract_continuous_features(xltdf)
    X_val, y_val, _ = extract_continuous_features(val_df, mean_and_std=ms)

    model = get_trained_model(X_train, y_train)

    # evaluate the model and the quality of its predictions on the validation set
    val_predictions = model.predict(X_val)

    losses_tree = []
    losses_nn = []
    for i, (_, row) in enumerate(val_df.iterrows()):
        if row["favorites_count"] <= eval_tresh:
            losses_tree.append(abs(val_preds[i] - y_val[i]).item())
        else:
            losses_nn.append(abs(val_predictions[i] - y_val[i]).item())

    y_ = train_df[train_df["favorites_count"] <= thresh]["retweets_count"].values
    print("nn difficulty", "tree difficulty")
    print((y_train - np.median(y_train)).mean(), (y_ - np.median(y_)).mean())
    print("tree", sum(losses_tree) / len(losses_tree), sum(losses_tree))
    print("nn", sum(losses_nn) / len(losses_nn), sum(losses_nn))
    losses = losses_nn + losses_tree
    overall = sum(losses) / len(losses)
    print("overall", overall)
    scores.append((overall, thresh, eval_tresh))

# save the experiments results
json.dump(scores, open("scores.json", "w"))
