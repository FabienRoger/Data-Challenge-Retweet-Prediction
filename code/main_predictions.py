#%%
# Imports
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from math import log10


import csv
from utils.mlp import get_trained_model

from utils.tree import Tree
from utils.data_processing import (
    extract_bucketized_features,
    extract_continuous_features,
)

#%%
# Load data
train_df = pd.read_csv("data/train.csv")
ev_df = pd.read_csv("data/evaluation.csv")

#%%

X_train, buckets_borders, cols = extract_bucketized_features(train_df)
y_train = train_df["retweets_count"].values
X_ev, _, _ = extract_bucketized_features(ev_df)

nb_cat = [X_train[:, i].max() + 1 for i in range(len(cols))]
#%%
# train tree
tree = Tree(min_bucket_size=300, nb_cat=nb_cat, cols_n=list(range(len(cols)))).fit(
    X_train, y_train
)
#%%
# Run tree on test data
preds = tree(X_ev)

#%%
# Train MLP model
thresh, eval_tresh = 100, 100

xltdf = train_df[train_df["favorites_count"] > thresh].copy()
X_train, y_train, ms = extract_continuous_features(xltdf)


model = get_trained_model(X_train, y_train)


#%%
# Run tree on test data
X_test = extract_continuous_features(ev_df, train=False, mean_and_std=ms)
y_test = model.predict(X_test)[:, 0]
name = f"predictions{thresh}_{eval_tresh}.txt"
with open(name, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["TweetID", "retweets_count"])
    for i, row in ev_df.iterrows():
        pred = preds[i] if row["favorites_count"] <= eval_tresh else y_test[i]
        writer.writerow([row["TweetID"], str(int(pred))])
