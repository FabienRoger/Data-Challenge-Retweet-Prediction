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

#%%
# Load data
all_df = pd.read_csv("data/train.csv")

train_df, val_df = train_test_split(all_df, test_size=0.1, random_state=1)

#%%
# Data preprocessing for tree-based models
to_bucketize = {
    "word_count": 5,
    "tweet_len": 5,
    "timestamp": 6,
    "favorites_count": 160,
    "statuses_count": 6,
    "friends_count": 10,
    "followers_count": 10,
    "hour": 4,
}  # , "day":7}


def pre_process_tree(df, buckets_borders=None):
    cols = ["urls_count", "hashtags_count", "verified"]

    df["urls_count"] = df.urls.map(lambda x: min(len(x.split(",")), 2) - 1)
    df["hashtags_count"] = df.hashtags.map(lambda x: min(len(x.split(",")), 5) - 1)
    df["word_count"] = df.text.map(lambda x: len(x))
    df["tweet_len"] = df.text.map(lambda x: len(x.split()))
    df["hour"] = df.timestamp.apply(lambda x: datetime.fromtimestamp(x / 1000.0).hour)

    if buckets_borders is None:
        buckets_borders = {}
    for col_name, n_buckets in to_bucketize.items():

        if col_name in buckets_borders:
            bd = buckets_borders[col_name]
        else:
            values = df[col_name].values
            quartiles = np.linspace(0, 1, n_buckets + 1)
            bd = sorted(list(set([np.quantile(values, q) for q in quartiles])))
            bd[0] -= 1
            bd[-1] += 1
            buckets_borders[col_name] = bd

        true_n_buckets = len(bd) - 1
        new_name = f"{col_name}_b_{true_n_buckets}"
        cols.append(new_name)
        df[new_name] = pd.cut(df[col_name], bins=bd, labels=list(range(true_n_buckets)))

    X = df[cols].values

    return X, buckets_borders, cols


X_train, buckets_borders, cols = pre_process_tree(train_df)
y_train = train_df["retweets_count"].values
X_val, _, _ = pre_process_tree(val_df)
y_val = val_df["retweets_count"].values

nb_cat = [X_train[:, i].max() + 1 for i in range(len(cols))]

#%%
#%%
# train tree
tree = Tree(min_bucket_size=300, nb_cat=nb_cat, cols_n=list(range(len(cols))))(
    X_train, y_train
)
#%%
# Run tree on new data
val_preds = tree(X_val)

#%%
# preprocessing for MLP model

important_words = [
    "rt",
    "fav",
    "favorie",
    "favories",
    "rewteet",
    "retweets",
    "click",
    "macron",
    "lepen",
    "melenchon",
]


def pre_process(df, train=True):
    if train:
        df["logrt"] = df.retweets_count.map(lambda x: log10(x + 1))
    df["logfav"] = df.favorites_count.map(lambda x: log10(x + 1))
    df["logfriend"] = df.friends_count.map(lambda x: log10(x + 1))
    df["logstatus"] = df.statuses_count.map(lambda x: log10(x + 1))
    df["text_len"] = df.text.map(lambda x: len(x) / 140)
    df["word_count"] = df.text.map(lambda x: len(x.split()) / 140)
    df["normed_time"] = df.timestamp.map(lambda t: log10(1.64775e12 - t))
    for w in important_words:
        df[w] = df.text.map(lambda t: 1 if w in [word.lower() for word in t] else 0)
    df["hour"] = df.timestamp.apply(lambda x: datetime.fromtimestamp(x / 1000.0).hour)
    df["day"] = df.timestamp.apply(
        lambda x: datetime.fromtimestamp(x / 1000.0).weekday()
    )
    Xt = df[
        [
            "logfav",
            "text_len",
            "logfriend",
            "word_count",
            "logstatus",
            "normed_time",
            "verified",
            *important_words,
        ]
    ].values
    if train:
        yt = df["retweets_count"].values[:, None]
        return Xt, yt
    else:
        return Xt


#%%
# Try variants of the MLP model

attempts = [(100, 100)]
scores = []
for thresh, eval_tresh in attempts:
    print()
    print(thresh, eval_tresh)

    xltdf = train_df[train_df["favorites_count"] > thresh].copy()
    X_train, y_train = pre_process(xltdf)
    X_val, y_val = pre_process(val_df)

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
