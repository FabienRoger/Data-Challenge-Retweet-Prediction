import datetime
from typing import Optional
import numpy as np
import pandas as pd
from math import log10

DEFAULT_BUCKETIZE = {
    "word_count": 5,
    "tweet_len": 5,
    "timestamp": 6,
    "favorites_count": 160,
    "statuses_count": 6,
    "friends_count": 10,
    "followers_count": 10,
    "hour": 4,
}


def extract_bucketized_features(
    df: pd.DataFrame,
    buckets_borders: Optional[list[float]] = None,
    to_bucketize=DEFAULT_BUCKETIZE,
):
    """Return the features X (np.ndarray), the bucket borders and the columns names.

    buckets_borders is a dict which tells the borders of the buckets
    (it should be computed on the training set).
    to_bucketize tells which columns to bucketize and how many buckets to use."""
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


DEFAULT_IMPORTANT_WORDS = [
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


def extract_continuous_features(
    df: pd.DataFrame,
    train=True,
    important_words=DEFAULT_IMPORTANT_WORDS,
    mean_and_std: Optional[tuple[np.ndarray, np.ndarray]] = None,
):
    """If train, return the features X (np.ndarray), the labels y (np.ndarray), and the mean_and_std
    else return only the features X  (np.ndarray)

    important_words is a list of words which are used to create new features

    mean_and_std is a tuple (mean, std) which is used to normalize the features,
    (it should be computed on the training set).
    """

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

    # Normalization
    if mean_and_std is None:
        mean = Xt.mean(axis=0)
        std = Xt.std(axis=0)
        mean_and_std = (mean, std)
    Xt = (Xt - mean_and_std[0]) / mean_and_std[1]

    if train:
        yt = df["retweets_count"].values[:, None]
        return Xt, yt
    else:
        return Xt
