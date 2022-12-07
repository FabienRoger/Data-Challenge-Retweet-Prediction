#%%
# Imports
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.data_processing import (
    extract_bucketized_features,
    extract_continuous_features,
    extract_many_continuous_features,
)
from utils.mlp import get_trained_model
from utils.tree import Tree

#%%
# Load data
all_df = pd.read_csv("data/train.csv")

train_df, val_df = train_test_split(all_df, test_size=0.3, random_state=1)

#%%
# Data preprocessing for tree-based models

X_train, buckets_borders, cols = extract_bucketized_features(train_df)
y_train = train_df["retweets_count"].values
X_val, _, _ = extract_bucketized_features(val_df)
y_val = val_df["retweets_count"].values

nb_cat = [X_train[:, i].max() + 1 for i in range(len(cols))]

#%%
# train tree
tree = Tree(min_bucket_size=300, nb_cat=nb_cat, cols_n=list(range(len(cols)))).fit(
    X_train, y_train
)
#%%
# Run tree on new data
val_preds = tree(X_val)
#%%
# # Experiment: remove features one by one and see how it affects the performance

# cols = []
# # words and weekdays will be added by extract_many_continuous_features

# _, n_features = extract_many_continuous_features(val_df, cols=cols)[0].shape

# assert n_features == len(cols)

# # Thresholds for using the Tree or the MLP model
# thresh, eval_tresh = 100, 100

# scores = []
# for removed_feature, col in enumerate(cols):
#     print()
#     print(removed_feature, col)

#     xltdf = train_df[train_df["favorites_count"] > thresh].copy()
#     X_train, y_train, ms = extract_many_continuous_features(xltdf)
#     X_val, y_val, _ = extract_many_continuous_features(val_df, mean_and_std=ms)

#     X_train = X_train[:, [i for i in range(len(cols)) if i != removed_feature]]
#     X_val = X_val[:, [i for i in range(len(cols)) if i != removed_feature]]

#     model = get_trained_model(X_train, y_train, epochs=1000)

#     # evaluate the model and the quality of its predictions on the validation set
#     val_predictions = model.predict(X_val)

#     losses_tree = []
#     losses_nn = []
#     for i, (_, row) in enumerate(val_df.iterrows()):
#         if row["favorites_count"] <= eval_tresh:
#             losses_tree.append(abs(val_preds[i] - y_val[i]).item())
#         else:
#             losses_nn.append(abs(val_predictions[i] - y_val[i]).item())

#     y_ = train_df[train_df["favorites_count"] <= thresh]["retweets_count"].values
#     print("nn difficulty", "tree difficulty")
#     print((y_train - np.median(y_train)).mean(), (y_ - np.median(y_)).mean())
#     print("tree", sum(losses_tree) / len(losses_tree), sum(losses_tree))
#     print("nn", sum(losses_nn) / len(losses_nn), sum(losses_nn))
#     losses = losses_nn + losses_tree
#     overall = sum(losses) / len(losses)
#     print("overall", overall)

#     scores.append((overall, col))

# # save the experiments results
# json.dump(scores, open("scores.json", "w"))
# #%%
# # Experiment: change the threshold for for using the Tree or the MLP model and see how it affects the performance

# scores = []
# for thresh, eval_tresh in [
#     (50, 50),
#     (100, 100),
#     (200, 200),
#     (500, 500),
#     (25, 50),
#     (50, 100),
#     (100, 200),
#     (250, 500),
# ]:
#     print()
#     print(thresh, eval_tresh)

#     xltdf = train_df[train_df["favorites_count"] > thresh].copy()
#     X_train, y_train, ms = extract_continuous_features(xltdf)
#     X_val, y_val, _ = extract_continuous_features(val_df, mean_and_std=ms)

#     model = get_trained_model(X_train, y_train, epochs=1000)

#     # evaluate the model and the quality of its predictions on the validation set
#     val_predictions = model.predict(X_val)

#     losses_tree = []
#     losses_nn = []
#     for i, (_, row) in enumerate(val_df.iterrows()):
#         if row["favorites_count"] <= eval_tresh:
#             losses_tree.append(abs(val_preds[i] - y_val[i]).item())
#         else:
#             losses_nn.append(abs(val_predictions[i] - y_val[i]).item())

#     y_ = train_df[train_df["favorites_count"] <= thresh]["retweets_count"].values
#     print("nn difficulty", "tree difficulty")
#     print((y_train - np.median(y_train)).mean(), (y_ - np.median(y_)).mean())
#     print("tree", sum(losses_tree) / len(losses_tree), sum(losses_tree))
#     print("nn", sum(losses_nn) / len(losses_nn), sum(losses_nn))
#     losses = losses_nn + losses_tree
#     overall = sum(losses) / len(losses)
#     print("overall", overall)

#     scores.append((overall, thresh, eval_tresh))

# # save the experiments results
# json.dump(scores, open("scores2.json", "w"))

# #%%
# # Experiment: change the number of units and the regularization of the MLP model and see how it affects the performance

# # Thresholds for using the Tree or the MLP model
# thresh, eval_tresh = 100, 100

# scores = []
# for units, reg in [(u, r) for u in [4, 16, 64, 128] for r in [0.0, 0.001, 0.1, 10.0]]:
#     print()
#     print(units, reg)

#     xltdf = train_df[train_df["favorites_count"] > thresh].copy()
#     X_train, y_train, ms = extract_continuous_features(xltdf)
#     X_val, y_val, _ = extract_continuous_features(val_df, mean_and_std=ms)

#     model = get_trained_model(X_train, y_train, epochs=1000, units=units, reg=reg)

#     # evaluate the model and the quality of its predictions on the validation set
#     val_predictions = model.predict(X_val)

#     losses_tree = []
#     losses_nn = []
#     for i, (_, row) in enumerate(val_df.iterrows()):
#         if row["favorites_count"] <= eval_tresh:
#             losses_tree.append(abs(val_preds[i] - y_val[i]).item())
#         else:
#             losses_nn.append(abs(val_predictions[i] - y_val[i]).item())

#     y_ = train_df[train_df["favorites_count"] <= thresh]["retweets_count"].values
#     print("nn difficulty", "tree difficulty")
#     print((y_train - np.median(y_train)).mean(), (y_ - np.median(y_)).mean())
#     print("tree", sum(losses_tree) / len(losses_tree), sum(losses_tree))
#     print("nn", sum(losses_nn) / len(losses_nn), sum(losses_nn))
#     losses = losses_nn + losses_tree
#     overall = sum(losses) / len(losses)
#     print("overall", overall)

#     scores.append((overall, units, reg))

# # save the experiments results
# json.dump(scores, open("scores3.json", "w"))

#%%
# Experiment: add a third part to the model: a linear model for the tweets with many favorites

scores = []
for thresh1, thresh2 in [(100,50_000), (100,10_000), (50,10_000)]:
    print()
    print(thresh1, thresh2)

    xltdf = train_df[(train_df["favorites_count"] > thresh1) & (train_df["favorites_count"] < thresh2)].copy()
    X_train, y_train, ms = extract_continuous_features(xltdf)
    X_val, y_val, _ = extract_continuous_features(val_df, mean_and_std=ms)
    model1 = get_trained_model(X_train, y_train, epochs=1000)
    
    xltdf2 = train_df[train_df["favorites_count"] > thresh2].copy()
    X_train2, y_train2, _ = extract_continuous_features(xltdf2, mean_and_std=ms)
    X_train2 = X_train2[:, 0:1] # Only keep the favorites count
    
    model2 = get_trained_model(X_train2, y_train2, linear=True, epochs=1000)

    val_predictions = model1.predict(X_val)
    val_predictions2 = model2.predict(X_val[:, 0:1])

    losses_tree = []
    losses_nn = []
    loss_above = []
    for i, (_, row) in enumerate(val_df.iterrows()):
        if row['favorites_count'] >= thresh2:
            loss_above.append(abs(val_predictions2[i] - y_val[i]).item())
        elif row['favorites_count'] <= thresh1:
            losses_tree.append(abs(val_preds[i] - y_val[i]).item())
        else:
            losses_nn.append(abs(val_predictions[i] - y_val[i]).item())

    y_ = train_df[train_df["favorites_count"] <= thresh1]["retweets_count"].values
    print("nn difficulty", "tree difficulty")
    print((y_train - np.median(y_train)).mean(), (y_ - np.median(y_)).mean())
    print("tree",sum(losses_tree) / len(losses_tree), sum(losses_tree))
    print("nn",sum(losses_nn) / len(losses_nn), sum(losses_nn))
    print("above",sum(loss_above) / len(loss_above), sum(loss_above))
    losses = losses_nn + losses_tree + loss_above
    overall = sum(losses) / len(losses)
    print("overall",overall)
    scores.append((overall, thresh1, thresh2))
json.dump(scores, open("scores4.json", "w"))

#%%
# Experiment: baseline model: k-NN

from sklearn.neighbors import KNeighborsRegressor

alpha_values = [2,10,20]
k_values=[3,15]
scores = []
for alpha in alpha_values:
    for k in k_values:
        print()
        print(alpha, k)

        X_train, y_train, ms = extract_continuous_features(train_df)
        X_val, y_val, _ = extract_continuous_features(val_df, mean_and_std=ms)
        X_val[:,0]*=alpha
        X_train[:,0]*=alpha

        model=KNeighborsRegressor(n_neighbors=k)
        model.fit(X_train, y_train)
        
        # evaluate the model and the quality of its predictions on the validation set
        val_predictions = model.predict(X_val)

        
        losses_nn = []
        for i, (_, row) in enumerate(val_df.iterrows()):
            losses_nn.append(abs(val_predictions[i] - y_val[i]).item())


        losses = losses_nn
        overall = sum(losses) / len(losses)
        print("overall", overall)
        scores.append((overall, alpha, k))

# save the experiments results
json.dump(scores, open("scores5.json", "w"))

#%%
# Experiment: baseline model: linear regression

scores = []

for reg in [0.0, 0.001, 0.1, 10.0]:
    print()
    print(reg)

    X_train, y_train, ms = extract_continuous_features(train_df)
    X_val, y_val, _ = extract_continuous_features(val_df, mean_and_std=ms)

    # evaluate the model and the quality of its predictions on the validation set
    model = get_trained_model(X_train, y_train, reg=reg, epochs=50, linear=True)
    
    val_predictions = model.predict(X_val)

    
    losses_nn = []
    for i, (_, row) in enumerate(val_df.iterrows()):
        losses_nn.append(abs(val_predictions[i] - y_val[i]).item())


    losses = losses_nn
    overall = sum(losses) / len(losses)
    print("overall", overall)
    scores.append((overall, reg))

# save the experiments results
json.dump(scores, open("scores6.json", "w"))

# %%
# Experiment: baseline model: decision tree alone

losses = []
for i, (_, row) in enumerate(val_df.iterrows()):
    losses.append(abs(val_preds[i] - y_val[i]).item())
overall = sum(losses) / len(losses)
print("overall", overall)
json.dump(overall, open("scores7.json", "w"))

# %%
# Experiment: baseline model: decision tree alone

X_train, y_train, ms = extract_continuous_features(train_df)
X_val, y_val, _ = extract_continuous_features(val_df, mean_and_std=ms)

# 50 epochs is enough since it runs on the whole dataset
model = get_trained_model(X_train, y_train, epochs=50)

# evaluate the model and the quality of its predictions on the validation set
val_predictions = model.predict(X_val)

losses = []
for i, (_, row) in enumerate(val_df.iterrows()):
    losses.append(abs(val_preds[i] - y_val[i]).item())
overall = sum(losses) / len(losses)
print("overall", overall)
json.dump(overall, open("scores8.json", "w"))