#%%
import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
from math import log10
import matplotlib

Path("figures").mkdir(parents=True, exist_ok=True)
# plt.subplots_adjust(left=0.2)

# Load the data for the features importance experiment: (score, feature name)
data = json.load(open("experiments_data/features_importance.json", "r"))
# Sort by order of importance
data.sort(key=lambda x: x[0], reverse=True)

# Plot a barplot of the features importance and save it
plt.clf()
plt.barh([x[1] for x in data], [x[0] for x in data])
plt.xlabel("Loss when this feature is excluded")
plt.savefig("figures/features_importance.png", bbox_inches='tight')
# %%
# Load the data for the threshold experiment: (score, treshold_train, treshold_eval)
data = json.load(open("experiments_data/threshold.json", "r"))
data_same_tresh = [d for d in data if d[1] == d[2]]
data_double_tresh = [d for d in data if d[1] != d[2]]
plt.clf()
plt.plot([d[2] for d in data_same_tresh], [d[0] for d in data_same_tresh], label="Same threshold")
plt.plot([d[2] for d in data_double_tresh], [d[0] for d in data_double_tresh], label="Halved train threshold threshold")
plt.ylabel("Loss")
plt.xlabel("Evaluation threshold")
plt.legend()
plt.savefig("figures/threshold.png", bbox_inches='tight')
# %%
# Load the data for the hyperparameter search: (score, neurons, regularization)
data = json.load(open("experiments_data/hyperparameters.json", "r"))
plt.clf()
for neurons in sorted(set([d[1] for d in data])):
    plt.plot([d[2] for d in data if d[1] == neurons], [d[0] for d in data if d[1] == neurons], label=f"{neurons} neurons")
plt.ylabel("Loss")
plt.xlabel("L2 Regularization")
plt.xscale("symlog", linthresh=0.001)
plt.legend()
plt.savefig("figures/hyperparameters.png", bbox_inches='tight')
# %%
# Load the data for the k-NN parameter search: (score, scaling factor, k)
data = json.load(open("experiments_data/knn.json", "r"))
k_values = set([d[2] for d in data])
plt.clf()
for k in k_values:
    plt.plot([d[1] for d in data if d[2] == k], [d[0] for d in data if d[2] == k], label=f"k={k}")
plt.ylabel("Loss")
plt.xlabel("Scaling factor")
plt.legend()
plt.savefig("figures/knn.png", bbox_inches='tight')
# %%
# Load the data for the Linear Regression parameter search: (score, regularization)
data = json.load(open("experiments_data/linear_regression.json", "r"))
plt.clf()
plt.plot([d[1] for d in data], [d[0] for d in data])
plt.xlabel("L2 Regularization")
plt.xscale("symlog", linthresh=0.001)
plt.ylabel("Loss")
plt.savefig("figures/linear_regression.png", bbox_inches='tight')
# %%
# Display distribution of retweets vs followers counts
train_df = pd.read_csv("data/train.csv")
plt.clf()
fig, ax = plt.subplots(figsize=(8,6))
matplotlib.rcParams.update({'font.size': 15})
for i in range(4):
    train_df[train_df["retweets_count"] == i]["followers_count"].map(lambda x: log10(x+1)).hist(ax=ax, alpha=0.3,bins=100, label=f"retweets_count = {i}")
train_df[train_df["retweets_count"] >= 4]["followers_count"].map(lambda x: log10(x+1)).hist(ax=ax, alpha=0.3,bins=100, label=f"retweets_count >= {4}")
plt.xlabel('Log of followers_count')
plt.ylabel('Nb of tweets')
plt.legend()
plt.savefig("figures/followers_retweets.png", bbox_inches='tight')
# %%
# Display distribution of retweets vs followers counts
plt.clf()
fig, ax = plt.subplots(figsize=(8,6))
matplotlib.rcParams.update({'font.size': 13})
for i in range(4):
    train_df[train_df["retweets_count"] == i]["timestamp"].map(lambda x: max(x, 1.646e12)).hist(ax=ax, alpha=0.3, label=f"retweets_count = {i}", bins=100)
train_df[train_df["retweets_count"] >= 4]["timestamp"].map(lambda x: max(x, 1.646e12)).hist(ax=ax, alpha=0.3, label=f"retweets_count >= {4}", bins=100)
plt.legend()
#ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
plt.xlabel('Timestamp')
plt.ylabel('Nb of tweets')
plt.xticks(rotation=45)
plt.savefig("figures/timestamps_retweets.png", bbox_inches='tight')
#%%
# Display distribution of log retweets vs log favorites counts
plt.clf()
plt.scatter(train_df["retweets_count"].map(lambda x: log10(x+1)),train_df["favorites_count"].map(lambda x: log10(x+1)),alpha=0.01)
plt.xlabel('Log of retweets_count')
plt.ylabel('Log of favorites_count')
plt.savefig("figures/favorites_retweets.png", bbox_inches='tight')
# %%
