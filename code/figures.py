#%%
import json
import matplotlib.pyplot as plt
from pathlib import Path

Path("figures").mkdir(parents=True, exist_ok=True)

# Load the data for the features importance experiment: (score, feature name)
data = json.load(open("experiments_data/features_importance.json", "r"))
# Sort by order of importance
data.sort(key=lambda x: x[0], reverse=True)

# Plot a barplot of the features importance and save it
plt.barh([x[1] for x in data], [x[0] for x in data])
plt.xlabel("Loss when this feature is excluded")
plt.subplots_adjust(left=0.2)
plt.savefig("figures/features_importance.png")
plt.clf()
# %%
# Load the data for the treshold experiment: (score, treshold_train, treshold_eval)
data = json.load(open("experiments_data/treshold.json", "r"))
data_same_tresh = [d for d in data if d[1] == d[2]]
data_double_tresh = [d for d in data if d[1] != d[2]]
plt.plot([d[2] for d in data_same_tresh], [d[0] for d in data_same_tresh], label="Same treshold")
plt.plot([d[2] for d in data_double_tresh], [d[0] for d in data_double_tresh], label="Halved train treshold treshold")
plt.ylabel("Loss")
plt.xlabel("Evaluation treshold")
plt.legend()
plt.savefig("figures/treshold.png")
plt.clf()
# %%
# Load the data for the hyperparameter search: (score, neurons, regularization)
data = json.load(open("experiments_data/hyperparameters.json", "r"))
for neurons in sorted(set([d[1] for d in data])):
    plt.plot([d[2] for d in data if d[1] == neurons], [d[0] for d in data if d[1] == neurons], label=f"{neurons} neurons")
plt.ylabel("Loss")
plt.xlabel("L2 Regularization")
plt.xscale("symlog", linthresh=0.001)
plt.legend()
plt.savefig("figures/hyperparameters.png")
plt.clf()
# %%
# Load the data for the k-NN parameter search: (score, scaling factor, k)
data = json.load(open("experiments_data/knn.json", "r"))
k_values = set([d[2] for d in data])
for k in k_values:
    plt.plot([d[1] for d in data if d[2] == k], [d[0] for d in data if d[2] == k], label=f"k={k}")
plt.ylabel("Loss")
plt.xlabel("Scaling factor")
plt.legend()
plt.savefig("figures/knn.png")
plt.clf()
# %%
# Load the data for the Linear Regression parameter search: (score, regularization)
data = json.load(open("experiments_data/linear_regression.json", "r"))
plt.plot([d[1] for d in data], [d[0] for d in data])
plt.xlabel("L2 Regularization")
plt.xscale("symlog", linthresh=0.001)
plt.ylabel("Loss")
plt.savefig("figures/linear_regression.png")
plt.clf()
# %%
