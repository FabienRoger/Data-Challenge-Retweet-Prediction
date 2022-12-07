#%%
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Load the data for the features importance experiment: (score, feature name)
data = json.load(open("scores.json", "r"))
# Sort by order of importance
data.sort(key=lambda x: x[0], reverse=True)

# Plot a barplot of the features importance and save it
plt.title("Features importance")
plt.barh([x[1] for x in data], [x[0] for x in data])
plt.xlabel("Loss when this feature is excluded")
plt.savefig("figures/features_importance.png")
# %%
# Load the data for the treshold experiment: (score, treshold_train, treshold_eval)
data = json.load(open("scores2.json", "r"))
data_same_tresh = [d for d in data if d[1] == d[2]]
data_double_tresh = [d for d in data if d[1] != d[2]]
plt.title("Score depending on the treshold")
plt.plot([d[2] for d in data_same_tresh], [d[0] for d in data_same_tresh], label="Same treshold")
plt.plot([d[2] for d in data_double_tresh], [d[0] for d in data_double_tresh], label="Halved train treshold treshold")
plt.ylabel("Score")
plt.xlabel("Evaluation treshold")
plt.legend()
plt.savefig("figures/treshold.png")
# %%
# Load the data for the hyperparameter search: (score, neurons, regularization)
plt.title("Score depending on the regularization and number of hidden neurons")
data = json.load(open("scores3.json", "r"))
for neurons in set([d[1] for d in data]):
    plt.plot([d[2] for d in data if d[1] == neurons], [d[0] for d in data if d[1] == neurons], label=f"{neurons} neurons")
plt.ylabel("Score")
plt.xlabel("L2 Regularization")
plt.xscale("log")
plt.legend()
plt.savefig("figures/hyperparameters.png")
# %%
# Load the data for the k-NN parameter search: (score, k, scaling factor)
data = json.load(open("scores5.json", "r"))