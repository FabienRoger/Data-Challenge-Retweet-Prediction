from attrs import define
import numpy as np


@define
class Tree:
    """A tree regressor that uses the MAE as decision criterion."""

    min_bucket_size: int
    nb_cat: list[int]
    cols_n: int

    def can_cut(self, x, c):
        """Check if node is leaf or not"""
        return all(
            (x[:, c] == i).sum() >= self.min_bucket_size for i in range(self.nb_cat[c])
        )

    def loss(self, y):
        """Compute the loss of a node"""
        return np.abs(y - np.median(y)).sum()

    def cut(self, x, *arrays, c=-1):
        """Cut a node into sub-nodes"""
        masks = [x[:, c] == i for i in range(self.nb_cat[c])]
        return [(x[m], *[a[m] for a in arrays]) for m in masks]

    def fit(self, x, y, depth=0):
        """Returns a function that predicts the y value for a given x value"""

        # If the node is a leaf, return the median of the y values
        possible_cuts = [c for c in self.cols_n if self.can_cut(x, c)]
        if not possible_cuts:
            return lambda x: np.full((x.shape[0],), np.median(y))

        # Find the best cut
        cuts_losses = [
            (sum(self.loss(sub_y) for (sub_x, sub_y) in self.cut(x, y, c=c)), c)
            for c in possible_cuts
        ]
        best_cut = min(cuts_losses)[1]

        # Recursively build the tree
        child_fns = []
        for x_c, y_c in self.cut(x, y, c=best_cut):
            child_fns.append(self.fit(x_c, y_c, depth=depth + 1))

        def pred(x_pred):
            idxs = np.arange(0, x_pred.shape[0])
            y_pred = np.empty(x_pred.shape[0])
            for (x_pred_c, idxs_c), fn_c in zip(
                self.cut(x_pred, idxs, c=best_cut), child_fns
            ):
                y_pred[idxs_c] = fn_c(x_pred_c)

            return y_pred

        return pred
