import numpy as np
import torch

from metrics.utils import re_ranking


class Metrics:
    def __init__(self):
        self.name = "Metric Name"

    def reset(self):
        pass

    def update(self, predicts, targets):
        pass

    def get_score(self):
        pass


class MulticlassAccuracy(Metrics):
    """ Multiclass Classification Accuracy
    """

    def __init__(self):
        self.name = "Acc."
        self.n_correct = 0
        self.n = 1e-20

    def reset(self):
        self.n_correct = 0
        self.n = 1e-20

    def update(self, predicts, targets):
        predicts = torch.exp(predicts).max(dim=1)[1]
        self.n_correct += (predicts == targets).sum().item()
        self.n += targets.shape[0]

    def get_score(self):
        return self.n_correct / self.n

    def print_score(self):
        score = self.get_score()
        return "{:.5f}".format(score)


class Accuracy(Metrics):
    """ Accuracy
    """

    def __init__(self):
        self.name = "Rank 1"
        self.n_correct = 0
        self.n = 1e-20

    def reset(self):
        self.n_correct = 0
        self.n = 1e-20

    def update(self, predicts, targets):
        self.n_correct += (predicts == targets).sum().item()
        self.n += targets.shape[0]

    def get_score(self):
        return self.n_correct / self.n

    def print_score(self):
        score = self.get_score()
        return "{:.5f}".format(score)


class ReRankingAccuracy(Metrics):
    """ Accuracy with re-ranking
    """

    def __init__(
        self, num_query, max_rank=35, feat_norm=True, k1=20, k2=6, lambda_value=0.3
    ):
        self.name = "Re-Rank 1"
        self.n_correct = 0
        self.n = 1e-20

        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

        self.k1 = k1
        self.k2 = k2
        self.lambda_value = lambda_value

        self.reset()

    def reset(self):
        self.n_correct = 0
        self.n = 1e-20

        self.feats = []
        self.labels = []

    def update(self, predicts, targets):
        # predicts are features and targets are labels at here

        self.feats.append(predicts)
        self.labels.append(targets)

    def get_score(self):
        feats = np.concatenate(self.feats)
        labels = np.concatenate(self.labels)

        query = feats[-self.num_query:]
        targets = labels[-self.num_query:]

        gallery = feats[: -self.num_query]
        gallery_labels = labels[: -self.num_query]

        distmat = re_ranking(
            query, gallery, k1=self.k1, k2=self.k2, lambda_value=self.lambda_value
        )

        self.n_correct = (gallery_labels[distmat.argmin(axis=1)] == targets).sum()
        self.n = targets.shape[0]

        return self.n_correct / self.n

    def print_score(self):
        score = self.get_score()
        return "{:.5f}".format(score)
