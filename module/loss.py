import torch
import torch.nn as nn

from module.utils import normalize, euclidean_dist, hard_example_mining


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, device="cpu"):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.device = device

    def forward(self, inputs, targets):
        targets = (
            torch.zeros(inputs.shape)
            .scatter_(1, targets.unsqueeze(1).cpu(), 1)
            .to(self.device)
        )
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * inputs).mean(0).sum()
        return loss


class CenterLoss(nn.Module):
    """Center loss.
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=72, feat_dim=2048, device="cpu"):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim)).to(
            self.device
        )

    def forward(self, inputs, labels):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert (
            inputs.shape[0] == labels.shape[0]
        ), "Batch size of features and labels are not equal."

        batch_size = inputs.size(0)
        distmat = (
            torch.pow(inputs, 2)
            .sum(dim=1, keepdim=True)
            .expand(batch_size, self.num_classes)
            + torch.pow(self.centers, 2)
            .sum(dim=1, keepdim=True)
            .expand(self.num_classes, batch_size)
            .t()
        )
        distmat.addmm_(1, -2, inputs, self.centers.t())

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None, dist="euclidean"):
        self.margin = margin
        if margin is not None:
            if dist == "euclidean":
                self.ranking_loss = nn.MarginRankingLoss(margin=margin)
            elif dist == "cos":
                self.ranking_loss = nn.CosineEmbeddingLoss(margin=margin)
        else:
            if dist == "euclidean":
                self.ranking_loss = self.ranking_loss = nn.SoftMarginLoss()
            elif dist == "cos":
                self.ranking_loss = nn.CosineEmbeddingLoss(margin=0)

    def __call__(self, inputs, labels, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = euclidean_dist(inputs, inputs)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an.unsqueeze(0), dist_ap.unsqueeze(0), y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)

        return loss
