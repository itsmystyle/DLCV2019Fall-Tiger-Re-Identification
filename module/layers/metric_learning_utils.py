import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    """generalize mean pooling
    ref:
    https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution
    """

    def __init__(self, p=3.0, eps=1e-6, freeze_p=True):
        super().__init__()
        self.p = p if freeze_p else Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        if isinstance(self.p, float):
            p = self.p
        else:
            p = self.p.data.tolist()[0]
        return self.__class__.__name__ + "(p=%s, eps=%s)" % (p, self.eps)


class ArcMarginProduct(nn.Module):
    """ Implement of large margin arc distance
    ref:
    https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution
    https://arxiv.org/abs/1801.07698

    Parameters
    ----------
    in_features: int
        size of each input sample
    out_features: int
        size of each output sample
    multi_task: bool
        as setting multi-task, expected the label is with same shape as (batch_size, out_features)
    s: float
        norm of input feature
    m: float
        cos(theta + m)
    """

    def __init__(
        self, in_features, out_features, multi_task, s=30.0, m=0.50, device=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if multi_task:
            self.label_dim = self.out_features
        else:
            self.label_dim = 1
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.device = device

    def forward(self, input, label):
        # cos(theta) & phi(theta)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # convert label to one-hot
        if self.device:
            one_hot = torch.zeros(cosine.size(), device=self.device)
        else:
            one_hot = torch.zeros(cosine.size(), device="cpu")

        one_hot.scatter_(1, label.view(-1, self.label_dim).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output
