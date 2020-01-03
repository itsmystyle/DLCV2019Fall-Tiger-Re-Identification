import pdb
import glob
import random
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as tvmodels
import pretrainedmodels


# class TigerNet(nn.Module):
#     def __init__(self, args, nclasses):
#         super(TigerNet, self).__init__()

#         """ declare layers used in this network"""
#         resnet50 = tvmodels.resnet50(pretrained=True)
#         self.resnet50 = nn.Sequential(*(list(resnet50.children())[:-1]))

#         for child in self.resnet50.children():
#             for param in child.parameters():
#                 param.requires_grad = False

#         self.linear = nn.Sequential(
#             nn.Linear(2048 * 2 * 4, 2048),
#             nn.BatchNorm1d(2048),
#             nn.ReLU(True),
#             nn.Linear(2048, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),
#             nn.Linear(1024, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(True),
#             nn.Linear(256, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(True),
#         )

#         self.classifier = nn.Linear(64, nclasses)
#         self.logsoftmax = nn.LogSoftmax(dim=1)

#     def forward(self, img):
#         # img = [batch, frame_len, 3, 240, 320]
#         flat = img.view(-1, img.shape[2], img.shape[3], img.shape[4])
#         img_emb = self.resnet50(flat)  # img_emb = [batch*frame_len, 2048, 2, 4]

#         img_emb = img_emb.view(img.shape[0], img.shape[1], -1)
#         img_emb = torch.mean(img_emb, 1)  # [batch, 1, 2048, 2, 4]
#         #         if img_emb.shape[0] == 1:
#         #             pdb.set_trace()
#         out = self.classifier(self.linear(img_emb))
#         results = self.logsoftmax(out)  # results = [batch, nclasses]
#         return results


from pretrained_senet import (
    SENet,
    SEResNetBottleneck,
    SEBottleneck,
    SEResNeXtBottleneck,
)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TigerNet(nn.Module):
    in_planes = 2048

    def __init__(
        self,
        num_classes,
        last_stride,
        model_path,
        neck,
        neck_feat,
        model_name,
        pretrain_choice,
    ):
        super(TigerNet, self).__init__()

        # if model_name == "se_resnet50":
        #     self.base = SENet(
        #         block=SEResNetBottleneck,
        #         layers=[3, 4, 6, 3],
        #         groups=1,
        #         reduction=16,
        #         dropout_p=None,
        #         inplanes=64,
        #         input_3x3=False,
        #         downsample_kernel_size=1,
        #         downsample_padding=0,
        #         last_stride=last_stride,
        #     )
        # elif model_name == "se_resnet101":
        #     self.base = SENet(
        #         block=SEResNetBottleneck,
        #         layers=[3, 4, 23, 3],
        #         groups=1,
        #         reduction=16,
        #         dropout_p=None,
        #         inplanes=64,
        #         input_3x3=False,
        #         downsample_kernel_size=1,
        #         downsample_padding=0,
        #         last_stride=last_stride,
        #     )

        # elif model_name == "se_resnet152":
        #     self.base = SENet(
        #         block=SEResNetBottleneck,
        #         layers=[3, 8, 36, 3],
        #         groups=1,
        #         reduction=16,
        #         dropout_p=None,
        #         inplanes=64,
        #         input_3x3=False,
        #         downsample_kernel_size=1,
        #         downsample_padding=0,
        #         last_stride=last_stride,
        #     )
        # elif model_name == "se_resnext50":
        #     self.base = SENet(
        #         block=SEResNeXtBottleneck,
        #         layers=[3, 4, 6, 3],
        #         groups=32,
        #         reduction=16,
        #         dropout_p=None,
        #         inplanes=64,
        #         input_3x3=False,
        #         downsample_kernel_size=1,
        #         downsample_padding=0,
        #         last_stride=last_stride,
        #     )
        # elif model_name == "se_resnext101":
        #     self.base = SENet(
        #         block=SEResNeXtBottleneck,
        #         layers=[3, 4, 23, 3],
        #         groups=32,
        #         reduction=16,
        #         dropout_p=None,
        #         inplanes=64,
        #         input_3x3=False,
        #         downsample_kernel_size=1,
        #         downsample_padding=0,
        #         last_stride=last_stride,
        #     )
        # elif model_name == "senet154":
        #     self.base = SENet(
        #         block=SEBottleneck,
        #         layers=[3, 8, 36, 3],
        #         groups=64,
        #         reduction=16,
        #         dropout_p=0.2,
        #         last_stride=last_stride,
        #     )

        self.base = pretrainedmodels.se_resnet101(
            num_classes=1000, pretrained="imagenet"
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == "no":
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == "bnneck":
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x, is_Train):
        """
            x : [batch, 3, 384, 384]
            x after doing backbone : [batch, 2048, 12, 12]
        """
        global_feat = self.gap(self.base.features(x))  # [batch, 2048, 1, 1]
        global_feat = global_feat.view(global_feat.shape[0], -1)  # [batch, 2048]

        if self.neck == "no":
            feat = global_feat
        elif self.neck == "bnneck":
            feat = self.bottleneck(global_feat)  # normalize for angular softmax

        # if self.training:
        if is_Train:
            cls_score = self.classifier(feat)
            cls_score = F.log_softmax(cls_score)  # [4, 72]
            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == "after":
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path, cpu=False):
        if cpu:
            param_dict = torch.load(trained_path, map_location="cpu")
        else:
            param_dict = torch.load(trained_path)
        for i in param_dict:
            if "classifier" in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
