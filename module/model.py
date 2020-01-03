import torch
import torch.nn as nn
import torchvision.models as models

from module.layers.utils import Flatten
from module.layers.metric_learning_utils import GeM, ArcMarginProduct


class ResNet152(nn.Module):
    def __init__(self, num_classes):
        super(ResNet152, self).__init__()

        self.num_classes = num_classes

        backbone = models.resnet152(pretrained=True)
        self.backbone = nn.Sequential(*(list(backbone.children())[:-1]))

        self.classifier = nn.Sequential(
            Flatten(), nn.Linear(2048 * 10 * 10, self.num_classes), nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)

        return x

    def extract_features(self, x):
        x = self.backbone(x)
        x = x.view(x.shape[0], -1)

        return x


class SeResNet50(nn.Module):
    def __init__(self, num_classes, fc_dim):
        super(SeResNet50, self).__init__()

        self.num_classes = num_classes
        self.fc_dim = fc_dim

        backbone = torch.hub.load(
            "moskomule/senet.pytorch", "se_resnet50", pretrained=True
        )
        self.backbone = nn.Sequential(*(list(backbone.children())[:-1]))
        self.fc = nn.Linear(2048, self.fc_dim)
        self.bn = nn.BatchNorm1d(self.fc_dim)
        self._init_params()

        self.classifier = nn.Sequential(
            nn.Linear(self.fc_dim, self.num_classes), nn.LogSoftmax(dim=1)
        )

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.extract_features(x)
        x = self.classifier(x)

        return x

    def extract_features(self, x):
        x = self.backbone(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.bn(x)

        return x


class ResNetArcFaceModel(nn.Module):
    def __init__(
        self, n_classes, scale, margin, fc_dim, use_fc=False, device=None,
    ):
        super(ResNetArcFaceModel, self).__init__()
        backbone = models.resnet152(pretrained=True)
        self.backbone = nn.Sequential(*(list(backbone.children())[:-2]))

        self.pooling = GeM()
        final_in_features = 2048

        self.use_fc = use_fc
        if use_fc:
            self.fc = nn.Linear(2048, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            final_in_features = fc_dim

        self.final = ArcMarginProduct(
            final_in_features,
            n_classes,
            multi_task=False,
            s=scale,
            m=margin,
            device=device,
        )

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, label):
        x = self.extract_features(x)
        logits = self.final(x, label)

        return self.logsoftmax(logits)

    def extract_features(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        if self.use_fc:
            x = self.fc(x)
            x = self.bn(x)

        return x


class SeResNetArcFaceModel(nn.Module):
    def __init__(
        self, n_classes, scale, margin, fc_dim, use_fc=False, device=None,
    ):
        super(SeResNetArcFaceModel, self).__init__()
        backbone = torch.hub.load(
            "moskomule/senet.pytorch", "se_resnet50", pretrained=True
        )
        self.backbone = nn.Sequential(*(list(backbone.children())[:-2]))

        self.pooling = GeM()
        final_in_features = 2048

        self.use_fc = use_fc
        if use_fc:
            self.fc = nn.Linear(2048, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            final_in_features = fc_dim

        self.final = ArcMarginProduct(
            final_in_features,
            n_classes,
            multi_task=False,
            s=scale,
            m=margin,
            device=device,
        )

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, label):
        x = self.extract_features(x)
        logits = self.final(x, label)

        return self.logsoftmax(logits)

    def extract_features(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        if self.use_fc:
            x = self.fc(x)
            x = self.bn(x)

        return x
