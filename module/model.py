import torch
import torch.nn as nn
import torchvision.models as models
from pretrainedmodels.models import (
    se_resnext50_32x4d,
    se_resnext101_32x4d,
    se_resnet152,
    nasnetalarge,
)

from module.layers.metric_learning_utils import GeM, ArcMarginProduct
from module.layers.utils import weights_init_kaiming


class ResNet152(nn.Module):
    def __init__(self, num_classes, fc_dim=512):
        super(ResNet152, self).__init__()

        self.num_classes = num_classes
        self.fc_dim = fc_dim

        backbone = models.resnet152(pretrained=True)
        self.backbone = nn.Sequential(*(list(backbone.children())[:-1]))
        self.fc = nn.Linear(2048 * 10 * 10, self.fc_dim)
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
        logits = self.classifier(x)

        return logits, x

    def extract_features(self, x):
        x = self.backbone(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.bn(x)

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
        logits = self.classifier(x)

        return logits, x

    def extract_features(self, x):
        x = self.backbone(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.bn(x)

        return x


class SeResNet152(nn.Module):
    def __init__(self, num_classes, fc_dim):
        super(SeResNet152, self).__init__()

        self.num_classes = num_classes
        self.fc_dim = fc_dim

        self.backbone = se_resnet152(num_classes=1000, pretrained="imagenet")
        final_in_features = self.backbone.last_linear.in_features

        self.fc = nn.Linear(final_in_features * 10 * 10, self.fc_dim)
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
        logits = self.classifier(x)

        return logits, x

    def extract_features(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avg_pool(x).view(x.shape[0], -1)
        x = self.fc(x)
        x = self.bn(x)

        return x


class SeResNeXt50(nn.Module):
    def __init__(self, num_classes, fc_dim):
        super(SeResNeXt50, self).__init__()

        self.num_classes = num_classes
        self.fc_dim = fc_dim

        self.backbone = se_resnext50_32x4d(num_classes=1000, pretrained="imagenet")
        final_in_features = self.backbone.last_linear.in_features

        self.fc = nn.Linear(final_in_features * 10 * 10, self.fc_dim)
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
        logits = self.classifier(x)

        return logits, x

    def extract_features(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avg_pool(x).view(x.shape[0], -1)
        x = self.fc(x)
        x = self.bn(x)

        return x


class NASNet(nn.Module):
    def __init__(self, num_classes, fc_dim):
        super(NASNet, self).__init__()

        self.num_classes = num_classes
        self.fc_dim = fc_dim

        self.backbone = nasnetalarge(num_classes=1000, pretrained="imagenet")
        final_in_features = self.backbone.last_linear.in_features

        self.fc = nn.Linear(final_in_features * 6 * 6, self.fc_dim)
        self.bn = nn.BatchNorm1d(self.fc_dim)
        self._init_params()

        self.classifier = nn.Sequential(
            nn.Linear(self.fc_dim, self.num_classes), nn.LogSoftmax(dim=1)
        )

    def _init_params(self):
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.extract_features(x)
        logits = self.classifier(x)

        return logits, x

    def extract_features(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avg_pool(x).view(x.shape[0], -1)
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

        return self.logsoftmax(logits), x

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

        return self.logsoftmax(logits), x

    def extract_features(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        if self.use_fc:
            x = self.fc(x)
            x = self.bn(x)

        return x


class SeResNeXtArcFaceModel(nn.Module):
    def __init__(
        self, n_classes, scale, margin, fc_dim, model=50, use_fc=False, device=None,
    ):
        super(SeResNeXtArcFaceModel, self).__init__()
        if model == 50:
            self.backbone = se_resnext50_32x4d(num_classes=1000, pretrained="imagenet")
        else:
            self.backbone = se_resnext101_32x4d(num_classes=1000, pretrained="imagenet")
        final_in_features = self.backbone.last_linear.in_features

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

        return self.logsoftmax(logits), x

    def extract_features(self, x):
        batch_size = x.shape[0]
        x = self.backbone.features(x)
        x = self.pooling(x).view(batch_size, -1)

        if self.use_fc:
            x = self.fc(x)
            x = self.bn(x)

        return x


class FRNet(nn.Module):
    def __init__(
        self,
        n_classes,
        scale,
        margin,
        fc_dim,
        cls_dim,
        init_img_size,
        model=50,
        use_fc=False,
        device=None,
    ):
        super(FRNet, self).__init__()

        # Encoder
        if model == 50:
            self.backbone = se_resnext50_32x4d(num_classes=1000, pretrained="imagenet")
        else:
            self.backbone = se_resnext101_32x4d(num_classes=1000, pretrained="imagenet")
        final_in_features = self.backbone.last_linear.in_features
        self.pooling = GeM()
        self.use_fc = use_fc
        self.fc_dim = fc_dim
        if use_fc:
            self.fc = nn.Sequential(
                nn.Linear(final_in_features, fc_dim), nn.BatchNorm1d(fc_dim)
            )
            self.fc.apply(weights_init_kaiming)

        # Decoder
        self.init_img_size = init_img_size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                fc_dim // init_img_size ** 2, 64 * 8, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),  # [batch, 512, 8, 8]
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),  # [batch, 256, 16, 16]
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),  # [batch, 128, 32, 32]
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),  # [batch, 64, 64, 64]
            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),  # [batch, 64, 128, 128]
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),  # [batch, 32, 256, 256]
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),  # [batch, 16, 512, 512]
            nn.Conv2d(16, 3, 3, 1, 1, bias=False),
        )
        self.decoder.apply(weights_init_kaiming)

        # Classifier
        self.cls_dim = cls_dim
        self.final = ArcMarginProduct(
            self.cls_dim, n_classes, multi_task=False, s=scale, m=margin, device=device,
        )
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, ref_x, label):
        x = self._encoder(x)
        # _, f_x = x[:, : -self.cls_dim], x[:, -self.cls_dim :]
        f_x = x[:, -self.cls_dim:]

        ref_x = self._encoder(ref_x)
        # b_ref_x, _ = ref_x[:, : -self.cls_dim], ref_x[:, -self.cls_dim :]
        ref_x = ref_x[:, : -self.cls_dim]

        # id classification
        logits = self.final(f_x, label)
        logits = self.logsoftmax(logits)

        # reconstruct images
        recon_img = torch.cat([ref_x, f_x], dim=1)
        recon_img = recon_img.view(
            label.shape[0], -1, self.init_img_size, self.init_img_size
        )
        recon_img = self._decode(recon_img)

        return logits, f_x, recon_img

    def _decode(self, x):
        x = self.decoder(x)

        return x

    def _encoder(self, x):
        x = self.backbone.features(x)
        x = self.pooling(x).view(x.shape[0], -1)

        if self.use_fc:
            x = self.fc(x)

        return x

    def extract_features(self, x):
        x = self._encoder(x)

        return x[:, -self.cls_dim:]
