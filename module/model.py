import torch.nn as nn
import torchvision.models as models


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], -1)


class ReIDNET(nn.Module):
    def __init__(self, num_classes):
        super(ReIDNET, self).__init__()

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
