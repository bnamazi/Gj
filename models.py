import torch
import torchvision
from torchvision import models
import torch.nn as nn
import math
from torch.nn.modules.utils import _triple
import timm
from pytorchvideo.models import x3d
import os

os.environ["HTTPS_PROXY"] = "http://proxy.swmed.edu:3128"


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        self.model = models.wide_resnet50_2(pretrained=True)
        self.num_ftrs = self.model.fc.in_features

        self.fc = nn.Linear(self.num_ftrs, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, frame_num):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        x = x + frame_num
        # return x

        return [self.fc(x)]


class CNN_timm(nn.Module):
    def __init__(self, num_classes):
        super(CNN_timm, self).__init__()

        self.model = timm.create_model(
            model_name="efficientnet_b3", pretrained=True, num_classes=0
        )
        # self.num_ftrs = self.model.get_classifier().in_features
        # print(self.num_ftrs)
        print(self.model)

        self.fc = nn.Linear(1536, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, frame_num):
        x = self.model(x)

        x = x + frame_num
        # return x

        return [self.fc(x)]


class CNN_3D(nn.Module):
    def __init__(self, num_classes):
        super(CNN_3D, self).__init__()

        self.model = torch.hub.load(
            "facebookresearch/pytorchvideo", "x3d_m", pretrained=True
        )  # x3d.create_x3d(model_num_class=num_classes)

        # print(self.model.blocks[5])
        # print(self.model.blocks[-1].proj)

        self.model.blocks[-1].proj = nn.Linear(2048, num_classes)
        self.model.blocks[-1].activation = None
        self.fc = nn.Linear(2048, num_classes)
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x, frame_num):
        x = self.model.blocks[0](x)
        x = self.model.blocks[1](x)
        x = self.model.blocks[2](x)
        x = self.model.blocks[3](x)
        x = self.model.blocks[4](x)
        x = self.model.blocks[5].pool(x)
        x = self.model.blocks[5].dropout(x)

        x = self.pool(x)

        x = torch.squeeze(x)
        x = x + frame_num
        x = self.fc(x)

        # x = self.pool(x)

        # x = self.model(x)

        return [x]


class TwoStream(nn.Module):
    def __init__(self, num_classes):
        super(TwoStream, self).__init__()

        self.cnn = timm.create_model(
            model_name="efficientnet_b3", pretrained=True, num_classes=0
        )

        self.cnn3d = torch.hub.load(
            "facebookresearch/pytorchvideo", "x3d_m", pretrained=True
        )

        self.cnn3d.blocks[-1].proj = nn.Linear(2048, num_classes)
        self.cnn3d.blocks[-1].activation = None
        self.fc1 = nn.Linear(2048 + 1536, 2048)
        self.fc = nn.Linear(2048 + 1536, num_classes)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x, y, frame_num):
        y = self.cnn(y)

        x = self.cnn3d.blocks[0](x)
        x = self.cnn3d.blocks[1](x)
        x = self.cnn3d.blocks[2](x)
        x = self.cnn3d.blocks[3](x)
        x = self.cnn3d.blocks[4](x)
        x = self.cnn3d.blocks[5].pool(x)
        x = self.cnn3d.blocks[5].dropout(x)
        x = self.pool(x)
        x = torch.squeeze(x)

        # print(x.shape, y.shape)
        c = torch.cat([x, y], dim=1)
        # print(c.shape)
        # c = self.fc1(c)
        # c = self.relu(c)

        c = c + frame_num
        c = self.fc(c)

        return [c]


if __name__ == "__main__":
    x3d = CNN_3D(num_classes=21)
    print(x3d)
    # timm = CNN_timm(num_classes=21)

    input = torch.randn(2, 3, 16, 224, 224)
    input = input.to("cuda")
    model = x3d.to("cuda")

    output = model(input)
