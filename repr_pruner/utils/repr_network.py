import math
import torch
import torch.nn as nn


def _make_layer(block, in_channels, planes, blocks, stride=1):
    layers = list()
    downsample = stride != 1 or in_channels != planes * block.expansion
    layers.append(block(in_channels, planes, stride, downsample))
    in_channels = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(in_channels, planes))
    return nn.Sequential(*layers), planes * block.expansion


class _Identity(nn.Module):
    def forward(self, x):
        return x


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, planes, stride=1, downsample=False):
        super(BasicBlock, self).__init__()
        out_channels = planes * self.expansion

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.activation1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.activation2 = nn.ReLU(inplace=True)

        self.downsample = None
        if downsample:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
            bn = nn.BatchNorm2d(num_features=out_channels)
            self.downsample = nn.Sequential(*[conv, bn])

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.activation2(out)
        return out


class PreActivationBlock(nn.Module):
    expandsion = 1
    def __init__(self, in_channels, planes, stride=1, downsample=False):
        super(PreActivationBlock, self).__init__()
        out_channels = planes * self.expandsion

        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.activation1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.activation2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.downsample = None
        if downsample:
            bn = nn.BatchNorm2d(num_features=in_channels)
            activation = _Identity()
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
            self.downsample = nn.Sequential(*[bn, activation, conv])


    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.bn1(x)
        out = self.activation1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.activation2(out)
        out = self.conv2(out)
        out += residual
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.activation1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.activation2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.activation3 = nn.ReLU(inplace=True)

        self.downsample = None
        if downsample:
            conv = nn.Conv2d(in_channels, planes * 4, kernel_size=1, stride=stride, padding=0, bias=False)
            bn = nn.BatchNorm2d(num_features=planes * 4)
            self.downsample = nn.Sequential(*[conv, bn])

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.activation3(out)
        return out


class Resnet20(nn.Module):
    def __init__(self, preactivation=False):
        super(Resnet20, self).__init__()
        block = PreActivationBlock if preactivation else BasicBlock

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.activation1 = nn.ReLU(inplace=True)
        in_channels = 16
        self.layer1, in_channels = _make_layer(block, in_channels, planes=16, blocks=3, stride=1)
        self.layer2, in_channels = _make_layer(block, in_channels, planes=32, blocks=3, stride=2)
        self.layer3, in_channels = _make_layer(block, in_channels, planes=64, blocks=3, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(in_features=64, out_features=10, bias=True)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.activation1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        in_channels = 64
        self.layer1, in_channels = _make_layer(block, in_channels, 64, layers[0], stride=1)
        self.layer2, in_channels = _make_layer(block, in_channels, 128, layers[1], stride=2)
        self.layer3, in_channels = _make_layer(block, in_channels, 256, layers[2], stride=2)
        self.layer4, in_channels = _make_layer(block, in_channels, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



class Vanilla(nn.Module):
    def __init__(self, num_classes=10):
        super(Vanilla, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32*32*32, num_classes)

    def forward(self, x):
        relu = nn.ReLU()

        out = self.conv1(x)
        out = relu(out)

        out = self.conv2(out)
        out = relu(out)

        out = self.conv3(out)
        out = relu(out)

        out = torch.flatten(out, start_dim=1)
        out = self.fc1(out)
        return out


def vanilla():
    return Vanilla()


def resnet18(preactivation=False):
    block = PreActivationBlock if preactivation else BasicBlock
    model = ResNet(block, [2, 2, 2, 2])
    return model


def resnet20(preactivation=False):
    return Resnet20(preactivation)

























