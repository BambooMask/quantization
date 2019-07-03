
import torch
import torch.nn as nn
from .twn_module import Conv2d
from .twn_module import Linear


class _Identity(nn.Module):
    def forward(self, x):
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False, quan_shortcut=False):
        super(BasicBlock, self).__init__()
        self.quan_shortcut = quan_shortcut

        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.activation1 = nn.ReLU(inplace=True)

        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.activation2 = nn.ReLU(inplace=True)
        if downsample:
            if quan_shortcut:
                residual_conv = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
                residual_bn = nn.BatchNorm2d(num_features=out_channels)
            else:
                residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
                residual_bn = nn.BatchNorm2d(num_features=out_channels)
            layers = [residual_conv, residual_bn]
            self.downsample = nn.Sequential(*layers)
        else:
            self.downsample = None

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
    def __init__(self, in_channels, out_channels, stride=1, downsample=False, quan_shortcut=False):
        super(PreActivationBlock, self).__init__()
        self.quan_shortcut = quan_shortcut

        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.activation1 = nn.ReLU(inplace=True)
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.activation2 = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        if downsample:
            if self.quan_shortcut:
                residual_bn = nn.BatchNorm2d(num_features=in_channels)
                residual_activation = _Identity()
                residual_conv = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
            else:
                residual_bn = nn.BatchNorm2d(num_features=in_channels)
                residual_activation = _Identity()
                residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
            layers = [residual_bn, residual_activation, residual_conv]
            self.downsample = nn.Sequential(*layers)
        else:
            self.downsample = None

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


class Resnet20Cifar(nn.Module):
    def __init__(self, quan_first_last=False, quan_shortcut=False, preactivation=False):
        super(Resnet20Cifar, self).__init__()
        self.quan_fist_last = quan_first_last
        self.quan_shortcut = quan_shortcut
        self.preactivation = preactivation
        if self.preactivation:
            block = PreActivationBlock
        else:
            block = BasicBlock

        if quan_first_last:
            self.conv1 = Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.activation1 = nn.ReLU(inplace=True)
        self.block1 = block(in_channels=16, out_channels=16, quan_shortcut=quan_shortcut)
        self.block2 = block(in_channels=16, out_channels=16, quan_shortcut=quan_shortcut)
        self.block3 = block(in_channels=16, out_channels=16, quan_shortcut=quan_shortcut)
        self.block4 = block(in_channels=16, out_channels=32, stride=2, downsample=True, quan_shortcut=quan_shortcut)
        self.block5 = block(in_channels=32, out_channels=32, quan_shortcut=quan_shortcut)
        self.block6 = block(in_channels=32, out_channels=32, quan_shortcut=quan_shortcut)
        self.block7 = block(in_channels=32, out_channels=64, stride=2, downsample=True, quan_shortcut=quan_shortcut)
        self.block8 = block(in_channels=64, out_channels=64, quan_shortcut=quan_shortcut)
        self.block9 = block(in_channels=64, out_channels=64, quan_shortcut=quan_shortcut)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        if quan_first_last:
            self.fc = Linear(in_features=64, out_features=10, bias=True)
        else:
            self.fc = nn.Linear(in_features=64, out_features=10, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block8(out)
        out = self.block9(out)
        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        return out


class Resnet18Imagenet(nn.Module):
    def __init__(self, quan_first_last=False, quan_shortcut=True, preactivation=False):
        super(Resnet18Imagenet, self).__init__()
        self.quan_first_last = quan_first_last
        self.quan_shortcut = quan_shortcut
        self.preactivation = preactivation
        if self.preactivation:
            block = PreActivationBlock
        else:
            block = BasicBlock

        if quan_first_last:
            self.conv1 = Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.activation1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 64, stride=1, quan_shortcut=quan_shortcut)
        self.layer2 = self._make_layer(block, 64, 128, stride=2, quan_shortcut=quan_shortcut)
        self.layer3 = self._make_layer(block, 128, 256, stride=2, quan_shortcut=quan_shortcut)
        self.layer4 = self._make_layer(block, 256, 512, stride=2, quan_shortcut=quan_shortcut)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        if quan_first_last:
            self.fc = Linear(512, 1000, bias=True)
        else:
            self.fc = nn.Linear(512, 1000, bias=True)

    @staticmethod
    def _make_layer(block, in_channels, out_channels, stride=1, quan_shortcut=True):
        downsample = False
        if stride != 1 or in_channels != out_channels:
            downsample = True
        layers = list()
        layers.append(block(in_channels, out_channels, stride, downsample, quan_shortcut=quan_shortcut))
        layers.append(block(out_channels, out_channels, quan_shortcut=quan_shortcut))
        return nn.Sequential(*layers)

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

