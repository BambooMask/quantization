import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


class TernaryFun(autograd.Function):
    def __init__(self):
        super(TernaryFun, self).__init__()
        self.alpha = None

    def forward(self, *args, **kwargs):
        weight = args[0]
        weight_abs = weight.abs()
        delta = 0.75 * weight_abs.mean()
        weight_nozero = weight * (weight_abs > delta).float()
        alpha = weight_nozero.abs().mean()
        weight_quant = alpha * weight_nozero.sign()
        self.alpha = alpha.item()
        return weight_quant

    def backward(self, *grad_outputs):
        grad_top = grad_outputs[0]
        return grad_top


class Ternary(nn.Module):
    def __init__(self):
        super(Ternary, self).__init__()
        self.alpha = None

    def forward(self, x):
        tfun = TernaryFun()
        x_quant = tfun(x)
        self.alpha = tfun.alpha
        return x_quant


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.wquantizer = None

    def forward(self, x):
        weight = self.weight if self.wquantizer is None else self.wquantizer(self.weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias):
        super(Linear, self).__init__(in_features, out_features, bias)
        self.wquantizer = None

    def forward(self, x):
        weight = self.weight if self.wquantizer is None else self.wquantizer(self.weight)
        return F.linear(x, weight, self.bias)


def add_twnmodule(net):
    for module in net.modules():
        if isinstance(module, Conv2d) or isinstance(module, Linear):
            module.wquantizer = Ternary()

