import torch
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from scipy.linalg import qr


def qr_null(weight, num=None):
    nrow, ncol = weight.shape
    if num is None:
        num = ncol - nrow
    Q, R = qr(weight.T, mode='full')
    X = np.eye(ncol)[:, nrow:nrow + num]
    X = np.matmul(Q, X)
    return X.T


@autograd.no_grad()
def repr_prune(name_weight, prune_ratio):
    print('Pruning...')
    if prune_ratio > 1:
        prune_ratio /= 100
    name_criterion = dict()
    all_criterion = list()
    for name in name_weight.keys():
        weight = name_weight[name]
        if weight.dim() < 4:
            continue

        weight_flatted = weight.flatten(start_dim=1)
        weight_normed = F.normalize(weight_flatted, p=2, dim=1)
        orthogonal_mat = torch.matmul(weight_normed, weight_normed.t())
        identity_mat = torch.eye(weight_normed.shape[0], device=weight_normed.device)
        critertion = (orthogonal_mat - identity_mat).abs().mean(dim=1)

        norm_of_weight = torch.norm(weight_flatted, p=1, dim=1) / weight_flatted.shape[1]
        small_norm_index = (norm_of_weight < 1e-8).nonzero().flatten()
        critertion[small_norm_index] = 1

        name_criterion[name] = critertion
        all_criterion.append(critertion)
    all_criterion = torch.cat(tuple(all_criterion)).cpu().numpy()
    threshold = np.percentile(all_criterion, 100 * (1 - prune_ratio))

    drop_mask = dict()
    drop_filters = dict()
    for name in name_weight.keys():
        weight = name_weight[name]
        if weight.dim() < 4:
            continue
        drop_mask[name] = (name_criterion[name] > threshold).nonzero().flatten()
        if drop_mask[name].numel() != 0:
            weight_flatted = weight.flatten(start_dim=1)
            drop_filters[name] = torch.index_select(weight_flatted, dim=0, index=drop_mask[name])
            weight[drop_mask[name]] = 0
        else:
            drop_filters[name] = None
    return name_weight, drop_mask, drop_filters


@autograd.no_grad()
def repr_reinit(name_weight, drop_mask, drop_filters):
    print('Reinitializing...')
    for name in name_weight:
        weight = name_weight[name]
        if weight.dim() < 4 or drop_filters[name] is None:
            continue
        size = weight.size()
        weight_flatted = weight.flatten(start_dim=1).cpu().numpy()
        weight_droped = drop_filters[name].cpu().numpy()
        index = drop_mask[name].cpu().numpy()
        weight_flatted[index] = weight_droped

        stdv = 1 / np.sqrt(size[1] * size[2] * size[3])
        null_space = qr_null(weight_flatted, num=weight_droped.shape[0])
        if null_space.shape[0] == 0:
            weight[drop_mask[name]].uniform_(-stdv, stdv)
        else:
            null_space = torch.tensor(null_space, dtype=torch.float, device=weight.device)
            null_space = F.normalize(null_space, p=2, dim=1)
            null_space = torch.clamp(null_space, min=-stdv, max=stdv)
            null_space  = torch.reshape(null_space, (-1, size[1], size[2], size[3]))
            len_null_space = null_space.shape[0]
            if len_null_space < drop_mask[name].shape[0]:
                pre_index = drop_mask[name][0:len_null_space]
                post_index = drop_mask[name][len_null_space:]
                weight[pre_index] = null_space
                weight[post_index].uniform_(-stdv, stdv)
            else:
                weight[drop_mask[name]] = null_space
    return name_weight


def show_drop_info(drop_mask):
    for name in drop_mask.keys():
        drop_num = drop_mask[name].shape[0] if drop_mask[name] is not None else 0
        print(name, drop_num)




















'''
@autograd.no_grad()
def prune_conv(module, last_keep_index, remove_filter_num):
    weight = module.weight.detach().clone().flatten(start_dim=1)
    assert remove_filter_num > 0, 'remove_filter_num must be greater than 0'
    if remove_filter_num < 1:
        remove_filter_num = int(weight.shape[0] * remove_filter_num)
    keep_filter_num = weight.shape[0] - remove_filter_num
    norm_of_weight = weight.norm(dim=1)
    _, curr_keep_index = torch.topk(norm_of_weight, keep_filter_num, largest=True)
    weight = module.weight.detach().clone()
    curr_keep_index, _ = curr_keep_index.sort()
    weight = torch.index_select(weight, dim=0, index=curr_keep_index)
    if last_keep_index is not None:
        weight = torch.index_select(weight, dim=1, index=last_keep_index)
    module.weight = nn.Parameter(weight)
    if module.bias is not None:
        bias = module.bias.detach().clone()
        bias = torch.index_select(bias, dim=0, index=curr_keep_index)
        module.bias = nn.Parameter(bias)
    module.in_channels = weight.shape[1]
    module.out_channels = weight.shape[0]
    return module, curr_keep_index


@autograd.no_grad()
def prune_bn(module, last_keep_index):
    weight = module.weight.detach().clone()
    weight = torch.index_select(weight, dim=0, index=last_keep_index)
    module.weight = nn.Parameter(weight)
    bias = module.bias.detach().clone()
    bias = torch.index_select(bias, dim=0, index=last_keep_index)
    module.bias = nn.Parameter(bias)
    running_mean = module.running_mean.clone()
    running_mean = torch.index_select(running_mean, dim=0, index=last_keep_index)
    module.running_mean = running_mean
    running_var = module.running_var.clone()
    running_var = torch.index_select(running_var, dim=0, index=last_keep_index)
    module.running_var = running_var
    module.num_features = weight.shape[0]
    return module, last_keep_index


@autograd.no_grad()
def prune_linear(module, last_keep_index):
    weight = module.weight.detach().clone()
    weight = torch.index_select(weight, dim=1, index=last_keep_index)
    module.weight = nn.Parameter(weight)
    module.in_features = weight.shape[1]
    module.out_features = weight.shape[0]
    last_keep_index = torch.arange(module.out_features)
    return module, last_keep_index


@autograd.no_grad()
def prune_basicblock(module, last_keep_index, remove_filter_num):
    origin_keep_index = last_keep_index
    module.conv1, last_keep_index = prune_conv(module.conv1, last_keep_index, remove_filter_num)
    module.bn1, last_keep_index = prune_bn(module.bn1, last_keep_index)
    module.conv2, last_keep_index = prune_conv(module.conv2, last_keep_index, remove_filter_num)
    module.bn2, last_keep_index = prune_bn(module.bn2, last_keep_index)
    if module.downsample is not None:
        module.downsample[0], origin_keep_index = prune_conv(module.downsample[0], origin_keep_index, remove_filter_num)
        module.downsample[1], _ = prune_bn(module.downsample[1], origin_keep_index)
    return module, last_keep_index


def no_need_prune_layer(module):
    flag = False
    if isinstance(module, nn.ReLU):
        flag = True
    elif isinstance(module, nn.MaxPool2d):
        flag = True
    elif isinstance(module, nn.AvgPool2d):
        flag = True
    elif isinstance(module, nn.AdaptiveMaxPool2d):
        flag = True
    elif isinstance(module, nn.AdaptiveAvgPool2d):
        flag = True
    return flag


@autograd.no_grad()
def prune_compose_module(module, last_keep_index, remove_filter_num):
    for submodule in module.children():
        if no_need_prune_layer(submodule):
            continue
        elif isinstance(submodule, BasicBlock):
            submodule, last_keep_index = prune_basicblock(submodule, last_keep_index, remove_filter_num)
        elif isinstance(submodule, nn.Conv2d):
            submodule, last_keep_index = prune_conv(submodule, last_keep_index, remove_filter_num)
        elif isinstance(submodule, nn.BatchNorm2d):
            submodule, last_keep_index = prune_bn(submodule, last_keep_index)
        elif isinstance(submodule, nn.Linear):
            submodule, last_keep_index = prune_linear(submodule, last_keep_index)
        else:
            submodule, last_keep_index = prune_compose_module(submodule, last_keep_index, remove_filter_num)
    return module, last_keep_index
'''

