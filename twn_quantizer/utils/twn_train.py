import os
import argparse
import torch.optim as optim
from .twn_module import Ternary

def get_arguments():
    argparser = argparse.ArgumentParser(description='TWN Training')
    argparser.add_argument('--model_name', required=True, type=str)
    argparser.add_argument('--dataset', required=True, choices=['cifar10', 'imagenet'], type=str)
    argparser.add_argument('--network', required=True, choices=['resnet20', 'resnet18'], type=str)
    argparser.add_argument('--data_root', default=None, type=str)
    argparser.add_argument('--model_root', default=None, type=str)

    argparser.add_argument('--preactivation', default=False, action='store_true')
    argparser.add_argument('--quan_first_last', default=False, action='store_true')
    argparser.add_argument('--quan_shortcut', default=False, action='store_true')

    argparser.add_argument('--total_epoch', default=90, type=int)
    argparser.add_argument('--optimizer', default='sgd', type=str)
    argparser.add_argument('--learning_rate', default=0.01, type=float)

    argparser.add_argument('--resume', default=False, action='store_true')
    argparser.add_argument('--prefix', default='', type=str)

    args = argparser.parse_args()
    print(' ... Experiment Setting ...')
    print('model_name=%s' % args.model_name)
    print('dataset=%s' % args.dataset)
    print('network=%s' % args.network)
    print('--------------------')

    print('quan_first_last=%s' % args.quan_first_last)
    print('quan_shortcut=%s' % args.quan_shortcut)
    print('--------------------')

    print('total_epoch=%s' % args.total_epoch)
    print('optimizer=%s' % args.optimizer)
    print('learning_rate=%s' % args.learning_rate)
    print('--------------------')

    print('resume=%s' % args.resume)
    print('prefix=%s' % (args.prefix if args.prefix else 'None'))
    return args


def log_fun(index, net, msg, log_cache, model_name):
    log_path = os.path.join(log_cache, model_name + '_trainlog.txt')
    scale_path = os.path.join(log_cache, model_name + '_scale_weight.txt')

    with open(log_path, 'a') as fd:
        fd.write(msg + '\n')

    if scale_path is not None and index % 100 == 0:
        info = ''
        for module in net.modules():
            if isinstance(module, Ternary):
                info += '%.10f, ' % module.alpha
        info = info[:-2] + '\n'
        with open(scale_path, 'a') as fd:
            fd.write(info)


def clear_logfile(log_cache, model_name, resume=False):
    if resume:
        return None
    log_path = os.path.join(log_cache, model_name + '_trainlog.txt')
    if os.path.exists(log_path):
        os.remove(log_path)
    scale_path = os.path.join(log_cache, model_name + '_scale_weight.txt')
    if os.path.exists(scale_path):
        os.remove(scale_path)


def get_optimizer(net, optimizer, lr_base, total_epoch):
    params = net.parameters()
    if 'adam' == optimizer:
        optimizer = optim.Adam(params, lr=lr_base, weight_decay=5e-5)
    else:
        optimizer = optim.SGD(params, lr=lr_base, momentum=0.9, weight_decay=5e-5)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch, eta_min=1e-9)
    return optimizer, lr_scheduler




