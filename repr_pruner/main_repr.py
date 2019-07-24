import os
import argparse
import torch.nn as nn
import torch.cuda as cuda
from utils.data_loader import dataloader_cifar
from utils.data_loader import dataloader_imagenet
from utils.repr_network import vanilla
from utils.repr_network import resnet18
from utils.repr_network import resnet20
from utils.repr_train import ReprTrain
from utils.repr_train import Reprune
from utils.utilities import LogHelper
from utils.repr_train import get_optimizer


def get_args():
    parser = argparse.ArgumentParser(description='REPR Training')
    parser.add_argument('--model_name', required=True, type=str)
    parser.add_argument('--dataset', required=True, choices=['cifar10', 'imagenet'], type=str)
    parser.add_argument('--network', required=True, choices=['vanillia', 'resnet20', 'resnet18'], type=str)
    parser.add_argument('--data_root', default=None, type=str)
    parser.add_argument('--model_root', default=None, type=str)

    parser.add_argument('--repr_time', default=3, type=int)
    parser.add_argument('--s1_epoch', default=20, type=int)
    parser.add_argument('--s2_epoch', default=10, type=int)
    parser.add_argument('--prune_ratio', default=0.1, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--total_epoch', default=90, type=int)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--lr_scheduler', default='exp', choices=['exp', 'cosine'], type=str)

    parser.add_argument('--prefix', default='', type=str)
    parser.add_argument('--resume', default=False, action='store_true')

    args = parser.parse_args()
    if args.data_root is None:
        if args.dataset == 'cifar10':
            args.data_root = os.path.join('F:', '0data', 'cifar')
        else:
            args.data_root = os.path.join('/', 'share', 'buyingjia', 'common_dataset', 'ImageNetOrigin')

    if args.model_root is None:
        if args.dataset == 'cifar10':
            args.model_root = os.path.join('F:', '1pretrain_model')
        else:
            args.model_root = os.path.join('/', 'data', '1pretrain_model')

    print(' ... Experiment Setting ...')
    print('model_name=%s' % args.model_name)
    print('dataset=%s' % args.dataset)
    print('network=%s' % args.network)
    print('data_root=%s' % args.data_root)
    print('model_root=%s' % args.model_root)
    print('----------------')

    print('repr_time=%s' % args.repr_time)
    print('s1_epoch=%s' % args.s1_epoch)
    print('s2_epoch=%s' % args.s2_epoch)
    print('prune_ratio=%s' % args.prune_ratio)
    print('----------------')

    print('optimizer=%s' % args.optimizer)
    print('total_epoch=%s' % args.total_epoch)
    print('learning_rate=%s' % args.learning_rate)
    print('----------------')

    print('prefix=%s' % (args.prefix if len(args.prefix) else 'None'))
    print('resume=%s' % args.resume)
    return args


def main():
    os.chdir(os.path.dirname(__file__))
    args = get_args()
    if args.dataset == 'cifar10':
        if args.network == 'vanilla':
            network = vanilla
        else:
            network = resnet20
        dataloader = dataloader_cifar
    else:
        network = resnet18
        dataloader = dataloader_imagenet

    train_loader = dataloader(args.data_root, split='train', batch_size=128)
    test_loader = dataloader(args.data_root, split='test', batch_size=128)
    net = network()
    net = net.cuda()
    net = nn.DataParallel(net, device_ids=range(cuda.device_count()))

    model_name = args.prefix + args.model_name + '_repr'
    cache_root = os.path.join('.', 'cache')
    train_loger = LogHelper(model_name, cache_root, args.resume)
    optimizer, lr_scheduler = get_optimizer(net, args.optimizer, args.learning_rate,
                                            args.lr_scheduler, args.total_epoch)
    reprune = Reprune(net, args.s1_epoch, args.s2_epoch, args.prune_ratio)
    trainer = ReprTrain(net=net,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        train_loger=train_loger,
                        model_name=model_name,
                        reprune=reprune)
    trainer(total_epoch=args.total_epoch,
            save_check_point=True,
            resume=args.resume)


if __name__ == '__main__':
    main()

