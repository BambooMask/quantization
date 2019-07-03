import os
import torch
import torch.nn as nn
import torch.cuda as cuda
from functools import partial
from utils.data_loader import dataloader_cifar
from utils.twn_module import add_twnmodule
from utils.twn_train import log_fun
from utils.twn_train import clear_logfile
from utils.twn_train import get_arguments
from utils.twn_train import get_optimizer
from utils.twn_network import Resnet20Cifar
from utils.twn_network import Resnet18Imagenet
from utils.utilities import Trainer


def main():
    args = get_arguments()
    os.chdir(os.path.dirname(__file__))
    if args.dataset == 'cifar10':
        if args.data_root is None:
            data_root = os.path.join('F:', '0data', 'cifar')
        else:
            data_root = args.data_root
        if args.model_root is None:
            model_root = os.path.join('F:', '1pretrain_model')
        else:
            model_root = args.model_root
        train_loader = dataloader_cifar(data_root, split='train', batch_size=128)
        test_loader = dataloader_cifar(data_root, split='test', batch_size=128)
        net = Resnet20Cifar(quan_first_last=args.quan_first_last,
                            quan_shortcut=args.quan_shortcut,
                            preactivation=args.preactivation)
    else:
        if args.data_root is None:
            data_root = os.path.join('/', 'share', 'buyingjia', 'common_dataset', 'ImageNetOrigin')
        else:
            data_root = args.data_root
        if args.model_root is None:
            model_root = os.path.join('/', 'data', '1pretrain_model')
        else:
            model_root = args.model_root
        train_loader = dataloader_cifar(data_root, split='train', batch_size=256)
        test_loader = dataloader_cifar(data_root, split='test', batch_size=128)
        net = Resnet18Imagenet(quan_first_last=args.quan_first_last,
                               quan_shortcut=args.quan_shortcut,
                               preactivation=args.preactivation)

    model_name = args.model_name
    model_path = os.path.join(model_root, model_name + '.pth')
    name_weights_old = torch.load(model_path)
    name_weights_new = net.state_dict()
    name_weights_new.update(name_weights_old)
    net.load_state_dict(name_weights_new)
    add_twnmodule(net)
    net = net.cuda()
    net = nn.DataParallel(net, device_ids=range(cuda.device_count()))

    new_model_name = args.prefix + model_name + '_twn'
    cache_root = os.path.join('.', 'cache')
    log_cache = os.path.join(cache_root, 'log_cache')
    clear_logfile(log_cache=log_cache, model_name=model_name, resume=args.resume)
    trainlog_fun = partial(log_fun, log_cache=log_cache, model_name=model_name)
    optimizer, lr_scheduler = get_optimizer(net=net,
                                            optimizer=args.optimizer,
                                            lr_base=args.learning_rate,
                                            total_epoch=args.total_epoch)
    trainer = Trainer(net=net,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      optimizer=optimizer,
                      lr_scheduler=lr_scheduler,
                      model_name=new_model_name,
                      trainlog_fun=trainlog_fun)
    trainer(total_epoch=args.total_epoch,
            cache_root=cache_root,
            save_check_point=True,
            resume=args.resume)


if __name__ == '__main__':
    main()


