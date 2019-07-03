import os
import numpy as np
import torch
import torch.nn as nn
import torch.cuda as cuda
from utils.data_loader import dataloader_cifar
from utils.data_loader import dataloader_imagenet
from utils.lsq_train import LogHelper
from utils.lsq_train import get_arguments
from utils.lsq_train import get_optimizer
from utils.lsq_module import add_lsqmodule
from utils.lsq_network import resnet18
from utils.lsq_network import resnet20
from utils.lsq_network import resnet50
from utils.utilities import Trainer
from utils.utilities import get_constraint


def main():
    os.chdir(os.path.dirname(__file__))
    args = get_arguments()
    constr_weight = get_constraint(args.weight_bits, 'weight')
    constr_activation = get_constraint(args.activation_bits, 'activation')
    if args.dataset == 'cifar10':
        network = resnet20
        dataloader = dataloader_cifar
    else:
        if args.network == 'resnet18':
            network = resnet18
        elif args.network == 'resnet50':
            network = resnet50
        else:
            print('Not Support Network Type: %s' % args.network)
            return
        dataloader = dataloader_imagenet
    train_loader = dataloader(args.data_root, split='train', batch_size=args.batch_size)
    test_loader = dataloader(args.data_root, split='test', batch_size=args.batch_size)
    net = network(quan_first_last=args.quan_first_last,
                  constr_activation=constr_activation,
                  preactivation=args.preactivation)

    model_path = os.path.join(args.model_root, args.model_name + '.pth')
    name_weights_old = torch.load(model_path)
    name_weights_new = net.state_dict()
    name_weights_new.update(name_weights_old)
    net.load_state_dict(name_weights_new)
    add_lsqmodule(net, constr_weight)
    print(net)
    net = net.cuda()
    net = nn.DataParallel(net, device_ids=range(cuda.device_count()))

    quan_activation = isinstance(constr_activation, np.ndarray)
    postfix = '_w' if not quan_activation else '_a'
    new_model_name = args.prefix + args.model_name + '_lsq' + postfix
    cache_root = os.path.join('.', 'cache')
    train_loger = LogHelper(new_model_name, cache_root, quan_activation, args.resume)
    optimizer, lr_scheduler = get_optimizer(net=net,
                                            optimizer=args.optimizer,
                                            lr_base=args.learning_rate,
                                            weight_decay=args.weight_decay,
                                            lr_scheduler=args.lr_scheduler,
                                            total_epoch=args.total_epoch,
                                            quan_activation=quan_activation)
    trainer = Trainer(net=net,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      optimizer=optimizer,
                      lr_scheduler=lr_scheduler,
                      model_name=new_model_name,
                      train_loger=train_loger)
    trainer(total_epoch=args.total_epoch,
            save_check_point=True,
            resume=args.resume)


if __name__ == '__main__':
    main()




