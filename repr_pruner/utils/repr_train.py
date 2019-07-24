import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from datetime import datetime
from collections import OrderedDict
from .utilities import train_one_epoch
from .utilities import eval_performance


def get_optimizer(net, optimizer, lr_base, lr_scheduler, total_epoch):
    param_groups = net.parameters()
    if optimizer == 'adam':
        optimizer = optim.Adam(param_groups, lr=lr_base, weight_decay=5e-5)
    else:
        optimizer = optim.SGD(param_groups, lr=lr_base, momentum=0.9, weight_decay=5e-5)
    if lr_scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch, eta_min=1e-9)
    else:
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    return optimizer, lr_scheduler


class ReprTrain:
    def __init__(self, net, train_loader, test_loader, optimizer, lr_scheduler, train_loger, model_name, reprune=None):
        self.net = net
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loger = train_loger
        self.model_name = model_name
        self.reprune = reprune
        self.criterion = nn.CrossEntropyLoss()

    def save_model(self):
        print('\n ... Save Model ...')
        model_path = os.path.join(self.train_loger.model_cache, self.model_name + '.pth')
        try:
            state_dict = self.net.module.state_dict()
        except AttributeError:
            state_dict = self.net.state_dict()
        torch.save(state_dict, model_path)

    def load_check_point(self):
        ckpt_cache = self.train_loger.ckpt_cache
        net_path = os.path.join(ckpt_cache, self.model_name + '_ckpt_net.pth')
        opt_path = os.path.join(ckpt_cache, self.model_name + '_ckpt_opt.pth')
        lrs_path = os.path.join(ckpt_cache, self.model_name + '_ckpt_lrs.pth')
        self.net.load_state_dict(torch.load(net_path))
        self.optimizer.load_state_dict(torch.load(opt_path))
        self.lr_scheduler.load_state_dict(torch.load(lrs_path))

        epoch_acc_path = os.path.join(ckpt_cache, self.model_name + '_ckpt_epoch_acc.pkl')
        with open(epoch_acc_path, 'rb') as fd:
            check_point = pickle.load(fd)
            start_epoch = check_point['start_epoch']
            best_quan_acc = check_point['best_quan_acc']
        return start_epoch, best_quan_acc

    def save_check_point(self, epoch, best_quan_acc):
        ckpt_cache = self.train_loger.ckpt_cache
        net_path = os.path.join(ckpt_cache, self.model_name + '_ckpt_net.pth')
        opt_path = os.path.join(ckpt_cache, self.model_name + '_ckpt_opt.pth')
        lrs_path = os.path.join(ckpt_cache, self.model_name + '_ckpt_lrs.pth')
        torch.save(self.net.state_dict(), net_path)
        torch.save(self.optimizer.state_dict(), opt_path)
        torch.save(self.lr_scheduler.state_dict(), lrs_path)

        epoch_acc_path = os.path.join(ckpt_cache, self.model_name + '_ckpt_epoch_acc.pkl')
        with open(epoch_acc_path, 'wb') as fd:
            check_point = {'start_epoch': epoch + 1, 'best_quan_acc': best_quan_acc}
            pickle.dump(check_point, fd)

    def __call__(self, total_epoch, save_check_point=True, resume=False):
        best_test_acc = 0
        start_epoch = 0
        if resume:
            start_epoch, best_test_acc = self.load_check_point()
            print('\n... Resume: start_epoch=%s, best_test_acc=%s' % (start_epoch, best_test_acc))

        for epoch in range(start_epoch, total_epoch):
            print('%s | Current %d | Total %d' % (str(datetime.now()), epoch, total_epoch))
            self.reprune(epoch)
            train_perf = train_one_epoch(net=self.net,
                                         train_loader=self.train_loader,
                                         criterion=self.criterion,
                                         optimizer=self.optimizer,
                                         loger=self.train_loger.print_log,
                                         reprune=self.reprune)
            test_perf = eval_performance(net=self.net,
                                         test_loader=self.test_loader,
                                         criterion=self.criterion,
                                         loger=self.train_loger.print_log)

            self.train_loger.print_perf(train_perf + test_perf)
            if test_perf[1] > best_test_acc:
                best_test_acc = test_perf[1]
                self.save_model()

            if save_check_point:
                self.save_check_point(epoch + 1, best_test_acc)


class Reprune:
    def __init__(self, net, s1_epoch, s2_epoch, ratio):
        self.net = net
        self.s1_epoch = s1_epoch
        self.s2_epoch = s2_epoch
        self.ratio = ratio
        self.shrink_train = False
        self.drop_index = OrderedDict()

    def update_grad(self):
        if self.shrink_train:
            for name, module in self.net.named_modules():
                if name in self.drop_index.keys():
                    drop_index = self.drop_index[name]
                    module.weight.grad[drop_index] = 0

    @staticmethod
    def qr_null(weight, num):
        nfilters, filter_size = weight.shape
        if num is None:
            num = filter_size - nfilters
        Q, R = np.linalg.qr(weight.T, mode='complete')
        B = np.eye(filter_size)[:, nfilters:(nfilters + num)]
        X = np.matmul(Q, B)
        return X.T


    @autograd.no_grad()
    def get_grad_mask(self):
        for name, module in self.net.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue
            weight = module.weight.data
            shape = weight.shape
            weight = torch.reshape(weight, (shape[0], -1))
            weight_normed = F.normalize(weight)
            ortho_mat = torch.matmul(weight_normed, weight_normed.t())
            criterion = ortho_mat.abs().mean(dim=(1,))

            small_weight_index = torch.le(weight.abs().mean(dim=(1,)), 1e-8)
            criterion[small_weight_index] = 1

            criterion = criterion.detach().cpu().numpy()
            thredhold = np.percentile(criterion, 100 * (1 - self.ratio))
            drop_index = np.nonzero(criterion > thredhold)
            drop_index = drop_index[0].tolist()
            self.drop_index[name] = drop_index
            module.weight[drop_index] = 0

    @autograd.no_grad()
    def reinit_weight(self):
        for name, module in self.net.named_modules():
            if name not in self.drop_index.keys():
                continue

            device = module.weight.device
            shape = module.weight.shape
            nfilters = shape[0]
            filter_size = shape[1] * shape[2] * shape[3]
            stdv = 1 / np.sqrt(filter_size)
            if nfilters >= filter_size:
                drop_index = self.drop_index[name]
                init_weight = module.weight.data[drop_index].uniform_(-stdv, stdv)
                module.weight.data[drop_index] = init_weight
            else:
                drop_index = self.drop_index[name]
                weight = module.weight.detach().cpu().numpy()
                weight = np.reshape(weight, (shape[0], -1))
                null_space = self.qr_null(weight, len(drop_index))
                northo = null_space.shape[0]
                null_space = torch.tensor(null_space, dtype=torch.float, device=device)
                null_space = F.normalize(null_space, p=2, dim=1)
                null_space = torch.reshape(null_space, (northo, shape[1], shape[2], shape[3]))
                null_space = F.hardtanh(null_space, -stdv, stdv)

                if len(drop_index) > northo:
                    n = len(drop_index) - northo
                    uniform_init = torch.empty(n, shape[1], shape[2], shape[3], device=device)
                    uniform_init = uniform_init.uniform_(-stdv, stdv)
                    init_weight = torch.cat((null_space, uniform_init))
                else:
                    init_weight = null_space
                module.weight.data[drop_index] = init_weight

    def __call__(self, epoch):
        if epoch >= self.s1_epoch:
            if epoch % (self.s1_epoch + self.s2_epoch) < self.s1_epoch:
                if self.shrink_train:
                    self.shrink_train = False
                    self.reinit_weight()
            else:
                if not self.shrink_train:
                    self.shrink_train = True
                    self.get_grad_mask()




