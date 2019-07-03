import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from datetime import datetime


def get_constraint(bits, obj):
    if bits == 0:
        return None
    if 'activation' in obj:
        lower = 0
        upper = 2 ** bits
    else:
        lower = -2 ** (bits - 1) + 1
        upper = 2 ** (bits - 1)
    constraint = np.arange(lower, upper)
    return constraint


def train_one_epoch(net, train_loader, optimizer, criterion, log_fun=None):
    print('\n ... Training Model For One Epoch ...')
    net.train()
    msg_print = MessagePrinter(len(train_loader), freq=10)
    for index, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, targets = inputs.cuda(), targets.long().cuda()
        outputs = net(inputs)
        loss_classify = criterion(outputs, targets)
        loss_classify.backward()
        optimizer.step()
        # print msg and write log
        msg = msg_print(loss_classify, targets, outputs)
        if log_fun is not None:
            log_fun(index, net, msg)
    return msg_print.get_performance()


@autograd.no_grad()
def eval_performance(net, test_loader, criterion, log_fun=None):
    print('\n ... Evaluate Model Performance ...')
    net.eval()
    msg_print = MessagePrinter(len(test_loader), freq=10)
    for index, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cuda(), targets.long().cuda()
        outputs = net(inputs)
        loss_classify = criterion(outputs, targets)
        msg = msg_print(loss_classify, targets, outputs)
        if log_fun is not None:
            log_fun(index, net, msg)
    return msg_print.get_performance()


class MessagePrinter:
    def __init__(self, total_iter, freq=10):
        self.iter_total = total_iter
        self.iter_curr = 0
        self.loss_cls = 0
        self.num_total = 0
        self.num_top1_correct = 0
        self.num_top5_correct = 0
        self.freq = freq

    def get_performance(self):
        loss_cls = round(self.loss_cls, 4)
        top1_acc = round(100. * self.num_top1_correct / self.num_total, 4)
        top5_acc = round(100. * self.num_top5_correct / self.num_total, 4)
        return [loss_cls, top1_acc, top5_acc]

    def __call__(self, loss_cls, targets, outputs):
        loss_cls = loss_cls.item()
        self.loss_cls = (self.loss_cls * self.iter_curr + loss_cls) / (self.iter_curr + 1)
        self.iter_curr += 1

        top1_correct, top5_correct = self.top1_top5_correct(targets, outputs)
        self.num_total += targets.shape[0]
        self.num_top1_correct += top1_correct
        self.num_top5_correct += top5_correct

        time_stamp = str(datetime.now())[:19]
        msg = '{}, {}/{}, Cls={:.3f}, Top1={:.3f}, Top5={:.3f}'
        top1_acc = 100. * self.num_top1_correct / self.num_total
        top5_acc = 100. * self.num_top5_correct / self.num_total
        show_msg = msg.format(time_stamp,
                              self.iter_curr,
                              self.iter_total,
                              self.loss_cls,
                              top1_acc,
                              top5_acc)
        if self.iter_curr % self.freq == 0 or self.iter_curr == self.iter_total:
            print(show_msg)
        return show_msg

    @staticmethod
    @autograd.no_grad()
    def top1_top5_correct(target, output, topk=(1, 5)):
        maxk = max(topk)
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).sum().item()
            res.append(correct_k)
        return res


class Trainer:
    def __init__(self, net, train_loader, test_loader, optimizer, lr_scheduler, model_name, train_loger=None):
        self.net = net
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.model_name = model_name
        self.train_loger = train_loger
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
        start_epoch = 0
        best_quan_acc = 0
        if resume:
            start_epoch, best_quan_acc = self.load_check_point()
            print('\n Resume, start_epoch=%d, best_quan_acc=%.3f' % (start_epoch, best_quan_acc))

        for epoch in range(start_epoch, total_epoch):
            print("\n %s | Current: %d | Total: %d" % (datetime.now(), epoch + 1, total_epoch))
            train_perf_epoch = train_one_epoch(net=self.net,
                                               train_loader=self.train_loader,
                                               optimizer=self.optimizer,
                                               criterion=self.criterion,
                                               log_fun=self.train_loger.print_log)
            quan_perf_epoch = eval_performance(net=self.net,
                                               test_loader=self.test_loader,
                                               criterion=self.criterion,
                                               log_fun=self.train_loger.print_log)
            self.lr_scheduler.step()

            self.train_loger.print_perf(train_perf_epoch + quan_perf_epoch)
            if quan_perf_epoch[1] > best_quan_acc:
                best_quan_acc = quan_perf_epoch[1]
                self.save_model()

            if save_check_point:
                self.save_check_point(epoch, best_quan_acc)


