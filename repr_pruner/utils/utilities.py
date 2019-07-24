import os
import torch.autograd as autograd
from datetime import datetime


def train_one_epoch(net, train_loader, criterion, optimizer, loger=None, reprune=None):
    print('\n ... Training Model For One Epoch ...')
    net.train()
    msg_print = Message_printer(len(train_loader), freq=10)
    for index, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, targets = inputs.cuda(), targets.long().cuda()
        outputs = net(inputs)
        loss_classify = criterion(outputs, targets)
        loss_classify.backward()

        if reprune is not None:
            reprune.update_grad()

        optimizer.step()
        # print msg and write log
        msg = msg_print(loss_classify, targets, outputs)
        if loger is not None:
            loger(index, net, msg)
    return msg_print.get_performance()


@autograd.no_grad()
def eval_performance(net, test_loader, criterion, loger):
    print('\n ... Evaluate Model Performance ...')
    net.eval()
    msg_print = Message_printer(len(test_loader), freq=10)
    for index, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cuda(), targets.long().cuda()
        outputs = net(inputs)
        loss_classify = criterion(outputs, targets)
        # print msg and write log
        msg = msg_print(loss_classify, targets, outputs)
        if loger is not None:
            loger(index, net, msg)
    return msg_print.get_performance()


class Message_printer:
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

        num_total = targets.shape[0]
        top1_correct, top5_correct = self.top1_top5_correct(targets, outputs)
        self.num_total += num_total
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


class LogHelper:
    def __init__(self, model_name, cache_root=None, resume=False):
        self.model_name = model_name
        self.log_cache = ''
        self.ckpt_cache = ''
        self.model_cache = ''
        self.prepare_cache_root(cache_root)
        self.clear_cache_root(resume)

        self.trainlog_path = os.path.join(self.log_cache, model_name + '_trainlog.txt')
        self.perf_path = os.path.join(self.log_cache, model_name + '_perf.txt')

    def prepare_cache_root(self, cache_root=None):
        if cache_root is None or not os.path.exists(cache_root):
            cache_root = os.path.join('.', 'cache')
            if not os.path.exists('./cache'):
                os.mkdir('./cache')
        self.ckpt_cache = os.path.join(cache_root, 'ckpt_cache')
        if not os.path.exists(self.ckpt_cache):
            os.mkdir(self.ckpt_cache)
        self.model_cache = os.path.join(cache_root, 'model_cache')
        if not os.path.exists(self.model_cache):
            os.mkdir(self.model_cache)
        self.log_cache = os.path.join(cache_root, 'log_cache')
        if not os.path.exists(self.log_cache):
            os.mkdir(self.log_cache)

    def clear_cache_root(self, resume):
        if resume:
            return None
        def clear_root(root, model_name):
            for file in os.listdir(root):
                if model_name in file:
                    os.remove(os.path.join(root, file))
        clear_root(self.log_cache, self.model_name)
        clear_root(self.ckpt_cache, self.model_name)
        clear_root(self.model_cache, self.model_name)

    def print_log(self, index, net, msg):
        with open(self.trainlog_path, 'a') as fd:
            fd.write(msg + '\n')

    def print_perf(self, perf):
        info = ''
        for item in perf:
            info += '%.4f, ' % item
        info = info[:-2] + '\n'
        with open(self.perf_path, 'a') as fd:
            fd.write(info)



