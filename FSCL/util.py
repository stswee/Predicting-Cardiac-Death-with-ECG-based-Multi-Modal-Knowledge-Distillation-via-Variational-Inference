from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim

#https://github.com/comp-well-org/ESI/blob/main/utils/augmentation.py
class ECGAugment:
    def __init__(self, jitter_sigma=0.03, scaling_sigma=0.1, channel_first=False):
        self.sigma = sigma

    def jitter(self, x):
        # Jitter is added to every point in the time series data, so no change is needed based on channel_first
        return x + np.random.normal(loc=0., scale=self.jitter_sigma, size=x.shape)

    def scaling(self, x, sigma):
        if self.channel_first:
            # For (B, C, L), generate a factor for each channel and broadcast across L
            factor = np.random.normal(loc=1., scale=scaling_sigma, size=(x.shape[0], x.shape[1], 1))
        else:
            # For (B, L, C), generate a factor for each channel and broadcast across L
            factor = np.random.normal(loc=1., scale=scaling_sigma, size=(x.shape[0], 1, x.shape[2]))
        
        return np.multiply(x, factor)
    
    def __call__(self, x):
        x = self.jitter(x)  # Apply Jitter
        x = self.scaling(x) # Apply Scaling
        return x


class Jitter(object):
    def __init__(self, sigma=0.03):
        self.sigma = sigma

    def jitter(self, x, sigma):
        # Jitter is added to every point in the time series data, so no change is needed based on channel_first
        return x + np.random.normal(loc=0., scale=sigma, size=x.shape)
    
    def __call__(self, x):
        return self.jitter(x, self.sigma)


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, ta, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = ta.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(ta.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def val_accuracy(output, ta, sa, ta_cls, sa_cls, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = ta.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(ta.view(1, -1).expand_as(pred))

        group=[]
        group_num=[]
        for i in range(ta_cls):
            sa_group=[]
            sa_group_num=[]
            for j in range(sa_cls):
                eps=1e-8
                sa_group.append(((sa==j)*(ta==i)*(correct==1)).float().sum() *(100 /(((sa==j)*(ta==i)).float().sum()+eps)))
                sa_group_num.append(((sa==j)*(ta==i)).float().sum()+eps)
            group.append(sa_group)
            group_num.append(sa_group_num)
       
        res=(correct==1).float().sum()*(100.0 / batch_size)
        
        return res,group,group_num


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state
