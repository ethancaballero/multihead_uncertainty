import os, sys, time
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib
import torch.nn.functional as F
matplotlib.use('agg')
import matplotlib.pyplot as plt


def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]

class Edl_log_loss(torch.nn.Module):
    def __call__(self, output, target, epoch_num, num_classes, annealing_step, device=None):
        if not device:
            device = get_device()
        evidence = relu_evidence(output)
        alpha = evidence + 1
        loss = torch.mean(edl_loss(torch.log, target, alpha,
                                   epoch_num, num_classes, annealing_step, device))
        return loss

def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = get_device()
    beta = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - \
        torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1,
                        keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1,
                   keepdim=True) + lnB + lnB_uni
    return kl

def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(torch.tensor(
        1.0, dtype=torch.float32), torch.tensor(epoch_num / annealing_step, dtype=torch.float32))

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * \
        kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


class Edl_digamma_loss(torch.nn.Module):
    def __call__(self, output, target, epoch_num, num_classes, annealing_step, device=None):
        if not device:
            device = get_device()
        evidence = relu_evidence(output)
        alpha = evidence + 1
        loss = torch.mean(edl_loss(torch.digamma, target, alpha,
                                   epoch_num, num_classes, annealing_step, device))
        return loss


def loglikelihood_loss(y, alpha, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum(
        (y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood

def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = torch.min(torch.tensor(
        1.0, dtype=torch.float32), torch.tensor(epoch_num / annealing_step, dtype=torch.float32))

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * \
        kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div

def relu_evidence(y):
    return F.relu(y)

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device
class Edl_mse_loss(torch.nn.Module):
    def __call__(self, output, target, epoch_num, num_classes, annealing_step, device=None):
        if not device:
            device = get_device()
        evidence = relu_evidence(output)
        alpha = evidence + 1
        loss = torch.mean(mse_loss(target, alpha, epoch_num,
                                   num_classes, annealing_step, device=device))
        return loss

def to_one_hot(inp, num_classes):
    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    y_onehot.zero_()

    y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)

    return Variable(y_onehot.cuda(), requires_grad=False)


def mixup_process(out, target_reweighted, lam):
    indices = np.random.permutation(out.size(0))
    out = out * lam + out[indices] * (1 - lam)
    target_shuffled_onehot = target_reweighted[indices]
    target_reweighted = target_reweighted * lam + target_shuffled_onehot * (1 - lam)

    # t1 = target.data.cpu().numpy()
    # t2 = target[indices].data.cpu().numpy()
    # print (np.sum(t1==t2))
    return out, target_reweighted


def mixup_data(x, y, alpha):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def get_lambda(alpha=1.0):
    '''Return lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam
