
import copy
import torch
import wandb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
from utils.prior import GaussianPrior
from models.omni_conv_net import Model_conv, Model_conv_idr

class Regularized_model(nn.Module):
    def __init__(self, channel_size, output_size, device="cpu", lamda=100, reg_matrix="diag", shuffle=False, consolidate=True, dropout=False, log=False, heads_number=1, activation='softplus', model=None, **kwargs):
        super(Regularized_model, self).__init__()
        self.criteria = nn.CrossEntropyLoss()
        self.device = device
        self.lamda=lamda
        self.current_task = 0
        self.loging_step = 0
        self.consolidate=consolidate
        self.reg_matrix = reg_matrix
        self.heads_number = heads_number
        self.model=Model_conv(channel_size, output_size, dropout=dropout, heads_number=heads_number, activation=activation) if model is None else model
        self.log = log
        if self.consolidate==False:
            self.prior=[]
        else:
            self.prior=None
        self.shuffle = shuffle

    def forward(self,x, target=None, mixup=False, mixup_hidden=False, mixup_alpha=None, task=0, noise=True, **kwargs):
        return self.model.forward(x, target=target, mixup=mixup, mixup_hidden=mixup_hidden, mixup_alpha=mixup_alpha, task=task)

    def loss_fim_mc_estimate(self, input, target):
        log_sm = F.log_softmax(self(input), dim=1)
        probs = torch.exp(log_sm)
        random_target = torch.multinomial(probs, 1)
        random_log_sm = torch.gather(log_sm, 1, random_target)
        return random_log_sm

    def loss_ewc(self):
        if self.consolidate == False:
            reg = self.prior[0].regularizer(self)#*1
            for i,prior in enumerate(self.prior[1:]):
                reg +=prior.regularizer(self)#*(i+1)
            return reg#/len(self.prior)
        else:
            if self.prior is not None:
                return self.prior.regularizer(self)
            else:
                # ewc loss is 0 if there's no consolidated parameters.
                return (
                    Variable(torch.zeros(1)).to(self.device)
                )
    def loss(self, output, target, **kwargs):
        loss = self.criteria(output, target)
        if self.prior is None:
            return loss
        elif not self.consolidate:
                if len(self.prior==0):
                    return loss
        else:
            return loss + (self.lamda)*self.loss_ewc()

    def on_task_switch(self, **kwargs):
        task = kwargs.get("task")
        self.current_task = task
        self.model.cur_task=self.current_task
        train_loader = kwargs.get("train_loader")
        self.eval()
        prior=GaussianPrior(self, train_loader, reg_matrix=self.reg_matrix, shuffle=self.shuffle)
        if self.consolidate == False:
            self.prior.append(prior)
        else:
            if self.prior is not None:
                self.prior.consolidate(prior, task)
            else:
                self.prior = prior
        if self.log:
            wandb.log({"prior_fn":prior.F_.frobenius_norm(), "global_step": self.loging_step})#, step=self.loging_step)
            wandb.log({"prior_fn_accumulated": self.prior.F_.frobenius_norm(), "global_step": self.loging_step})#, step=self.loging_step)

            sum, eig_val = prior.F_.sum_of_ev()
            wandb.log({"prior_fn_sum_eig_val": sum, "global_step": self.loging_step})  # , step=self.loging_step)
            for i, eigval_dist in enumerate(eig_val):
                plt.bar(x=np.arange(100), height=np.sort(eigval_dist.cpu().numpy())[-100:])
                wandb.log({"prior_fn_eig_val_dist" + str(i): plt,
                           "global_step": self.loging_step})  # ,  step=self.loging_step)

            sum_full, eig_val_full = self.prior.F_.sum_of_ev()
            wandb.log({"prior_fn_consolidated_sum_eig_val": sum_full,
                       "global_step": self.loging_step})  # ,  step=self.loging_step)
            for i, eigval_dist in enumerate(eig_val):
                plt.bar(x=np.arange(100), height=np.sort(eigval_dist.cpu().numpy())[-100:])
                wandb.log({"prior_fn_consoidated_eig_val_dist" + str(i): plt,
                           "global_step": self.loging_step})  # ,  step=self.loging_step)

            wandb.log({"prior_fn_consolidate": self.prior.F_.frobenius_norm(),
                       "global_step": self.loging_step})  # , step=self.loging_step)

            #wandb.log({"mean_inf_dropout": self.prior.F_.frobenius_norm(),
            #           "global_step": self.loging_step})  # , step=self.loging_step)

        del prior

    def parameters(self):
        return self.model.parameters()
    def modules(self):
        return self.model.modules()



class Regularized_model_idr(nn.Module):
    def __init__(self, channel_size, output_size, device="cpu", lamda=100, reg_matrix="diag", shuffle=False,
                 consolidate=True, heads_number=1, log=False, activation='softplus', model=None, **kwargs):
        super(Regularized_model_idr, self).__init__()
        self.device = device
        self.lamda=lamda
        self.criteria = nn.CrossEntropyLoss()
        self.current_task = 0
        self.loging_step = 0
        self.consolidate=consolidate
        self.heads_number = heads_number
        self.reg_matrix = reg_matrix
        self.model = Model_conv_idr(channel_size, output_size, heads_number, activation=activation) if model is None else model
        self.log = log
        if self.consolidate==False:
            self.prior=[]
        else:
            self.prior=None
        self.shuffle = shuffle
        self.sequential_model = None
        #self.model.to(self.device)

    def forward(self,x, target=None, mixup=False, mixup_hidden=False, mixup_alpha=None, task=0, noise=-1):
        return self.model.forward(x,target=target, mixup=mixup, mixup_hidden=mixup_hidden, mixup_alpha=mixup_alpha, task=task, noise=noise)

    def loss_fim_mc_estimate(self, input, target):
        log_sm = F.log_softmax(self(input), dim=1)
        probs = torch.exp(log_sm)
        random_target = torch.multinomial(probs, 1)
        random_log_sm = torch.gather(log_sm, 1, random_target)
        return random_log_sm

    def loss_ewc(self):
        if self.consolidate == False:
            reg = self.prior[0].regularizer(self)#*1
            for i,prior in enumerate(self.prior[1:]):
                reg +=prior.regularizer(self)#*(i+1)
            return reg#/len(self.prior)
        else:
            if self.prior is not None:
                return self.prior.regularizer(self)
            else:
                # ewc loss is 0 if there's no consolidated parameters.
                return (
                    Variable(torch.zeros(1)).to(self.device)
                )
    def loss(self, output, target, **kwargs):
        loss = self.criteria(output, target)
        if self.prior is None:
            return loss
        elif not self.consolidate:
                if len(self.prior==0):
                    return loss
        else:
            return loss + (self.lamda)*self.loss_ewc()

    def on_task_switch(self, **kwargs):
        task = kwargs.get("task")
        self.current_task = task
        self.model.cur_task = self.current_task
        train_loader = kwargs.get("train_loader")
        self.model.noisy_fisher=kwargs.get("noisy_fisher")
        self.eval()
        self.model.memorization()
        prior=GaussianPrior(self, train_loader, reg_matrix=self.reg_matrix, shuffle=self.shuffle)
        if self.consolidate == False:
            self.prior.append(prior)
        else:
            if self.prior is not None:
                self.prior.consolidate(prior, task)
            else:
                self.prior = prior
        if self.log:
            wandb.log({"prior_fn":prior.F_.frobenius_norm(), "global_step": self.loging_step})#, step=self.loging_step)
            wandb.log({"prior_fn_accumulated": self.prior.F_.frobenius_norm(), "global_step": self.loging_step})#, step=self.loging_step)

            sum, eig_val = prior.F_.sum_of_ev()
            wandb.log({"prior_fn_sum_eig_val": sum, "global_step": self.loging_step})#, step=self.loging_step)
            for i, eigval_dist in enumerate(eig_val):
                plt.bar(x=np.arange(100), height=np.sort(eigval_dist.cpu().numpy())[-100:])
                wandb.log({"prior_fn_eig_val_dist" + str(i): plt,
                           "global_step": self.loging_step})  # ,  step=self.loging_step)

            sum_full, eig_val_full = self.prior.F_.sum_of_ev()
            wandb.log({"prior_fn_consolidated_sum_eig_val": sum_full, "global_step": self.loging_step})#,  step=self.loging_step)
            for i, eigval_dist in enumerate(eig_val):
                plt.bar(x=np.arange(100), height=np.sort(eigval_dist.cpu().numpy())[-100:])
                wandb.log({"prior_fn_consoidated_eig_val_dist" + str(i): plt, "global_step": self.loging_step})#,  step=self.loging_step)

            wandb.log({"prior_fn_consolidate": self.prior.F_.frobenius_norm(), "global_step": self.loging_step})#, step=self.loging_step)
        del prior

    def parameters(self):
        return self.model.parameters()
    def modules(self):
        return self.model.modules()


    def make_sequential(self):
        layers = []
        for mod in self.modules():
            mod_class = mod.__class__.__name__
            if mod_class in ['Flatten']:
                layers.append(copy.deepcopy(mod))
            if mod_class in ['Linear', 'Conv2d', 'BatchNorm1d']:
                layers.append(copy.deepcopy(mod))
                layers.append(nn.ReLU())
        self.sequential_model = nn.Sequential(*layers).to(self.device)


class Diagonal(Regularized_model):
    def __init__(self, channel_size, output_size, device="cpu", lamda=100, reg_matrix="diag", shuffle=False,
                 consolidate=True, heads_number=1, activation='softplus', model=None, **args):
        super(Diagonal, self).__init__(channel_size, output_size, device=device, lamda=lamda, reg_matrix="diag",
                                       shuffle=shuffle, consolidate=consolidate, heads_number=heads_number, activation=activation, model=model)

class Fisher_KFAC_reg(Regularized_model):
    def __init__(self, channel_size, output_size, device="cpu", lamda=100, reg_matrix="kfac", shuffle=False,
                 consolidate=True, heads_number=1, log=False, activation='softplus', model=None, **kwargs):
        super(Fisher_KFAC_reg, self).__init__(channel_size, output_size, device=device, lamda=lamda, reg_matrix="kfac",
                                              shuffle=shuffle, consolidate=consolidate, heads_number=heads_number, log=log,
                                              dropout=kwargs.get("dropout"), activation=activation, model=model)

class Fisher_KFAC_reg_id(Regularized_model_idr):
    def __init__(self, channel_size, output_size, device="cpu", lamda=100, reg_matrix="kfac", shuffle=False,
                 consolidate=True, heads_number=1, log=False, activation='softplus', model=None, **args):
        super(Fisher_KFAC_reg_id, self).__init__(channel_size, output_size, device=device, lamda=lamda, reg_matrix="kfac",
                                                 shuffle=shuffle, consolidate=consolidate,
                                                 heads_number=heads_number, log=log, activation=activation, model=model)

class Fisher_EKFAC_reg(Regularized_model):
    def __init__(self, channel_size, output_size, device="cpu", lamda=100, reg_matrix="ekfac", shuffle=False,
                 consolidate=True, heads_number=1, log=False, activation='softplus', model=None, **args):
        super(Fisher_EKFAC_reg, self).__init__(channel_size, output_size, device=device, lamda=lamda, reg_matrix="ekfac",
                                               shuffle=shuffle, consolidate=consolidate,
                                               heads_number=heads_number, log=log, activation=activation, model=model)

class Fisher_EKFAC_eigendiag_reg(Regularized_model):
    def __init__(self, channel_size, output_size, device="cpu", lamda=100, reg_matrix="ekfac", shuffle=False,
                 consolidate=True, heads_number=1, log=False, activation='softplus', model=None, **args):
        super(Fisher_EKFAC_reg, self).__init__(channel_size, output_size, device=device, lamda=lamda, reg_matrix="ekfac",
                                               shuffle=shuffle, consolidate=consolidate,
                                               heads_number=heads_number, log=log, activation=activation, model=model)



