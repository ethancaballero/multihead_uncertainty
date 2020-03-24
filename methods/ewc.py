import torch
import wandb
import torch.nn as nn
import copy
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils import one_hot_embedding
from utils.prior import GaussianPrior
from models.abc_mlp import Model as Model
from models.abc_mlp_inf_drop import Model_Inf_Dropout as Model_idr




class Regularized_model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, device="cpu", lamda=100, bias=True, reg_matrix="diag", shuffle=False, consolidate=True, dropout=False, model=None, **kwargs):
        super(Regularized_model, self).__init__()
        self.criteria = nn.CrossEntropyLoss()
        self.device = device
        self.lamda=lamda
        self.current_task = 0
        self.consolidate=consolidate
        self.reg_matrix = reg_matrix
        print(model)
        self.model= Model(input_size, output_size, hidden_size, bias=bias, dropout=dropout) if model is None else model
        self.model.to(device)

        if self.consolidate==False:
            self.prior=[]
        else:
            self.prior=None
        self.shuffle = shuffle

    def forward(self,x, task=None):
        #x=x.view(len(x), -1)
        return self.model.forward(x, task=task)

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
        train_loader = kwargs.get("train_loader")
        self.model.eval()
        prior=GaussianPrior(self, train_loader, reg_matrix=self.reg_matrix, shuffle=self.shuffle)
        if self.consolidate == False:
            self.prior.append(prior)
        else:
            if self.prior is not None:
                self.prior.consolidate(prior, task)
            else:
                self.prior = prior
        #wandb.log({"prior_fn":prior.F_.frobenius_norm()})
        del prior
    def parameters(self):
        return self.model.parameters()
    def modules(self):
        return self.model.modules()

class Regularized_model_uncertainty(Regularized_model):
    def __init__(self, input_size, output_size, hidden_size=64, device="cpu", lamda=100, bias=True, reg_matrix="diag", shuffle=False, consolidate=True, dropout=False, model=None, use_uncertainty=True, **kwargs):
        super(Regularized_model_uncertainty, self).__init__(input_size, output_size, hidden_size, device, lamda, bias, reg_matrix, shuffle, consolidate, dropout, model, **kwargs)
        self.use_uncertainty = use_uncertainty

    def meta_named_parameters(self):
        return self.model.meta_named_parameters()
    def forward(self,x, task=None, current_task_number=None):
        return self.model.forward(x,task=task, current_task_number=current_task_number)

    def loss(self, output, target, **kwargs):
        if self.use_uncertainty:
            epoch, num_classes, annealing_step = kwargs.get('epoch', None), kwargs.get('num_classes', None), kwargs.get('annealing_step', 10)
            target = one_hot_embedding(target, num_classes=num_classes)
            loss = self.criteria(output, target, epoch, num_classes, annealing_step, self.device)
        else:
            loss = self.criteria(output, target)
        if self.prior is None:
            return loss
        elif not self.consolidate:
                if len(self.prior==0):
                    return loss
        else:
            return loss + (self.lamda)*self.loss_ewc()


class Regularized_model_idr(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=50, device="cpu", lamda=100, bias=True, reg_matrix="diag", shuffle=False, consolidate=True, model=None, **args):
        super(Regularized_model_idr, self).__init__()
        self.device = device
        self.lamda=lamda
        self.criteria = nn.CrossEntropyLoss()
        self.current_task = 0
        self.consolidate=consolidate
        self.reg_matrix = reg_matrix
        self.model = Model_idr(input_size, output_size, hidden_size, bias=bias) if model is None else model
        self.model.to(device)
        #if device !='cpu':
        #    torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if self.consolidate==False:
            self.prior=[]
        else:
            self.prior=None
        self.shuffle = shuffle
        self.sequential_model = None

    def forward(self,x):
        return self.model.forward(x, self.prior)

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
        wandb.log({"prior_fn":prior.F_.frobenius_norm()})
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


class Fisher_KFAC_reg_uncertainty(Regularized_model_uncertainty):
    def __init__(self, input_size, output_size, hidden_size=64, device="cpu", lamda=100, bias=True, reg_matrix="kfac", shuffle=False, consolidate=True, model=None, **kwargs):
        super(Fisher_KFAC_reg_uncertainty, self).__init__(input_size, output_size, hidden_size, device=device, lamda=lamda, bias=bias, reg_matrix="kfac", shuffle=shuffle, consolidate=consolidate, dropout=kwargs.get("dropout"), model=model)

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