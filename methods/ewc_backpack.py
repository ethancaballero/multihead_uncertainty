from copy import deepcopy

import torch
import torch.nn as nn
from backpack import extend
from backpack.core.layers import Flatten
from backpack.extensions import KFRA
from torch.autograd import Variable
from backpack import backpack


class Regularized_model_gn(object):
    def __init__(self, input_size, output_size, hidden_size, device="cpu", lamda=100, bias=True, reg_matrix="diag", **args):
        super(Regularized_model_gn, self).__init__()
        self.criteria = nn.CrossEntropyLoss()
        self.device = device
        self.lamda=lamda
        self.prior=None
        self.reg_matrix = reg_matrix
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.kfra=[]
        self.params_previous=None

        self.layers = extend(nn.Sequential(*[ Flatten(), nn.Linear(self.input_size, self.hidden_size, bias=bias), nn.ReLU()] + \
            [nn.Linear(self.hidden_size, self.hidden_size, bias=bias), nn.ReLU()] + \
            [nn.Linear(self.hidden_size, self.output_size, bias=bias)]))

    def __call__(self,input):
        return self.layers(input)

    def train(self):
        self.layers.train()

    def eval(self):
        self.layers.eval()

    def zero_grad(self):
        self.layers.zero_grad()

    def on_task_switch(self, train_loader, task, **kwargs):
        loss_function = extend(self.criteria)
        model = extend(self.layers)
        print("Coputing GN")
        for batch_idx, (data, target) in enumerate(train_loader):
            model.zero_grad()
            output = model(data.view(len(data),-1).to(self.device))
            loss = loss_function(output, target.to(self.device))
            with backpack(KFRA()):
                loss.backward()
        print("Done")
        params = []
        for i, param in enumerate(model.parameters()):# param.data=torch.ones(10)
            params.append({'data': deepcopy(param.data), 'kfra': deepcopy(param.kfra)})
            param.kfra=None

        if self.params_previous is None:
            self.params_previous = deepcopy(params)
        else:
            self.consolidate(deepcopy(params), task)

    def consolidate(self, new_params, task):
        for param_old, param_new in zip(self.params_previous, new_params):
            param_old['data'].data=deepcopy(param_new['data'].data)
            param_old['kfra'][0].data = ((param_old['kfra'][0].data*task) + param_new['kfra'][0].data)/(task+1)
            if len(param_old['kfra']) > 1:
                param_old['kfra'][1].data = ((param_old['kfra'][1].data*task) + param_new['kfra'][1].data)/(task+1)


    def loss_regularizer(self):
        if self.params_previous is not None:
            loss = 0
            for params_car, params_prev in zip(self.layers.parameters(),self.params_previous):
                diff = (params_car-params_prev['data'].detach())#.view(-1)
                if len(params_prev['kfra'])==1:
                    #bias
                    loss+= ( torch.dot( torch.mv(params_prev['kfra'][0],diff.view(-1)).view(-1), diff.view(-1)))
                elif len(params_prev['kfra'])==2:
                    # Kroneckor vector
                    a = torch.mm(params_prev['kfra'][0], diff)
                    b = torch.mm(a, params_prev['kfra'][1]).view(-1)
                    loss += ( torch.dot(diff.view(-1),b))

            return loss

        else:
            # ewc loss is 0 if there's no consolidated parameters.
            return (
                Variable(torch.zeros(1)).to(self.device)
            )

    def loss(self, output, target, **kwargs):
        loss = self.criteria(output, target)
        return loss + self.lamda*self.loss_regularizer()


