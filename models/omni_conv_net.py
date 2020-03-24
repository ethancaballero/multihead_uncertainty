import torch
import torch.nn as nn
from torch import autograd
#import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
from .InformationDropout import InformationDropoutLayer
from torch.nn import Conv2d
#Abstract base class


class Model_conv(nn.Module):
    def __init__(self, channel_size, output_size, heads_number=1, dropout=False, activation='softplus', **args):
        super(Model_conv, self).__init__()
        self.channel_size = channel_size
        in_linear = channel_size*4
        self.output_size = output_size
        self.heads_number = heads_number
        self.cur_task = 0
        self.criteria = nn.CrossEntropyLoss()
        if activation=="relu":
            act=nn.ReLU()
        elif activation=='softplus':
            act=nn.Softplus()
        else:
            raise NotImplementedError
        if dropout:
            p = 0.5
        else:
            p = 0.0
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, self.channel_size, 3, padding=1),
            nn.Dropout2d(p, True),
            act,
            nn.MaxPool2d(2, stride=2)
        )  # 1@28*28 > 64@14*14
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.channel_size, self.channel_size, 3, padding=1),
            nn.Dropout2d(p, True),
            act,
            nn.MaxPool2d(2, stride=2)
        )  # 64@14*14 > 64@7*7
        self.layer3 = nn.Sequential(
            nn.Conv2d(self.channel_size, self.channel_size, 2, padding=1),
            nn.Dropout2d(p, True),
            act,
            nn.MaxPool2d(2, stride=2)
        )  # 64@7*7 > 64@4*4
        self.layer4 = nn.Sequential(
            nn.Conv2d(self.channel_size, self.channel_size, 3, padding=1),
            nn.Dropout2d(p, True),
            act,
            nn.MaxPool2d(2, stride=2)
        )  # 64@4*4 > 64@2*2
        # resize 64@2*2 > 256
        self.linear = nn.ModuleList()
        for t in range(heads_number):
            self.linear.append(nn.Linear(in_linear, self.output_size))

        self.layers = nn.ModuleList([
            self.layer1, self.layer2, self.layer3, self.layer4, nn.Flatten()#, self.linear
        ])

    def loss(self, output, target):
        return self.criteria(output, target)
    def on_task_switch(self, **kwargs):
        pass
    def forward(self, x, task=-1):
        for layer in self.layers:
            x = layer(x)
        if task==-1:
            x = self.linear[self.cur_task](x)
        else:
            x = self.linear[task](x)
        return x


class Model_conv_idr(nn.Module):
    def __init__(self, channel_size, output_size, heads_number=1, activation='softplus', **kwargs):
        super(Model_conv_idr, self).__init__()
        self.channel_size = channel_size
        self.output_size = output_size
        self.heads_number = heads_number
        in_linear=channel_size*4
        self.learnable_prior = kwargs.get("learnable_prior")
        self.cur_task = 0
        self.noisy_fisher=False
        self.criteria = nn.CrossEntropyLoss()
        self._memorizing = False
        if activation=="relu":
            act = nn.ReLU()
        elif activation=='softplus':
            act = nn.Softplus()
        else:
            raise NotImplementedError
        self.layer1 = nn.Sequential(
            InformationDropoutLayer(Conv2d, max_alpha=0.7, activate=True, activation_function=act, learnable_prior=self.learnable_prior,
                                    **dict(in_channels=1, out_channels=self.channel_size,kernel_size=3, padding=1)),
            nn.MaxPool2d(2, stride=2)
        )  # 1@28*28 > 64@14*14
        self.layer2 = nn.Sequential(
            InformationDropoutLayer(Conv2d, max_alpha=0.7, activate=True,  activation_function=act, learnable_prior=self.learnable_prior,
                                    **dict(in_channels=self.channel_size, out_channels=self.channel_size, kernel_size=3,
                                           padding=1)),
            nn.MaxPool2d(2, stride=2)
        )  # 64@14*14 > 64@7*7
        self.layer3 = nn.Sequential(
            InformationDropoutLayer(Conv2d, max_alpha=0.7, activate=True,  activation_function=act, learnable_prior=self.learnable_prior,
                                    **dict(in_channels=self.channel_size, out_channels=self.channel_size, kernel_size=2,
                                           padding=1)),
            nn.MaxPool2d(2, stride=2)
        )  # 64@7*7 > 64@4*4
        self.layer4 = nn.Sequential(
            InformationDropoutLayer(Conv2d, max_alpha=0.7, activate=True,  activation_function=act, learnable_prior=self.learnable_prior,
                                    **dict(in_channels=self.channel_size, out_channels=self.channel_size, kernel_size=3,
                                           padding=1)),
            nn.MaxPool2d(2, stride=2)
        )  # 64@4*4 > 64@2*2
        # resize 64@2*2 > 256
        self.linear = nn.ModuleList()
        for t in range(heads_number):
            self.linear.append(nn.Linear(in_linear, self.output_size))

        self.layers = nn.ModuleList([
            self.layer1, self.layer2, self.layer3, self.layer4, #, self.linear
        ])

    def loss(self, output, target):
        return self.criteria(output, target)
    def on_task_switch(self, **kwargs):
        pass

    def forward(self, x, prior=None, task=-1, noise=True):
        if self._memorizing:
            noise=self.noisy_fisher
        out = x
        kl = Variable(torch.zeros(x.size()[0]).type(torch.FloatTensor))
        if torch.cuda.is_available():
            kl = kl.cuda()
        for layer in self.layers:
            for sublayer in layer:
                # print(x.shape)
                if isinstance(sublayer, InformationDropoutLayer): # and prior is not None:
                    out = sublayer(out, noise=noise)  # prior.F_.evals[layer.layer])
                else:
                    out = sublayer(out)
                if isinstance(out, list):  # InfoActivations):
                    kl = kl + out[1]  # .kl  # Gather KL divergences
                    out = out[0]  # out.activations  # Prepare output for next layer
                # assert output.size()[1:] == size
                # print(output)
        out=out.view((out.shape[0],-1))
        if task == -1:
            out = self.linear[self.cur_task](out)
        else:
            out = self.linear[task](out)
        if not self.training:
            return out
        else:
            return [out, kl]  # InfoActivations(activations=out, kl=kl)

    def memorization(self, mode: bool = ...):
        self._memorizing=True
        return self
    def train(self, mode: bool = ...):
        super(Model_conv_idr, self).train(mode)
        self._memorizing=False
        return self
    def eval(self):
        super(Model_conv_idr, self).eval()
        self._memorizing=False
        return self