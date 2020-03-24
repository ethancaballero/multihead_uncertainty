import torch
import torch.nn as nn
from torch import autograd
#import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
#Abstract base class
from .InformationDropout import InformationDropoutLayer, InfoActivations
from torch.nn import Module, ModuleList, Conv2d, MaxPool2d, ZeroPad2d, Linear


class Flatten(nn.Module):
    def forward(self, input):
        return input.flatten(1)


class Model(nn.Module):
    def __init__(self, input_channels, output_size, hidden_size=64, bias=True, dropout=False, **args):
        super(Model, self).__init__()
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.criteria = nn.CrossEntropyLoss()
        self.layer1 = InformationDropoutLayer(Conv2d, max_alpha=0.7, activate=True, **dict(in_channels=self.input_channels, out_channels=self.hidden_size, kernel_size=3, padding=1))
            #nn.Conv2d(1, 64, 3),
            #nn.ReLU(),
        self.mp_1 = nn.MaxPool2d(2, stride=2)
        #)  # 1@28*28 > 64@12*12
        self.layer2 = InformationDropoutLayer(Conv2d, max_alpha=0.7, activate=True,
                                    **dict(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=3, padding=1))
            #nn.Conv2d(64, 64, 3),
            #nn.ReLU(),
        self.mp_2 = nn.MaxPool2d(2, stride=2)
        #)  # 64@12*12 > 64@4*4
        self.layer3 = InformationDropoutLayer(Conv2d, max_alpha=0.7, activate=True,
                                    **dict(in_channels=self.hidden_size, out_channels=self.hidden_size,
                                           kernel_size=3, padding=1))
            #nn.Conv2d(64, 64, 3),
            #nn.ReLU(),
        self.mp_3 = nn.MaxPool2d(2, stride=2)
        #)  # 128@21*21 > 128@9*9
        self.layer4 = InformationDropoutLayer(Conv2d, max_alpha=0.7, activate=True,
                                    **dict(in_channels=self.hidden_size, out_channels=self.hidden_size,
                                           kernel_size=3, padding=1))
            #nn.Conv2d(64, 64, 2),
            #nn.ReLU()
        #)  # 128@9*9 > 256@6*6
        # resize 256@6*6 > 9216 > 4096
        self.linear = nn.Linear(576,self.output_size)
        self.final_act = nn.Sigmoid()
        #)
        #self.final = nn.Sequential(
        #    nn.Linear(4096, 1, bias=False),
        #    #                nn.Linear(4096,1),
        #    nn.Sigmoid()
        #)

        self.layers = nn.ModuleList([
            self.layer1, self.mp_1, self.layer2, self.mp_2, self.layer3,
            self.mp_3, self.layer4, Flatten(), self.linear
        ])
        if dropout:
            self.layers.insert(2,nn.Dropout(0.5, True))
            self.layers.insert(5, nn.Dropout(0.5, True))

    def loss(self, output, target):
        return self.criteria(output, target)
    def on_task_switch(self, **kwargs):
        pass

    def forward(self, x, prior=None):
        out=x
        kl = Variable(torch.zeros(x.size()[0]).type(torch.FloatTensor))
        if torch.cuda.is_available():
            kl=kl.cuda()
            #print(kl)
        for layer in self.layers:
            #print(x.shape)
            if isinstance(layer,InformationDropoutLayer) and prior is not None:
                out = layer(out,None) # prior.F_.evals[layer.layer])
            else:
                out = layer(out)
            if isinstance(out, list):    # InfoActivations):
                #print("out",out[1].device)
                #print(kl.device)
                kl = kl + out[1].to(kl.device)#.kl  # Gather KL divergences
                out = out[0] #out.activations  # Prepare output for next layer
            #

            #assert output.size()[1:] == size
            #print(output)
        if not self.training:
            return out
        else:
            return [out,kl]  #InfoActivations(activations=out, kl=kl)