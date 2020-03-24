import torch
import torch.nn as nn
from torch.autograd import Variable
from .InformationDropout import InformationDropoutLayer, InfoActivations
from torch.nn import Module, ModuleList, Conv2d, MaxPool2d, ZeroPad2d, Linear



class Flatten(nn.Module):
    def forward(self, input):
        return input.flatten(1)


class Model_Inf_Dropout(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, bias=True, **args):
        super(Model_Inf_Dropout, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        #self.criteria = nn.CrossEntropyLoss()
        self.layers = nn.ModuleList([
            Flatten(),
            InformationDropoutLayer(Linear, max_alpha=0.7, activate=True, **dict(output_size=self.hidden_size ,in_features=self.input_size, out_features=self.hidden_size)),
            InformationDropoutLayer(Linear, max_alpha=0.7, activate=True, **dict(output_size=self.hidden_size, in_features=self.hidden_size, out_features=self.hidden_size)),
            nn.Linear(self.hidden_size, self.output_size, bias=bias)
            #InformationDropoutLayer(Linear, max_alpha=0.7, activate=True, **dict(output_size=self.hidden_size, in_features=self.hidden_size, out_features=self.output_size))
        ])
    def loss(self, output, target):
        return self.criteria(output, target)
    def on_task_switch(self, **kwargs):
        pass
    def forward(self, x, prior=None):
        out=x
        kl = Variable(torch.zeros(x.size()[0]).type(torch.FloatTensor))
        if torch.cuda.is_available():
            kl=kl.cuda()
        for layer in self.layers:
            #print(x.shape)
            if isinstance(layer,InformationDropoutLayer) and prior is not None:
                out = layer(out,None) # prior.F_.evals[layer.layer])
            else:
                out = layer(out)
            if isinstance(out, list):    # InfoActivations):
                kl = kl + out[1]#.kl  # Gather KL divergences
                out = out[0] #out.activations  # Prepare output for next layer
            #assert output.size()[1:] == size
            #print(output)
        if not self.training:
            return out
        else:
            return [out,kl]  #InfoActivations(activations=out, kl=kl)