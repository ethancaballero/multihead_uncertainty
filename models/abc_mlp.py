import torch
import torch.nn as nn
from torch import autograd
#import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
#Abstract base class


class Flatten(nn.Module):
    def forward(self, input):
        return input.flatten(1)


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, bias=True, dropout=False, heads_number=1, **args):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.heads_number = heads_number
        self.criteria = nn.CrossEntropyLoss()
        self.layers = nn.ModuleList([
            Flatten(),
            nn.Linear(self.input_size, self.hidden_size, bias=bias), nn.Softplus(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=bias), nn.Softplus(),
        ])
        if dropout:
            self.layers.insert(2,nn.Dropout(0.5, True))
            self.layers.insert(5, nn.Dropout(0.5, True))
        if self.heads_number>1:
            self.linear = nn.ModuleList()
            for t in range(heads_number):
                self.linear.append(nn.Linear(self.hidden_size, self.output_size))
        else:
            self.linear  = nn.Linear(self.hidden_size, self.output_size, bias=bias)


    def loss(self, output, target):
        return self.criteria(output, target)
    def on_task_switch(self, **kwargs):
        pass
    def forward(self, x, task=None):
        for layer in self.layers:
            x = layer(x)
        if not task==None:
            x = self.linear[task](x)
        else:
            x = self.linear(x)
        return x