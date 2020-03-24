import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class LeNet(nn.Module):
    def __init__(self, dropout=False, class_per_task=10, heads_number=0, **kwargs):
        super().__init__()
        self.current_task=0
        self.class_per_task = class_per_task
        self.heads_number = heads_number
        self.use_dropout = dropout
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(20000, 500)
        #self.fc2 = nn.Linear(500, 10)
        if self.heads_number>1:
            self.linear = nn.ModuleList()
            for t in range(heads_number):
                self.linear.append(nn.Linear(500, self.class_per_task))
        else:
            self.linear  = nn.Linear(500, self.class_per_task)

    def forward(self, x, task=None, current_task_number=None):
        x = F.relu(F.max_pool2d(self.conv1(x), 1))
        x = F.relu(F.max_pool2d(self.conv2(x), 1))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = F.dropout(x, training=self.training)

        if not task==None:
            x = self.linear[task](x)
        else:
            if current_task_number is None:
                x = self.linear[self.current_task](x)
            else:
                x = [self.linear[i](x) for i in range(current_task_number)]

        #x = self.fc2(x)
        return x