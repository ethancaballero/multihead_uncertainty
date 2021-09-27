import copy
import torch
import wandb
import easydict
import random
import argparse
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm as tqdm
from utils.dataloaders import splitmnist as dataloader

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

LR = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
print(torch.cuda.is_available())

# Prepare data for chosen experiment
(train_datasets, valid_datasets, test_datasets), config, classes_per_task = dataloader.get_multitask_experiment(
    name='splitMNIST', scenario='domain', tasks=2, data_dir='./data',
    verbose=True, exception=False)
def set_seed(manualSeed):
    #####seed#####
    np.random.seed(manualSeed)
    #random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # if you are suing GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(manualSeed)
        #torch.cuda.manual_seed_all(manualSeed)
        #torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    ######################################################

class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        h_hidden = args.h_size
        self.h_hidden = h_hidden
        if self.args.multi_head:
            self.class_numb = 5
            self.out = nn.ModuleList([nn.Linear(self.h_hidden, self.class_numb), nn.Linear(self.h_hidden, self.class_numb)])
        else:
            self.class_numb = 10
            self.out = nn.Linear(self.h_hidden, self.class_numb)
        self.layer_in = nn.Linear(28 * 28, h_hidden)

    def forward(self, x, task):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.layer_in(x))
        if self.args.multi_head:
            output = self.out[task](x)
        else:
            output = self.out(x)
        return output


def train(net, optimizer, epoch, train_loader, task):
    net.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(DEVICE), target.to(DEVICE)
        net.zero_grad()
        out = net(data, task)
        loss = F.cross_entropy(out, target)
        loss.backward()
        optimizer.step()


def test(net, n_tasks, datasets, current_task=None, split="test"):
    net.eval()
    correct = 0
    test_loss = 0
    acc_current = None
    acc_tasks = []
    if current_task is None:
        for task in range(n_tasks):
            with torch.no_grad():
                loader = torch.utils.data.DataLoader(datasets[task], batch_size=64)
                for data, target in loader:
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    output = net(data, task)
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()
                print('Task {:.0f}, {} Accuracy: {}/{} ({:.2f}%)\n'.format(task, split, correct, len(loader.dataset),
                                                                           100. * correct / len(loader.dataset)))
                acc_tasks.append(100. * correct / len(loader.dataset))
                wandb.log({
                    f"Avv. {split} acc. task {task} ": 100. * correct / len(loader.dataset)})
            correct = 0

        wandb.log({
            f"Avv. {split} Accuracy ": np.mean(acc_tasks)})
        acc_current = np.mean(acc_tasks)
    else:
        task = current_task
        with torch.no_grad():
            loader = torch.utils.data.DataLoader(datasets[current_task], batch_size=64)
            for data, target in loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = net(data, task)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
            print('Task {:.0f}, {} Accuracy: {}/{} ({:.2f}%)\n'.format(task, split, correct, len(loader.dataset),
                                                                       100. * correct / len(loader.dataset)))


        acc_current = correct / len(loader.dataset)
    return acc_current

def _get_optimizer(model,lr=None):
    if lr is None: lr=LR
    return torch.optim.Adam(model.parameters(),lr=lr)
def get_model(model):
    return copy.deepcopy(model.state_dict())
def set_model_(model,state_dict):
    model.load_state_dict(copy.deepcopy(state_dict))
    return

def wandb_init(net, group, name, config_dict):
    run = wandb.init(
        project='Growing Net2',
        name=name,
        group=group,
        config=config_dict,
        reinit=True
        #resume="allow",
    )
    wandb.watch(net)
    print(f"Using wandb. Group name: {group} run name: {name}")
    return run

# Training settings
config = easydict.EasyDict({
    'n_tasks': 2,
    'lr': 0.001,
    'epochs': 40,
    'n_runs':3,
    'seed':np.random.randint(4294967290),
    #'lr_factor': 0.1,
    #lr_min=0.00001
    #LR_PATIENCE = 3
})


def main(args):
    h_hidden = args.h_size
    config.update({'h_size': args.h_size,
                   'multi_head': args.multi_head,
    })
    #for h_hidden in [10, 100, 1000, 10000, 100000, 1000000]: #, 10000000]:
    #for run in range(config.n_runs):
    best_acc = 0
    accs = []
    set_seed(config.seed)
    best_model = None
    net = MLP(args).to(DEVICE)
    optimizer = _get_optimizer(net, config.lr)
    wandb_init(net, f'hs_{h_hidden}', 'run', config)
    try:
        for task in range(config.n_tasks):
            print("Training on task: " + str(task))
            train_loader = torch.utils.data.DataLoader(train_datasets[task], batch_size=64)
            for epoch in range(config.epochs):
                print("Epoch: " + str(epoch))
                train(net, optimizer, epoch, train_loader, task)
                valid_acc = test(net, 2, valid_datasets, current_task=task, split="valid")

                if valid_acc > best_acc:
                    best_acc = valid_acc
                    best_model = get_model(net)
            set_model_(net, best_model)
            acc_av = test(net, task + 1, test_datasets, current_task=None, split="test")
            accs.append(acc_av)
    except KeyboardInterrupt:
        pass
    slope = accs[1] - accs[0]
    wandb.log({"slope": slope})
    del(net)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--h_size', default=10, type=int, help='hidden size')
    parser.add_argument('--multi_head', type=str2bool, default=True)
    prs = parser.parse_args()
    main(prs)
