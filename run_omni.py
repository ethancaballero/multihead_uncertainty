import copy
import time
import torch
import wandb
import atexit
import random
import easydict
import argparse
import torch.optim as optim
from datetime import datetime
import torch.nn.functional as F
from utils.dataloaders.omniglot import SplitOmniglot
from models.omni_conv_net import Model_conv, Model_conv_idr
from methods.ewc_conv import Fisher_KFAC_reg, Fisher_KFAC_reg_id, Fisher_EKFAC_reg

import os
import numpy as np
from utils.clb.utils.metrics import accuracy, AverageMeter
#import matplotlib.pyplot as plt
#from tqdm import tqdm
#%matplotlib inline


parser = argparse.ArgumentParser()
parser.add_argument('--log', action='store_true',
                    help='Log the results?')
parser.add_argument('--multi_head', action='store_true',
                    help='use several heads for the several tasks')
parser.add_argument('--device', default="cuda", type=str,
                    help='device cuda or cpu')
parser.add_argument('--n_epochs', default=100, type=int,
                    help='depochs number per task')
parser.add_argument('--samples_no_f', default=None, type=int,
                    help='n samples fisher')
parser.add_argument('--lamda', default=100, type=int,
                    help='lamda')
parser.add_argument('--channels', default=256, type=int,
                    help='channel size')
parser.add_argument('--n_tasks', default=50, type=int,
                    help='n_tssks') #number of tasks (default 50 for 50 alphabets)
parser.add_argument('--class_per_task', default=5, type=int,
                    help='n_classes') #number of classes per task
parser.add_argument('--augmentations', default=100, type=int,
                    help='n_augmentations') #number of random augmentations per image (random rotations and translations)
parser.add_argument('--method', choices=['kfac','diag','kfac_id','mlp','kfac_dropout', 'ekfac'], default="kfac", type=str,
                    help='Method to run')
parser.add_argument('--dropout', default=False, type=bool,
                    help='Use standard dropout layer')
parser.add_argument('--log_dir', default="2_o_methods", type=str)
parser.add_argument('--noise_val', action='store_true', help='Add noise at val time')
parser.add_argument('--beta', default=0.001, type=float, help='Beta')
parser.add_argument('--wandb', default='2o_methods_omniglot', type=str, help='wandb')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--noisy_fisher', default=0, type=int, help='Wether of not caluclate fisher with noise')
parser.add_argument('--activation', default='softplus', choices=['relu', 'softplus'], type=str, help='activation funciton')
parser.add_argument('--comment', default='', type=str)
parser.add_argument('--learnprior', default=1, type=int, help='Wether of not caluclate fisher with noise')
parser.add_argument('--sport_period', default=0, type=int, help='Noise periodicity')

prs = parser.parse_args()

if prs.method=="kfac_id":
    prs.dropout = False
elif prs.method=="kfac_dropout":
    prs.dropout = True

log_dir = prs.log_dir
dataroot = '/network/home/ostapeno/dev/2o_methods/.data/'
# if os.environ.get('SCRATCH') is not None:
#     log_dir = os.environ['SCRATCH'] + '/2_o_methods/'
# elif os.environ.get('/network/tmp1/ostapeno/') is not None:
#     log_dir = os.environ['/network/tmp1/ostapeno/'] + '/2_o_methods/'
# else:
#     log_dir = '2_o_methods/'

args = easydict.EasyDict({
    "dataset": "split-omniglot",
    "method": prs.method,
    "log_dir":log_dir,
    "multi_head": prs.multi_head,
    "comment":"_ntasks_"+str(prs.n_tasks)+"_channel_size_"+str(prs.channels,)+"_"+prs.method+" "+prs.activation+" "+prs.comment,
    "n_epochs": prs.n_epochs,
    "batch_size": prs.batch_size,
    "samples_no_f": prs.samples_no_f,
    "learning_rate": prs.lr,
    "momentum": 0.5,
    "log_interval": 10,
    "task_number": prs.n_tasks,
    "class_per_task": prs.class_per_task,
    "augmentations":prs.augmentations,
    "omniglot_size":(28, 28),
    "seed": 100,
    "kfac_batch":True,
    "log":prs.log,
    "log_interval_epoch":2,
    "device": prs.device,
    "lamda":prs.lamda,
    "channel_size":prs.channels,
    "beta": prs.beta,
    "beta_factor": .9,
    "max_epoch_id":100,
    "beta_min":prs.beta*0.1,
    "dropout":prs.dropout,
    "noise_val":prs.noise_val,
    "wandb":prs.wandb,
    "noisy_fisher": prs.noisy_fisher,
    "activation": prs.activation,
    "learnable_prior": prs.learnprior,
    "sport_period": prs.sport_period
})
if args.log:
    ts = time.time()
    # wandb.init(sync_tensorboard=True, project="2o_methods", name=datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S') + args.comment)
    wandb.init(project=args.wandb, name=datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S') + args.comment)
print(args)
best_val_acc_kfac=[0]*args.task_number


if torch.cuda.is_available() and args.device!="cpu":
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

#####seed#####
manualSeed = args.seed
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# if you are suing GPU
if torch.cuda.is_available() and args.device!="cpu":
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
######################################################

def accumulate_acc(output, target, task, meter):
    meter.update(accuracy(output, target), len(target))
    return meter

def eval(model, dataset, cur_task_id, global_step, device, split="validation", noise_val=False):
    model.eval()
    test_loss = 0
    correct = 0
    all_data_size = 0
    accs = {}
    with torch.no_grad():
        for task_id in range(cur_task_id+1):
            valid_loader = dataset[task_id]["validation"]
            correct_task=0
            for data, target in valid_loader:
                if args.multi_head:
                    output = model(data.to(device), task=task_id, noise=noise_val)
                else:
                    output = model(data.to(device))
                if isinstance(output, list):
                    output=output[0]
                output = F.log_softmax(output,dim=1)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct_ = pred.eq(target.view_as(pred).to(device)).sum().item()
                correct_task+=correct_
                correct+=correct_
                all_data_size += len(data)
            valid_acc_task = 100. * correct_task / len(valid_loader.dataset)
            if not noise_val:
                print('Task {:.0f}, {} Accuracy: {}/{} ({:.2f}%)\n'.format(task_id, split, correct_task, len(valid_loader.dataset),
                                                                     valid_acc_task))
            if args.log:
                accs["Task_valid_acc_ "+"noisy_"+str(noise_val)+"_"+str(task_id)] = valid_acc_task
            accs["global_step"]=model.loging_step
    if args.log:
        wandb.log(accs)
    acc = 100. * correct / all_data_size
    return acc

def train(name, model, optimizer, dataset, device="cpu"):
    writer=None
    valid_accs = []
    for task_dataset in dataset:
        train_loader = task_dataset["train"]
        # valid_loader = task_dataset["validation"]
        # test_loader = task_dataset["test"]
        cur_task_id = train_loader.dataset.task_id
        beta = args.beta
        model.train()
        print("Task: ", cur_task_id)
        # standart training
        ####
        max_acc_task = 0
        add_noise=False
        last_sport = 0
        ####
        for epoch in range(args.n_epochs):
            model.loging_step += 1
            acc = AverageMeter()
            add_noise = (last_sport >= args.sport_period)
            if add_noise:
                last_sport = 0
            else:
                last_sport += 1
            print("Noise ",add_noise)
            for batch_idx, (data, target) in enumerate(train_loader):
                    model.train()
                    optimizer.zero_grad()
                    model.zero_grad()
                    if args.multi_head:
                        output = model(data.to(device), task=cur_task_id, noise=add_noise)
                    else:
                        output = model(data.to(device), noise=add_noise)
                    reg = 0
                    if name=="kfac_id":
                        reg = beta * torch.mean(output[1]) / len(data) #.kl) / len(data)
                        output = output[0] #.activations
                    loss = model.loss(output, target.to(device)) + reg
                    loss.backward()
                    optimizer.step()
                    acc = accumulate_acc(F.log_softmax(output, dim=1), target, 0, acc)
                    if batch_idx % args.log_interval == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tacc.tr: {:.2f}'.format(
                            epoch, batch_idx * len(data), len(train_loader.dataset),
                                   100. * batch_idx / len(train_loader), loss.item(), acc.val))

            global_step = args.n_epochs*cur_task_id+epoch
            if epoch % args.log_interval_epoch==0:
                valid_acc = eval(model, dataset, cur_task_id, global_step, device)
                if valid_acc>max_acc_task:
                    max_acc_task=valid_acc
                print('Validation task {:.0f}, Epoch: {}, Average Acc. {:.2f}%'.format(cur_task_id, epoch, valid_acc))
                #if name=="kfac":
                #    best_val_acc_kfac[n_datasets]=max(best_val_acc_kfac[n_datasets],valid_acc)
                ### loging
                valid_acc_noisy = eval(model, dataset, cur_task_id, global_step,device, noise_val=True)
                if args.log:
                    wandb.log({
                        "avv_valid_acc_sofar": valid_acc, "global_step": model.loging_step})
                    wandb.log({
                        "avv_valid_noisy_acc_sofar": valid_acc_noisy, "global_step": model.loging_step})
                valid_accs.append((valid_acc,cur_task_id,epoch))
            if name == "kfac_id":
                print(beta)
                if beta > args.beta_min:
                    beta *= args.beta_factor
                else:
                    beta = args.beta_min
        if args.log:
            wandb.log({
                "max_acc_task": max_acc_task, "global_step": model.loging_step})
        model.on_task_switch(train_loader=train_loader, device=device, samples_no=args.samples_no_f, task=cur_task_id, noisy_fisher=args.noisy_fisher)
        train_loader.dataset.empty_cache()

if __name__ == '__main__':
    ts = time.time()
    log_dir=None
    net = Model_conv(args.channel_size, args.class_per_task, dropout=args.dropout, heads_number=args.task_number,
                     activation=args.activation, learnable_prior=args.learnable_prior)
    if args.log:
        wandb.config.update(args)
    dev = torch.device(args.device)
    if args.method=="kfac" or args.method=="kfac_dropout":
        model = Fisher_KFAC_reg(args.channel_size, args.class_per_task, device=dev, lamda=args.lamda, dropout=args.dropout,
                                heads_number=args.task_number, log=args.log, activation=args.activation, model=net)
    elif args.method == "kfac_id":
        model = Fisher_KFAC_reg_id(args.channel_size, args.class_per_task, device=dev, lamda=args.lamda,
                                   heads_number=args.task_number, log=args.log, activation=args.activation, model=net)
    elif args.method == "ekfac":
        model = Fisher_EKFAC_reg(args.channel_size, args.class_per_task, device=dev, lamda=args.lamda,
                                 heads_number=args.task_number, log=args.log, activation=args.activation, model=net)
    model = model.to(dev)
    #if args.log:
    #    wandb.watch(model.model, log="all")

    omniglot_dataset = SplitOmniglot(dataroot, num_tasks=args.task_number, classes_per_task=args.class_per_task,
                                     batch_size=args.batch_size, augmentations=args.augmentations, seed=args.seed,
                                     incremental_class=False, image_size=(28,28), download=True)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train(args.method, model, optimizer, omniglot_dataset, dev)