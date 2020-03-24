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
import utils.data as data_loader
from models.lenet import LeNet
from methods.ewc import Fisher_KFAC_reg, Fisher_KFAC_reg_id, Fisher_EKFAC_reg, Fisher_KFAC_reg_uncertainty
from utils.utils import Edl_mse_loss, Edl_digamma_loss, relu_evidence, Edl_log_loss

import os
import numpy as np
from utils.clb.utils.metrics import accuracy, AverageMeter

parser = argparse.ArgumentParser()
parser.add_argument('--lamda', default=5, type=int,
                    help='lamda')
parser.add_argument('--annealing_step', default=5, type=int,
                    help='annealing_step')
prs = parser.parse_args()
args = easydict.EasyDict({
    "input_size":784,
    "dataset": "perm-mnist",
    "method": 'kfac',
    "multi_head": True,
    "comment":'',
    'hidden_units':50,
    "n_epochs":21,
    "task_number":5,
    "batch_size_train": 120,
    "batch_size_test":120,
    "samples_no_f": 0,
    "learning_rate": 0.001,
    "momentum": 0.5,
    "log_interval": 100,
    "num_ways": 10,
    #"class_per_task": 2,
    "augmentations":None,
    "omniglot_size":(28, 28),
    "seed": 100,
    "kfac_batch":True,
    "log_interval_epoch":2,
    "device": 'cuda',
    "lamda":prs.lamda,
    "annealing_step":prs.annealing_step,
    "channel_size":64,
    #"beta": prs.beta,
    #"beta_factor": .9,
    #"max_epoch_id":100,
    #"beta_min":prs.beta*0.1,
    #"dropout":prs.dropout,
    #"noise_val":prs.noise_val,
    "wandb":'multi_head',
    'log':True,
    "wandb_key":None,
    "name":None,
    #"noisy_fisher": prs.noisy_fisher,
    #"activation": prs.activation,
    #"learnable_prior": prs.learnprior,
    #"sport_period": prs.sport_period
})

if args.wandb is not None:
    #if not is_connected():
    #    print('no internet connection. Going in dry')
    #    os.environ['WANDB_MODE'] = 'dryrun'
    #import wandb

    # wandb.init(project="cl_harmonics_hparam_search")

    if args.wandb_key is not None:
        wandb.login(key=args.wandb_key)
    if args.name is None:
        wandb.init(project=args.wandb)
    else:
        wandb.init(project=args.wandb, name=args.name)
    wandb.config.update(args, allow_val_change=True)
else:
    wandb = None



if wandb is not None:
    wandb.config.update(args)

args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


def test(model, global_step, current_task_number, device, split="test", test_loaders=None):
    model.eval()
    test_loss = 0
    correct = 0
    correct_tl = 0
    task_pred_acc_acc = 0
    all_data_size = 0
    accs = {}
    with torch.no_grad():
        for task in range(current_task_number + 1):
            if test_loaders is None:
                _, test_loader, _ = data_loader.get_new_dataset(num_workers=0, dataset_number=task,
                                                                permutation=past_permutations[task])
            else:
                test_loader = torch.utils.data.DataLoader(test_loaders[task], batch_size=args.batch_size_test)

            correct_task = 0
            correct_task_tl = 0
            task_correct = 0
            n_batches = 0
            for data, target in test_loader:
                n_batches+=1
                output = model(data.to(device), current_task_number=current_task_number + 1)  # .view(len(data), -1)
                min_uncertainty = (np.inf, None)
                for i, out in enumerate(output):
                    preds = torch.max(out, dim=1)
                    match = torch.reshape(torch.eq(
                        preds[1], target).float(), (-1, 1))
                    evidence = relu_evidence(out)
                    alpha = evidence + 1
                    u = args.num_ways / torch.sum(alpha, dim=1, keepdim=True)
                    mean_uncertainty = torch.mean(u)

                    if mean_uncertainty < min_uncertainty[0]:
                        min_uncertainty = (mean_uncertainty, i)

                    total_evidence = torch.sum(evidence, 1, keepdim=True)
                    mean_evidence = torch.mean(total_evidence)
                    mean_evidence_succ = torch.sum(
                        torch.sum(evidence, 1, keepdim=True) * match) / torch.sum(match + 1e-20)
                    mean_evidence_fail = torch.sum(
                        torch.sum(evidence, 1, keepdim=True) * (1 - match)) / (torch.sum(torch.abs(1 - match)) + 1e-20)
                    if args.log and not wandb is None:
                        wandb.log({
                            "mean_evidence_fail_test_task_" + str(task) + "out_" + str(
                                i): mean_evidence_fail.detach().cpu().numpy(), 'global_step': global_step})
                        wandb.log({
                            "uncertainty_test_task_" + str(task) + "out_" + str(i): u.detach().cpu().numpy(),
                            'global_step': global_step})
                        wandb.log({
                            "mean_uncertainty_test_task_" + str(task) + "out_" + str(
                                i): mean_uncertainty.detach().cpu().numpy(),
                            'global_step': global_step})
                        wandb.log({
                            "mean_evidence_succ_test_task_" + str(task) + "out_" + str(
                                i): mean_evidence_succ.detach().cpu().numpy(), 'global_step': global_step})
                        wandb.log({
                            "mean_evidence_test_task_" + str(task) + "out_" + str(
                                i): mean_evidence.detach().cpu().numpy(), 'global_step': global_step})
                        wandb.log({
                            "total_evidence_test_task_" + str(task) + "out_" + str(
                                i): total_evidence.detach().cpu().numpy(), 'global_step': global_step})
                # print(min_uncertainty)

                task_correct += int(min_uncertainty[1]==task)
                pred = F.log_softmax(output[min_uncertainty[1]], dim=1).argmax(dim=1,
                                                                               keepdim=True)  # get the index of the max log-probability
                pred_tl = F.log_softmax(output[task], dim=1).argmax(dim=1,
                                                                               keepdim=True)
                correct_ = pred.eq(target.view_as(pred).to(device)).sum().item()
                correct_task += correct_
                correct += correct_

                correct_tl_ = pred_tl.eq(target.view_as(pred_tl).to(device)).sum().item()
                correct_task_tl += correct_tl_
                correct_tl += correct_tl_

                all_data_size += len(data)
            test_acc_task = 100. * correct_task / len(test_loader.dataset)
            test_acc_task_tl = 100. * correct_task_tl / len(test_loader.dataset)
            task_pred_acc_task = 100. * task_correct / n_batches
            task_pred_acc_acc += task_pred_acc_task
            print('Task {:.0f}, {} Accuracy: {}/{} ({:.2f}%)\n'.format(task, split, correct, len(test_loader.dataset),
                                                                       test_acc_task))
            # if writer is not None:
            #    writer.add_scalars("scalars/test_accs_task_"+method, {"task_"+str(task):test_acc_task},global_step)

            if args.log:
                accs["Task_test_acc_ " + str(task)] = test_acc_task
                accs["Task_test_acc_tl_ " + str(task)] = test_acc_task_tl
                accs["Task_pred_acc_task_ " + str(task)] = task_pred_acc_task
                accs['global_step'] = global_step
    if args.log:
        wandb.log(accs)  # , step=global_step)
    acc = 100. * correct / all_data_size
    acc_tl = 100. * correct_tl / all_data_size
    task_pred_acc_acc = task_pred_acc_acc/(current_task_number+1)
    return acc, acc_tl, task_pred_acc_acc


def train(name, model, optimizer, device="cpu", datasets=None, log_dir=None):
    acc = AverageMeter()
    global_step = 0
    if args.log:
        wandb.watch(model.model, log="all")
    test_accs = []
    for n_datasets in range(args.task_number):
        # beta = args.beta
        if datasets is None:
            train_loader, _, permutation = data_loader.get_new_dataset(
                random_seed=args.seed + (n_datasets * 100), batch_size_train=args.batch_size_train,
                batch_size_test=args.batch_size_test, num_workers=0, dataset_number=n_datasets,
                permutation=past_permutations[n_datasets])
            past_permutations[n_datasets] = permutation
        else:
            train_loader = torch.utils.data.DataLoader(datasets[0][n_datasets], batch_size=args.batch_size_train)
        model.train()
        print("Task: ", str(n_datasets))
        # standart training
        ####
        max_acc_task = 0
        ####
        for epoch in range(args.n_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                model.train()
                optimizer.zero_grad()
                model.zero_grad()
                output = model(data.to(device), task=n_datasets)
                reg = 0
                # if name=="kfac_id":
                #    reg = beta * torch.mean(output[1]) / len(data) #.kl) / len(data)
                #    output = output[0] #.activations

                loss = model.loss(output, target.to(device), epoch=epoch, num_classes=args.num_ways, annealing_step=args.annealing_step) + reg
                preds = torch.max(output, dim=1)
                match = torch.reshape(torch.eq(
                    preds[1], target).float(), (-1, 1))
                evidence = relu_evidence(output)
                alpha = evidence + 1
                u = args.num_ways / torch.sum(alpha, dim=1, keepdim=True)
                mean_uncertainty = torch.mean(u)
                total_evidence = torch.sum(evidence, 1, keepdim=True)
                mean_evidence = torch.mean(total_evidence)
                mean_evidence_succ = torch.sum(
                    torch.sum(evidence, 1, keepdim=True) * match) / torch.sum(match + 1e-20)
                mean_evidence_fail = torch.sum(
                    torch.sum(evidence, 1, keepdim=True) * (1 - match)) / (torch.sum(torch.abs(1 - match)) + 1e-20)
                if args.log and not wandb is None:
                    wandb.log({
                        "mean_evidence_fail": mean_evidence_fail.detach().cpu().numpy(), 'global_step': global_step})
                    wandb.log({
                        "uncertainty": u.detach().cpu().numpy(),
                        'global_step': global_step})
                    wandb.log({
                        "mean_uncertainty": mean_uncertainty.detach().cpu().numpy(),
                        'global_step': global_step})
                    wandb.log({
                        "mean_evidence_succ": mean_evidence_succ.detach().cpu().numpy(), 'global_step': global_step})
                    wandb.log({
                        "mean_evidence": mean_evidence.detach().cpu().numpy(), 'global_step': global_step})
                    wandb.log({
                        "total_evidence": total_evidence.detach().cpu().numpy(), 'global_step': global_step})

                # if name=="gn":
                #    with backpack(KFAC()):
                #        loss.backward()
                # else:
                loss.backward()
                optimizer.step()
                acc = accumulate_acc(F.log_softmax(output, dim=1), target, 0, acc)
                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tacc.tr: {:.2f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.item(), acc.val))

            global_step = args.n_epochs * n_datasets + epoch
            if epoch % args.log_interval_epoch == 0:
                test_acc, test_acc_tl, task_pred_acc_acc = test(model, global_step,n_datasets, device, test_loaders= datasets[1] if not datasets is None else None)


                if test_acc > max_acc_task:
                    max_acc_task = test_acc
                print('Test task {:.0f}, Epoch: {}, Average Acc. {:.2f}%'.format(n_datasets, epoch, test_acc))
                print('Test task tl {:.0f}, Epoch: {}, Average Acc. {:.2f}%'.format(n_datasets, epoch, test_acc_tl))
                # if name=="kfac":
                #    best_val_acc_kfac[n_datasets]=max(best_val_acc_kfac[n_datasets],test_acc)
                ### loging
                # if writer is not None:
                if args.log:
                    wandb.log({
                        "avv_test_acc_sofar": test_acc, 'global_step': global_step})  # , step=global_step)
                    wandb.log({
                        "avv_test_acc_sofar_tl": test_acc_tl, 'global_step': global_step})  # , step=global_step)
                    wandb.log({
                        "avv_task_pred_acc": task_pred_acc_acc, 'global_step': global_step})  # , step=global_step)
                    # writer.add_scalars("scalars/avv_test_acc_sofar", {model.__class__.__name__+name:test_acc},global_step)
                test_accs.append((test_acc, n_datasets, epoch))
            # if name == "kfac_id":
            #    print(beta)
            #    if beta > args.beta_min:
            #        beta *= args.beta_factor
            #    else:
            #        beta = args.beta_min
        if args.log:
            wandb.log({
                "max_acc_task": max_acc_task})  # , step=n_datasets)
        model.on_task_switch(train_loader=train_loader, device=device, samples_no=args.samples_no_f, task=n_datasets)


if __name__ == '__main__':
    if args.dataset=="split-mnist":
        from utils.dataloaders import splitmnist as dataloader
        # Prepare data for chosen experiment
        (train_datasets, test_datasets), config, classes_per_task = dataloader.get_multitask_experiment(
            name='splitMNIST', scenario='domain', tasks=args.task_number, data_dir='./data',
            verbose=True, exception=True if args.seed == 0 else False,
        )

    net = LeNet(class_per_task=args.num_ways, heads_number=args.task_number)

    past_permutations = [None] * args.num_ways
    best_val_acc_kfac = [0] * args.num_ways
    #if not args.wandb is None:
    #    wandb.config.update(args)
    dev = torch.device(args.device)

    if args.method == "kfac" or args.method == "kfac_dropout":
        model = Fisher_KFAC_reg_uncertainty(args.channel_size, args.num_ways, args.hidden_units, device=dev,
                                            lamda=args.lamda,
                                            heads_number=args.num_ways, log=True, model=net)
    elif args.method == "kfac_id":
        model = Fisher_KFAC_reg_id(args.channel_size, args.num_ways, device=dev, lamda=args.lamda,
                                   heads_number=args.num_ways, log=True, activation=args.activation, model=net)
    elif args.method == "ekfac":
        model = Fisher_EKFAC_reg(args.channel_size, args.num_ways, device=dev, lamda=args.lamda,
                                 heads_number=args.num_ways, log=True, activation=args.activation, model=net)
    model = model.to(dev)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    model.criteria = Edl_mse_loss()  # Edl_mse_loss() #Edl_log_loss() #Edl_digamma_loss()
    train(args.method, model, optimizer, dev) #, datasets=(train_datasets, test_datasets))




