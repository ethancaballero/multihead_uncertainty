import copy
import time
import torch
import wandb
import atexit
import random
import easydict
import argparse
import numpy as np
import torch.optim as optim
from datetime import datetime
import torch.nn.functional as F
from torch.autograd import Variable
from utils.folder import ImageFolder
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from utils.clb.utils.metrics import accuracy, AverageMeter
from methods.ewc_conv import Fisher_KFAC_reg, Fisher_KFAC_reg_id, Fisher_EKFAC_reg


parser = argparse.ArgumentParser()
parser.add_argument('--log', action='store_true',
                    help='Log the results?')
parser.add_argument('--multi_head', action='store_true',
                    help='use several heads for the several tasks')
parser.add_argument('--device', default="cpu", type=str,
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
parser.add_argument('--method', choices=['kfac','diag','kfac_id','mlp','kfac_dropout', 'ekfac'], default="kfac", type=str,
                    help='Method to run')
parser.add_argument('--dropout', default=False, type=bool,
                    help='Use standard dropout layer')
parser.add_argument('--log_dir', default="2_o_methods", type=str)
parser.add_argument('--noise_val', action='store_true', help='Add noise at val time')
parser.add_argument('--beta', default=0.001, type=float, help='Beta')
parser.add_argument('--wandb', default='temp', type=str, help='wandb')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--noisy_fisher', default=0, type=int, help='Wether of not caluclate fisher with noise')
parser.add_argument('--activation', default='softplus', choices=['relu', 'softplus'], type=str, help='activation funciton')
parser.add_argument('--comment', default='', type=str)
parser.add_argument('--learnprior', default=1, type=int, help='Wether of not caluclate fisher with noise')
parser.add_argument('--sport_period', default=0, type=int, help='Noise periodicity')
parser.add_argument('-d','--debug', action='store_true', help='enable debug mode to go faster')
parser.add_argument('-act_bfore_bn', action='store_true', help='same order of activation as in dropout net')
parser.add_argument('--mixup', type=str, default = 'None', choices =['vanilla','mixup', 'mixup_hidden','cutout'])
parser.add_argument('--mixup_alpha', type=float, default=1.0, help='alpha parameter for mixup')


prs = parser.parse_args()

if prs.method=="kfac_id":
    prs.dropout = False
elif prs.method=="kfac_dropout":
    prs.dropout = True

log_dir = prs.log_dir
dataroot = '/network/data1/ImageNet2012_jpeg/'

args = easydict.EasyDict({
    "dataset": "tiered_imnet",
    "method": prs.method,
    "log_dir":log_dir,
    "multi_head": prs.multi_head,
    "comment":"_ntasks_"+str(prs.n_tasks)+"_"+prs.method+" "+prs.activation+" "+prs.comment,
    "n_epochs": prs.n_epochs,
    "batch_size": prs.batch_size,
    "samples_no_f": prs.samples_no_f,
    "learning_rate": prs.lr,
    "momentum": 0.5,
    "log_interval": 10,
    "task_number": prs.n_tasks,
    "class_per_task": prs.class_per_task,
    "imageSize":64,
    "seed": 100,
    "kfac_batch":True,
    "log":prs.log,
    "log_interval_epoch":5,
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
    "sport_period": prs.sport_period,
    "workers":1,
    "dataroot": dataroot,
    "act_bfore_bn": prs.act_bfore_bn,
    "mixup": prs.mixup,
    "mixup_alpha":prs.mixup_alpha
    #"folder":'/Users/oleksostapenko/Projects/data/tiered-imagenet/tiered-imagenet' #'/network/home/ostapeno/data'
})
if args.log:
    ts = time.time()
    # wandb.init(sync_tensorboard=True, project="2o_methods", name=datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S') + args.comment)
    wandb.init(project=args.wandb, name=datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S') + args.comment)
print(args)
best_val_acc_kfac=[0]*args.task_number


#if torch.cuda.is_available() and args.device!="cpu":
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    torch.set_default_tensor_type('torch.cuda.FloatTensor')

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


bce_loss = torch.nn.BCELoss().cuda()
softmax = torch.nn.Softmax(dim=1).cuda()
log_softmax = torch.nn.LogSoftmax(dim=1).cuda()
def accumulate_acc(output, target, task, meter):
    meter.update(accuracy(output, target), len(target))
    return meter

def eval(model, past_idx, cur_task_id, global_step, device, split="val", noise_val=False, act_last=log_softmax):
    model.eval()
    test_loss = 0
    correct = 0
    all_data_size = 0
    accs = {}
    with torch.no_grad():
        print(past_idx)
        for task_id, task_idx in enumerate(past_idx):
            dataset = ImageFolder(
                root=args.dataroot+split+'/',
                transform=transforms.Compose([
                    transforms.Resize((args.imageSize,args.imageSize)),
                    #transforms.CenterCrop(args.imageSize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]), #(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
                classes_idx=(task_idx),
            )
            valid_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                          shuffle=True, num_workers=int(1))
            correct_task=0
            for data, target in valid_loader:
                if args.multi_head:
                    output = model(data.to(device), task=task_id, noise=noise_val)
                else:
                    output = model(data.to(device))
                if isinstance(output, list):
                    output=output[0]
                output = act_last(output)
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
            accs["global_step"]=global_step #model.loging_step
    if args.log:
        wandb.log(accs)
    acc = 100. * correct / all_data_size
    return acc

def train(name, model, device="cpu"):
    valid_accs = []
    past_idx =[]
    cls_idxs = list(range(1000))
    for cur_task_id in range(args.task_number):
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        #scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        idx_ = random.sample(cls_idxs,k=args.class_per_task)
        past_idx.append(idx_)
        cls_idxs = [e for e in cls_idxs if e not in idx_]
        print(past_idx)
        dataset = ImageFolder(
            root=args.dataroot+'train/',
            transform=transforms.Compose([
                transforms.Resize((args.imageSize,args.imageSize)),
                #transforms.CenterCrop(args.imageSize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]),
            classes_idx=(idx_)
        )
        train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(
            args.workers))

        beta = args.beta
        model.train()
        print("Task: ", cur_task_id, " current task idx", idx_)
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
            #print("Noise ",add_noise)
            act_last = log_softmax
            for batch_idx, (data, target) in enumerate(train_loader):
                    model.train()
                    optimizer.zero_grad()
                    model.zero_grad()
                    data, target = data.to(device), target.to(device)
                    #print(data.shape)

                    if args.mixup == 'mixup':
                        input_var, target_var = Variable(data), Variable(target)
                        if args.multi_head:
                            output,reweighted_target = model(input_var, target_var, mixup=True,
                                                          mixup_alpha=args.mixup_alpha, task=cur_task_id, noise=add_noise)
                        else:
                            output,reweighted_target = model(input_var, target_var, mixup=True,
                                                          mixup_alpha=args.mixup_alpha, noise=add_noise)

                        loss = bce_loss(softmax(output), reweighted_target)  # mixup_criterion(target_a, target_b, lam)
                        reg = 0
                        if name == "kfac_id":
                            reg = beta * torch.mean(output[1]) / len(data)  # .kl) / len(data)
                            output = output[0]  # .activations
                        loss += reg
                        act_last = softmax

                        """
                        mixed_input, target_a, target_b, lam = mixup_data(input, target, args.mixup_alpha)
                        input_var, mixed_input_var, target_var, target_a_var, target_b_var = Variable(input),Variable(mixed_input), Variable(target), Variable(target_a), Variable(target_b)

                        mixed_output = model(mixed_input_var)
                        output = model(input_var)

                        loss_func = mixup_criterion(target_a_var, target_b_var, lam)
                        loss = loss_func(criterion, mixed_output)
                        """

                    elif args.mixup == 'mixup_hidden':
                        input_var, target_var = Variable(data), Variable(target)
                        if args.multi_head:
                            output, reweighted_target = model(input_var, target_var, mixup_hidden=True,
                                                          mixup_alpha=args.mixup_alpha, task=cur_task_id, noise=add_noise)
                        else:
                            output, reweighted_target = model(input_var, target_var, mixup_hidden=True,
                                                          mixup_alpha=args.mixup_alpha, noise=add_noise)
                        #output, reweighted_target = model(input_var, target_var, mixup_hidden=True,
                        #                                  mixup_alpha=args.mixup_alpha)
                        #print(output)
                        #print(reweighted_target)
                        loss = bce_loss(softmax(output), reweighted_target)  # mixup_criterion(target_a, target_b, lam)
                        reg = 0
                        if name == "kfac_id":
                            reg = beta * torch.mean(output[1]) / len(data)  # .kl) / len(data)
                            output = output[0]  # .activations
                        loss +=reg
                        act_last = softmax
                        """
                        input_var, target_var = Variable(input), Variable(target)
                        mixed_output, target_a, target_b, lam = model(input_var, target_var, mixup_hidden = True,  mixup_alpha = args.mixup_alpha)
                        output = model(input_var)

                        lam = lam[0]
                        target_a_one_hot = to_one_hot(target_a, args.num_classes)
                        target_b_one_hot = to_one_hot(target_b, args.num_classes)
                        mixed_target = target_a_one_hot * lam + target_b_one_hot * (1 - lam)
                        loss = bce_loss(softmax(output), mixed_target)
                        """
                    elif args.mixup == 'cutout':
                        cutout = Cutout(1, args.cutout)
                        cut_input = cutout.apply(data)

                        input_var = torch.autograd.Variable(data)
                        target_var = torch.autograd.Variable(target)
                        cut_input_var = torch.autograd.Variable(cut_input)
                        # if dataname== 'mnist':
                        #    input = input.view(-1, 784)
                        if args.multi_head:
                            output = model(cut_input_var, target_var, noise=add_noise)
                        else:
                            output = model(cut_input_var, target_var, noise=add_noise)
                        #output, reweighted_target = model(cut_input_var, target_var)
                        # loss = criterion(output, target_var)
                        loss = bce_loss(softmax(output), reweighted_target)
                    else:
                        if args.multi_head:
                            output = model(data, task=cur_task_id, noise=add_noise)
                        else:
                            output = model(data, noise=add_noise)
                        reg = 0
                        if name=="kfac_id":
                            reg = beta * torch.mean(output[1]) / len(data) #.kl) / len(data)
                            output = output[0] #.activations
                        loss = model.loss(output, target.to(device)) + reg
                        act_last = log_softmax
                    loss.backward()
                    optimizer.step()
                    #scheduler.step()
                    acc = accumulate_acc(act_last(output), target, 0, acc)
                    if batch_idx % args.log_interval == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tacc.tr: {:.2f}'.format(
                            epoch, batch_idx * len(data), len(train_loader.dataset),
                                   100. * batch_idx / len(train_loader), loss.item(), acc.val))

            if args.mixup is None:
                global_step = args.n_epochs*cur_task_id+epoch
            else:
                if epoch % int(args.n_epochs/10)==0:
                    global_step = args.n_epochs * cur_task_id + epoch
            if epoch % args.log_interval_epoch==0:
                valid_acc = eval(model, past_idx, cur_task_id, global_step, device, act_last=act_last)
                if valid_acc>max_acc_task:
                    max_acc_task=valid_acc
                print('Validation task {:.0f}, Epoch: {}, Average Acc. {:.2f}%'.format(cur_task_id, epoch, valid_acc))
                valid_acc_noisy = eval(model, past_idx, cur_task_id, global_step,device, noise_val=True, act_last=act_last)
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
        #train_loader.dataset.empty_cache()

if __name__ == '__main__':
    ts = time.time()
    log_dir=None
    heads_number = args.task_number if args.multi_head else 1
    if args.log:
        wandb.config.update(args)
    dev = torch.device(args.device)
    if args.method=="kfac" or args.method=="kfac_dropout":
        from models.preresnet import preactresnet18, preactresnet34, preactresnet50
        net = preactresnet34(args.class_per_task, False, False, 2,
                             activation=torch.nn.Softplus(),heads_number=heads_number, act_bfore_bn=args.act_bfore_bn)  # .cuda() #args.channel_size, args.class_per_task, dropout=args.dropout, heads_number=args.task_number,

        # activation=args.activation, learnable_prior=args.learnable_prior)
        model = Fisher_KFAC_reg(args.channel_size, args.class_per_task, device=dev, lamda=args.lamda, dropout=args.dropout,
                                heads_number=args.task_number, log=args.log, activation=args.activation, model=net)
    elif args.method == "kfac_id":
        from models.preresnet_inf_drop import preactresnet18, preactresnet34, preactresnet50
        net = preactresnet34(args.class_per_task, False, False, 2,
                             activation=torch.nn.Softplus(),heads_number=heads_number)  # .cuda() #args.channel_size, args.class_per_task, dropout=args.dropout, heads_number=args.task_number,
        # activation=args.activation, learnable_prior=args.learnable_prior)
        model = Fisher_KFAC_reg_id(args.channel_size, args.class_per_task, device=dev, lamda=args.lamda,
                                   heads_number=args.task_number, log=args.log, activation=args.activation, model=net)
    elif args.method == "ekfac":
        model = Fisher_EKFAC_reg(args.channel_size, args.class_per_task, device=dev, lamda=args.lamda,
                                 heads_number=args.task_number, log=args.log, activation=args.activation, model=net)
    model = model.to(dev)

    torch.autograd.set_detect_anomaly(True)
    train(args.method, model, dev)