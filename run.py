
import copy
import time
import torch
import wandb
import atexit
import random
import easydict
import argparse
import utils.data as data_loader
from datetime import datetime
#from methods.ewc import Regularized_model
from methods.ewc import Fisher_KFAC_reg, Fisher_KFAC_reg_id, Fisher_EKFAC_reg
#from methods.ewc_backpack import Regularized_model_gn
#from backpack import extend, backpack, KFAC
#from methods.ewc_backpack import Regularized_model_gn
#from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim


import os
import numpy as np
from utils.clb.utils.metrics import accuracy, AverageMeter
#import matplotlib.pyplot as plt
#from tqdm import tqdm
#%matplotlib inline


parser = argparse.ArgumentParser()
parser.add_argument('--log', action='store_true',
                    help='Log the results?')
parser.add_argument('--device', default="cuda", type=str,
                    help='device cuda or cpu')
parser.add_argument('--n_epochs', default=2, type=int,
                    help='depochs number per task')
parser.add_argument('--samples_no_f', default=None, type=int,
                    help='n samples fisher')
parser.add_argument('--lamda', default=100, type=int,
                    help='lamda')
parser.add_argument('--hu', default=100, type=int,
                    help='hidden_units')
parser.add_argument('--n_tsks', default=50, type=int,
                    help='n_tsks')
parser.add_argument('--method', choices=['kfac','diag','kfac_id','mlp','kfac_dropout', 'ekfac'], default="kfac", type=str,
                    help='Method to run')
parser.add_argument('--dropout', default=False, type=bool,
                    help='Use standard dropout layer')
parser.add_argument('--log_dir', default="2_o_methods", type=str)
prs = parser.parse_args()

if prs.method=="kfac_id":
    prs.dropout = False
elif prs.method=="kfac_dropout":
    prs.dropout = True

log_dir = prs.log_dir
# if os.environ.get('SCRATCH') is not None:
#     log_dir = os.environ['SCRATCH'] + '/2_o_methods/'
# elif os.environ.get('/network/tmp1/ostapeno/') is not None:
#     log_dir = os.environ['/network/tmp1/ostapeno/'] + '/2_o_methods/'
# else:
#     log_dir = '2_o_methods/'

args = easydict.EasyDict({
    "method": prs.method,
    "log_dir":log_dir,
    "comment":"_ntasks_"+str(prs.n_tsks)+"_hu_"+str(prs.hu,)+"_"+prs.method+"_softplus",
    "n_epochs": prs.n_epochs,
    "batch_size_train": 64,
    "samples_no_f": prs.samples_no_f,
    "batch_size_test": 64,
    "learning_rate": 0.001,
    "momentum": 0.5,
    "log_interval": 100,
    "task_number": prs.n_tsks,
    "mnist_size":(28, 28),
    "seed": 100,
    "kfac_batch":True,
    "log":prs.log,
    "log_interval_epoch":5,
    "device": prs.device,
    "lamda":prs.lamda,
    "hidden_units":prs.hu,
    "beta": 0.001,
    "beta_factor": .9,
    "max_epoch_id":100,
    "beta_min":0.0001,
    "dropout":prs.dropout
})
if args.log:
    ts = time.time()
    # wandb.init(sync_tensorboard=True, project="2o_methods", name=datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S') + args.comment)
    wandb.init(project="2o_methods", name=datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S') + args.comment)
print(args)
past_permutations=[None]*args.task_number
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



#train_datasets, test_datasets = data.get_datasets(random_seed=args.seed,
#                                                  task_number=args.task_number,
#                                                  batch_size_train=args.batch_size_train,
#                                                  batch_size_test=args.batch_size_test,
#                                                  num_workers=0
#                                                  )

def accumulate_acc(output, target, task, meter):
    meter.update(accuracy(output, target), len(target))
    return meter


def test(model, test_loader_current, writer, method, global_step, current_task_number, device, split="test"):
    model.eval()
    test_loss = 0
    correct = 0
    all_data_size = 0
    accs = {}
    with torch.no_grad():
        for task in range(current_task_number+1):
            if task == current_task_number:
                test_loader=test_loader_current
            else:
                _,test_loader, _ = data_loader.get_new_dataset(num_workers=0, dataset_number=task, permutation=past_permutations[task])
            correct_task=0
            for data, target in test_loader:
                output = F.log_softmax(model(data.view(len(data), -1).to(device)),dim=1)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct_ = pred.eq(target.view_as(pred).to(device)).sum().item()
                correct_task+=correct_
                correct+=correct_
                all_data_size += len(data)
            test_acc_task = 100. * correct_task / len(test_loader.dataset)
            print('Task {:.0f}, {} Accuracy: {}/{} ({:.2f}%)\n'.format(task, split, correct, len(test_loader.dataset),
                                                                 test_acc_task))
            #if writer is not None:
            #    writer.add_scalars("scalars/test_accs_task_"+method, {"task_"+str(task):test_acc_task},global_step)
            if args.log:
                accs["Task_test_acc_ "+str(task)] = test_acc_task
    if args.log:
        wandb.log(accs, step=global_step)
    acc = 100. * correct / all_data_size
    return acc

def train(name, model, optimizer, device="cpu", log_dir=None):
    acc = AverageMeter()
    writer=None
    if args.log:
        wandb.watch(model.model,log="all")
    #if device!="cpu":
    #    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    #torch.cuda.set_device(device)
    test_accs = []
    #writer=None
    #if args.log:
    #    writer = SummaryWriter(log_dir, comment=args.comment)
    for n_datasets in range(args.task_number):
        beta = args.beta
        train_loader, test_loader_current, permutation = data_loader.get_new_dataset(random_seed=args.seed+(n_datasets*100),batch_size_train=args.batch_size_train,batch_size_test=args.batch_size_test, num_workers=0, dataset_number=n_datasets, permutation=past_permutations[n_datasets])
        past_permutations[n_datasets] = permutation
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
                output = model(data.to(device))
                reg = 0
                if name=="kfac_id":
                    reg = beta * torch.mean(output[1]) / len(data) #.kl) / len(data)
                    output = output[0] #.activations
                loss = model.loss(output, target.to(device)) + reg
                #if name=="gn":
                #    with backpack(KFAC()):
                #        loss.backward()
                #else:
                loss.backward()
                optimizer.step()
                acc = accumulate_acc(F.log_softmax(output, dim=1), target, 0, acc)
                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tacc.tr: {:.2f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.item(), acc.val))

            global_step = args.n_epochs*n_datasets+epoch
            if epoch % args.log_interval_epoch==0:
                test_acc = test(model, test_loader_current, writer, model.__class__.__name__+name, global_step, n_datasets, device)
                if test_acc>max_acc_task:
                    max_acc_task=test_acc
                print('Test task {:.0f}, Epoch: {}, Average Acc. {:.2f}%'.format(n_datasets, epoch, test_acc))
                #if name=="kfac":
                #    best_val_acc_kfac[n_datasets]=max(best_val_acc_kfac[n_datasets],test_acc)
                ### loging
                #if writer is not None:
                if args.log:
                    wandb.log({
                        "avv_test_acc_sofar": test_acc}, step=global_step)
                    #writer.add_scalars("scalars/avv_test_acc_sofar", {model.__class__.__name__+name:test_acc},global_step)
                test_accs.append((test_acc,n_datasets,epoch))
            if name == "kfac_id":
                print(beta)
                if beta > args.beta_min:
                    beta *= args.beta_factor
                else:
                    beta = args.beta_min
        if args.log:
            wandb.log({
                "max_acc_task": max_acc_task}) #, step=n_datasets)
        model.on_task_switch(train_loader=train_loader, device=device, samples_no=args.samples_no_f, task=n_datasets)


if __name__ == '__main__':
    ts = time.time()
    log_dir=None
    if args.log:
        #log_dir = args.log_dir + datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S') + args.comment
        #writer = SummaryWriter(log_dir, comment=args.comment)
        #writer.add_text(text_string=str(args), tag='arguments')
        wandb.config.update(args)

    dev = dev1 = dev2 = dev3 = torch.device(args.device)
    #gn_model = Regularized_model_gn(28 * 28, 10, args.hidden_units, device=dev, lamda=args.lamda, bias=True)
    #bldiag_model = Regularized_model(28 * 28, 10, args.hidden_units, device=dev, lamda=args.lamda, bias=True, reg_matrix="block_diag")
    #ewc_model = Regularized_model(28 * 28, 10, args.hidden_units, device=dev, lamda=args.lamda, bias=True, reg_matrix="diag")
    if args.method=="kfac" or args.method=="kfac_dropout":
        model = Fisher_KFAC_reg(28 * 28, 10, args.hidden_units, device=dev, lamda=args.lamda, bias=True, dropout=args.dropout)
    elif args.method == "kfac_id":
        model = Fisher_KFAC_reg_id(28 * 28, 10, args.hidden_units, device=dev, lamda=args.lamda, bias=True)
    elif args.method == "ekfac":
        model = Fisher_EKFAC_reg(28 * 28, 10, args.hidden_units, device=dev, lamda=args.lamda, bias=True)


    #ekfa_model = Regularized_model(28 * 28, 10, args.hidden_units, device=dev, lamda=args.lamda, bias=True, reg_matrix="ekfac")
    #ewc_model_shuffle = Regularized_model(28 * 28, 10, args.hidden_units, device=dev, lamda=args.lamda, bias=True, reg_matrix="diag", shuffle=True)
    #kfa_model_shuffle = Regularized_model(28 * 28, 10, args.hidden_units, device=dev, lamda=args.lamda, bias=True, reg_matrix="kfac", shuffle=True)
    #gn_model = extend(Regularized_model_gn(28 * 28, 10, 100, device=dev, lamda=args.lamda, bias=True))


    #optimizer_gn = optim.Adam(gn_model.layers.parameters(), lr=args.learning_rate)

    #optimizer_block_diag = optim.Adam(bldiag_model.parameters(), lr=args.learning_rate)
    #optimizer_ewc = optim.Adam(ewc_model.parameters(), lr=args.learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    #optimizer_kfac_id = optim.Adam(kfa_model_id.parameters(), lr=args.learning_rate)

    #optimizer_ekfa = optim.Adam(ekfa_model.parameters(), lr=args.learning_rate)
    #optimizer_ewc_shuffle = optim.Adam(ewc_model_shuffle.parameters(), lr=args.learning_rate)  # momentum=momentum)
    #optimizer_kfa_shuffle = optim.Adam(kfa_model_shuffle.parameters(), lr=args.learning_rate)
    #optimizer_gn = optim.Adam(kfa_model.parameters(), lr=args.learning_rate)
    #model_list = {"kfac_id":[kfa_model_id,optimizer_kfac_id, dev],"kfac":[kfa_model,optimizer_kfac, dev]}#,"ewc":[ewc_model,optimizer_ewc, dev]} #, "gn": [gn_model, optimizer_gn, dev]}#,"ewc_shuffle":[ewc_model_shuffle, optimizer_ewc_shuffle, dev]}#,"kfa_gn_shuffle":[kfa_model_shuffle,optimizer_kfa_shuffle, dev]}#,"kfa_fi":[kfa_fi_model,optimizer_kfa_fi, dev],"ewc":[ewc_model,optimizer_ewc, dev],"mlp":[mlp_model,optimizer_mlp, dev]} #"kfa":[kfa_model,optimizer_kfa, dev1],"ewc":[ewc_model,optimizer_ewc, dev2], "mlp":[mlp_model,optimizer_mlp, dev3]}
    #"gn": [gn_model, optimizer_gn, dev]}  #

    #"gn": [gn_model, optimizer_gn, dev],
    #criterias = [Kfa_loss(),nn.CrossEntropyLoss(),nn.CrossEntropyLoss()]
    #train(None, ewc_model, optimizer_ewc, dev3, log_dir)
    #train(None, kfa_gn_model, optimizer_kfa_gn, dev3, log_dir)

    #for key, (model, optimizer, device) in model_list.items():
        #p = mp.Process(target=train, args=(q,model,optimizer, device, log_dir))
    train(args.method,model,optimizer, dev, log_dir)

    #processes = []
    #def cleanup():
    #    for p in processes:
    #        p.terminate() #kill()  # supported from python 2.6
    #    print('cleaned up!')

    #atexit.register(cleanup)
    #mp.set_start_method('spawn')
    #q  = mp.Queue()
    #for model, optimizer, device in model_list.values():
     #   p = mp.Process(target=train, args=(q,model,optimizer, device, log_dir))
     #   p.start()
     #   processes.append(p)
     #   p.join()
        #time.sleep(1)
    #for p in processes:
    #    p.join()
    #print(q.get())



    #plt.plot(test_acc_wl, label="KFA, lambda = 1")
    #plt.plot(test_acc_5, label="KFA, lambda = 3")
    #plt.plot(test_acc_without_loss, label="online learning")
    #plt.legend()


