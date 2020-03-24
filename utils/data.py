import torch
from torchvision import datasets, transforms

import os
import numpy as np
try: 
    copressed_data_folder = os.environ['SCRATCH']+'/mnist'
except:
    #if os.path.isdir('/network/data1/'):
    #    copressed_data_folder = "/network/data1/mnist"
    #else:
    copressed_data_folder = "./.data"#os.environ['SLURM_TMPDIR']+ '/mnist'


class Permute(object):
    def __init__(self,permutation):
        self.permutation=permutation
    def __call__(self, sample):
        return {'image': sample.view(-1)[self.permutation].view(1, 28, 28), 'landmarks': None}
def permute(x,permutation):
    return x.view(-1)[permutation].view(1, 28, 28)



def _get_dataset_MNIST(permutation, get_train=True, batch_size=64, root=copressed_data_folder, num_workers=0):
    if permutation is not None:
        trans_perm = transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x: x.view(-1)[permutation].view(1, 28, 28))])# ,Permute(permutation)])
    else:
        trans_perm = transforms.Compose([transforms.ToTensor()])

    dataset = datasets.MNIST(root=root, train=get_train, transform=trans_perm, download=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader

def get_new_dataset(dataset_name="pMNIST", random_seed=42, batch_size_train=64, batch_size_test=100, num_workers=1, dataset_number=0, permutation=None):
    if dataset_name == "pMNIST":
        np.random.seed(random_seed)
        root = copressed_data_folder
        if not os.path.exists(root):
            os.mkdir(root)

        if dataset_number==0:
            train_set = _get_dataset_MNIST(None, True, batch_size_train, root, num_workers)
            test_set = _get_dataset_MNIST(None, False, batch_size_test, root, num_workers)
            return (train_set, test_set, permutation)
        else:
            if permutation is None:
                permutation = np.random.permutation(28 * 28)
            train_set = _get_dataset_MNIST(permutation, True, batch_size_train, root, num_workers)
            test_set = _get_dataset_MNIST(permutation, False, batch_size_test, root,num_workers)
            return (train_set, test_set, permutation)



def get_datasets(dataset_name="pMNIST", random_seed=42, task_number=10,
                 batch_size_train=64, batch_size_test=100,num_workers=1):
    
    if dataset_name == "pMNIST":
        
        root = copressed_data_folder
        if not os.path.exists(root):
            os.mkdir(root)

        np.random.seed(random_seed)
        permutations = [
            np.random.permutation(28 * 28) for
            _ in range(task_number)
        ]

        train_datasets = [
            _get_dataset_MNIST(p, True, batch_size_train, root,num_workers) for p in permutations
        ]
        test_datasets = [
            _get_dataset_MNIST(p, False, batch_size_test, root,num_workers) for p in permutations
        ]
    return train_datasets, test_datasets