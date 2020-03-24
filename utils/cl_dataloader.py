import sys
import torch
import numpy as np
from pdb import set_trace

# --------------------------------------------------------------------------
# utils
# --------------------------------------------------------------------------

def select_from_tensor(tensor, index):
    """ equivalent to tensor[index] but for batched / 2D+ tensors """

    last_dim = index.dim() - 1

    assert tensor.dim() >= index.dim()
    assert index.size()[:last_dim] == tensor.size()[:last_dim]

    # we have to make `train_idx` the same shape as train_data, or else
    # `torch.gather` complains.
    # see https://discuss.pytorch.org/t/batched-index-select/9115/5

    missing_dims = tensor.dim() - index.dim()
    index = index.view(index.size() + missing_dims * (1,))
    index = index.expand((-1,) * (index.dim() - missing_dims) + tensor.size()[(last_dim+1):])

    return torch.gather(tensor, last_dim, index)

def order_and_split(data_x, data_y):
    """ given a dataset, returns (num_classes, samples_per_class, *data_x[0].size())
        tensor where samples (and labels) are ordered and split per class """

    # sort according to the label
    out_train = [
        (x,y) for (x,y) in sorted(zip(data_x, data_y), key=lambda v : v[1]) ]

    # stack in increasing label order
    data_x, data_y = [
            torch.stack([elem[i] for elem in out_train]) for i in [0,1] ]

    # find first indices of every class
    n_classes = data_y.unique().size(0)
    idx       = [((data_y + i) % n_classes).argmax() for i in range(n_classes)]
    idx       = [0] + [x + 1 for x in sorted(idx)]

    # split into different classes
    to_chunk = [a - b for (a,b) in zip(idx[1:], idx[:-1])]
    data_x   = data_x.split(to_chunk)
    data_y   = data_y.split(to_chunk)

    # give equal amt of points for every class
    #TODO(if this is restrictive for some dataset, we can change)
    min_amt  = min([x.size(0) for x in data_x])
    data_x   = torch.stack([x[:min_amt] for x in data_x])
    data_y   = torch.stack([y[:min_amt] for y in data_y])

    # sanity check
    for i, item in enumerate(data_y):
        assert item.unique().size(0) == 1 and item[0] == i, 'wrong result'

    return data_x, data_y


# --------------------------------------------------------------------------
# Datasets and Streams (the good stuff)
# --------------------------------------------------------------------------

class MetaDataset(torch.utils.data.Dataset):
    """ Dataset similar to BatchMetaDataset in TorchMeta """

    def __init__(self, train_data, test_data,  n_shots_tr, n_shots_te, n_way,
                    args=None, **kwargs):

        '''
        Parameters
        ----------

        train_data : Array of (x,) pairs, one for each class. Contains all the
            training data that should be available at meta-training time (inner loop).
        test_data  : Array of (x,) pairs, one for each class. These are the
            same classes as in `train_data`. Used at meta-testing time (outer loop).
        n_way      : number of classes per task at meta-testing
        n_shots_tr : number of samples per classes
        n_shots_te : number of samples per classes

        '''

        # NOTE: for now assume train_data and test_data have shape
        # (n_classes, n_samples_per_task, *data_shape).

        # TODO: should torchvision transforms be passed in here ?

        # separate the classes into tasks
        n_classes   = len(train_data)

        self._len        = None
        self.n_way       = n_way
        self.kwargs      = kwargs
        self.n_classes   = n_classes
        self.n_shots_tr  = n_shots_tr
        self.n_shots_te  = n_shots_te

        if args is None:
            self.input_size  = [28,28]
            self.device      = 'cpu'
        else:
            self.input_size  = args.input_size
            self.device      = args.device
        self.all_classes = np.arange(n_classes)

        self.train_data  = train_data
        self.test_data   = test_data

        if args.dataset == 'tiered-imagenet':
            self.cpu_dset = True
        else:
            self.cpu_dset = False

    def __len__(self):
        # return the number of train / test batches that can be built
        # without sample repetition
        if self._len is None:
            n_samples = sum([x.shape[0] for x in self.train_data])
            self._len = n_samples // (self.n_way * (self.n_shots_tr + self.n_shots_te))

        return self._len


    def __getitem__(self, index):
        # NOTE: This method COMPLETELY ignores the index. This will be a problem
        # if you wish to recover a specific batch of data.

        classes_in_task = np.random.choice(self.all_classes, self.n_way, replace=False)
        train_samples_in_class = self.train_data.shape[1]
        test_samples_in_class  = self.test_data.shape[1]

        train_data = self.train_data[classes_in_task]
        test_data  = self.test_data[classes_in_task]

        # sample indices for meta train
        train_idx = torch.Tensor(self.n_way, self.n_shots_tr)
        if not(self.cpu_dset): 
            train_idx = train_idx.to(self.device)
        train_idx = train_idx.uniform_(0, train_samples_in_class).long()

        # samples indices for meta test
        test_idx = torch.Tensor(self.n_way, self.n_shots_te)
        if not(self.cpu_dset): 
            test_idx = test_idx.to(self.device)
        test_idx = test_idx.uniform_(0, test_samples_in_class).long()

        train_x = select_from_tensor(train_data, train_idx)
        test_x  = select_from_tensor(test_data,  test_idx)

        train_x = train_x.view(-1, *self.input_size)
        test_x = test_x.view(-1, *self.input_size)

        # build label tensors
        train_y = torch.arange(self.n_way).view(-1, 1).expand(-1, self.n_shots_tr)
        train_y = train_y.flatten()

        test_y  = torch.arange(self.n_way).view(-1, 1).expand(-1, self.n_shots_te)
        test_y  = test_y.flatten()

        if self.cpu_dset:
            train_x = train_x.float().to(self.device)
            train_y = train_y.to(self.device)
            test_x = test_x.float().to(self.device)
            test_y = test_y.to(self.device)

        #return train_x, train_y, test_x, test_y

        # same signature are TorchMeta
        out = {}
        out['train'], out['test'] = [train_x,train_y], [test_x, test_y]

        return out


class StreamDataset(torch.utils.data.Dataset):
    """ stream of non stationary dataset as described by Mass """

    def __init__(self, train_data, test_data, ood_data, n_shots=1,
            n_way=5, prob_statio=.8, prob_train=0.1, prob_test=0.8,
            prob_ood=0.1, args=None, **kwargs):

        '''
        Parameters
        ----------

        train_data : Array of (x,) pairs, one for each class. Contains the SAME
            classes used during (meta) training, but different samples.
        test_data  : Array of (x,) pairs, one for each class. These are DIFFERENT
            classes from the ones used during (meta) training.
        n_way      : number of classes per task at cl-test time
        n_shots    : number of samples per classes at cl-test time

        '''

        assert prob_train + prob_test + prob_ood == 1.
        if args.dataset == 'tiered-imagenet':
            self.cpu_dset = True
        else:
            self.cpu_dset = False

        self.n_shots    = n_shots
        self.n_way      = n_way

        self.modes    = ['train', 'test', 'ood']
        self.modes_id = [0, 1, 2]
        self.probs    = np.array([prob_train, prob_test, prob_ood])
        self.data     = [train_data, test_data, ood_data]
        self.p_statio = prob_statio

        if args is None:
            self.input_size  = [28,28]
            self.device      = 'cpu'
        else:
            self.input_size  = args.input_size
            self.device      = args.device

        # mode in which to start ( 0 --> 'train' )
        self._mode = 0
        self._classes_in_task = None
        self._samples_in_class = None


    def __len__(self):
        # this is a never ending stream
        return sys.maxsize


    def __getitem__(self, index):
        # NOTE: This method COMPLETELY ignores the index. This will be a problem
        # if you wish to recover a specific batch of data.

        # NOTE: using multiple workers (`num_workers > 0`) or `batch_size  > 1`
        # will have undefined behaviour. This is because unlike regular datasets
        # here the sampling process is sequential.

        if (np.random.uniform() > self.p_statio) or (self._classes_in_task is None):
            task_switch = 1

            self._mode  = np.random.choice([0,1,2], p=self.probs)
        
            mode_data = self.data[self._mode]
            n_classes = len(mode_data)
            self._samples_in_class = mode_data.size(1)

            # sample `n_way` classes
            self._classes_in_task = np.random.choice(np.arange(n_classes), self.n_way, 
                    replace=False)
        
        else:
            
            task_switch = 0
        
        mode_data = self.data[self._mode]
        data = mode_data[self._classes_in_task]

        # sample indices for meta train
        idx = torch.Tensor(self.n_way, self.n_shots)#.to(self.device)
        idx = idx.uniform_(0, self._samples_in_class).long()
        if not(self.cpu_dset):
            idx = idx.to(self.device)
        data = select_from_tensor(data, idx)

        # build label tensors
        labels = torch.arange(self.n_way).view(-1, 1).expand(-1, self.n_shots).to(self.device)

        # squeeze
        data = data.view(-1, *self.input_size)
        labels = labels.flatten()

        if self.cpu_dset:
            data = data.float().to(self.device)
            labels = labels.to(self.device)

        return data, labels, task_switch, self.modes[self._mode]


if __name__ == '__main__':
    from torchvision.datasets import MNIST, FashionMNIST, Omniglot

    """ Testing with Mnist and FashionMnist """

    # fetch MNIST
    train = MNIST('data/', train=True,  download=True)
    test  = MNIST('data/', train=False, download=True)

    train_x, train_y = order_and_split(train.train_data, train.train_labels)
    test_x, test_y   = order_and_split(test.test_data, test.train_labels)

    # fetch FashionMNIST
    ood_tr = FashionMNIST('data/', train=True,   download=True)
    ood_te = FashionMNIST('data/', train=False,  download=True)

    ood_x = torch.cat((ood_tr.train_data, ood_te.test_data))
    ood_y = torch.cat((ood_te.test_labels, ood_te.test_labels))
    ood_x, ood_y = order_and_split(ood_x, ood_y)


    # keep 2 classes for cl-test time
    split = int(.8 * train_x.size(1))
    meta_train_x, meta_train_y = train_x[:8, :split], train_y[:8, :split]
    meta_test_x,  meta_test_y  = train_x[:8, split:], train_y[:8, split:]

    cl_train_x, cl_train_y = test_x[:8], test_y[:8]
    cl_test_x,  cl_test_y  = test_x[8:], test_y[8:]

    """ 1st dataset / loader: meta learning time """
    meta_ds = MetaDataset(meta_train_x, meta_test_x, n_shots_tr=5,
            n_shots_te=10, n_way=2)

    meta_loader = torch.utils.data.DataLoader(meta_ds, batch_size=16)

    for i, batch in enumerate(meta_loader):
        print('batch {} / {}'.format(i, len(meta_loader)))

        #train_x, train_y, test_x, test_y = batch

        if i == 0:
            print('meta train x size : {}'.format(batch['train'][0].size()))
            print('meta train y size : {}'.format(batch['train'][1].size()))
            print('meta test  x size : {}'.format(batch['test'][0].size()))
            print('meta test  y size : {}'.format(batch['test'][1].size()))

        if i == 5 : break


    """ 2st dataset / loader: continual learning time """
    cl_stream = StreamDataset(cl_train_x, cl_test_x, ood_x, n_shots=2, n_way=2)
    cl_loader = torch.utils.data.DataLoader(cl_stream, batch_size=1)

    for i, batch in enumerate(cl_loader):
        data, labels, mode = batch
        print('batch {} / {}, mode: {}'.format(i, len(cl_loader), mode[0]))

        if i == 0:
            print('data x size : {}'.format(data.size()))
            print('data y size : {}'.format(labels.size()))

        if i == 25 : break




