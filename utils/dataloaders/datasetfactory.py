import os
import torchvision.transforms as transforms
from . import omniglot as om
#path = "/network/data1/omniglot"

try:
    data_folder = os.environ['SCRATCH']+'/omniglot'
except:
    if os.path.isdir("/network/data1/"):
        data_folder = "/network/data1/omniglot"
    else:
        data_folder = ".data/"

class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(name, train=True, path=None, background=True, all_=False):

        if name == "omniglot":
            train_transform = transforms.Compose(
                [transforms.Resize((84, 84)),
                 transforms.ToTensor()])
            if path is None:
                return om.Omniglot(data_folder, background=background, download=True, train=train,
                                   transform=train_transform, all_=all_)
            else:
                return om.Omniglot(path, background=background, download=True, train=train,
                                   transform=train_transform, all_=all_)
                #return om.Omniglot(path, download=True, background=train, transform=train_transform,all_=all_)


        else:
            print("Unsupported Dataset")
            assert False

    @staticmethod
    def get_tasks(name, train=True, path=None, background=True, all_=False, redo_augment=False, augment=False):

        if name == "omniglot":dataset_indices = list(np.random.permutation(num_samples))
            augment_transform = transforms.RandomAffine(degrees=(-30, 30), translate=(0.1, 0.3))
            train_transform = transforms.Compose(
                [transforms.Resize((28, 28)),
                 transforms.ToTensor()])
            if path is None:
                return om.MultitaskOmniglot(data_folder, background=background, download=True, train=train,
                                   transform=train_transform, augment_transform=augment_transform, all_=all_, redo_augment=redo_augment, augment=augment)
            else:
                return om.MultitaskOmniglot(path, background=background, download=True, train=train,
                                   transform=train_transform, augment_transform=augment_transform, all_=all_,redo_augment=redo_augment, augment=augment)
                # return om.Omniglot(path, download=True, background=train, transform=train_transform,all_=all_)


        else:
            print("Unsupported Dataset")
            assert False

