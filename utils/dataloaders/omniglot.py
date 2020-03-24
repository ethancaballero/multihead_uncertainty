from __future__ import print_function

import os
import shutil
from os.path import join

import attr
from attr.validators import instance_of
import numpy as np
import random
from PIL import Image
import torch.utils.data as data
from utils.mypy_shim import MyPyShim
import torchvision.transforms as transforms

from .utils import download_url, check_integrity, list_dir, list_files


class SplitOmniglot(data.Dataset):
    """`Omniglot <https://github.com/brendenlake/omniglot>`_ Dataset.
    in the continual learning setup (no background set and evaluation set, just train/validation/test set per character)
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        num_tasks (int): number of tasks (alphabets) to use (if more than 50, it repeats the alphabets with
            new random classes/characters per task)
        classes_per_task (int): number of classes per each task (number of characters to take from each alphabet)
        batch_size (int): batch_size
        augmentations (int): The number of data augmentations (random rotations and translations) to apply to each image
        incremental_class (bool): Whether to treat it as a multi-task (where the target labels starts from 0 for each task) or
            as an incremental class setting (where the target labels keeps incrementing with each task)
        image_size (tuple or int): The resized image size (None for no resizing)
        download (bool, optional): If true, downloads the dataset zip files from the internet and
            puts it in root directory. If the zip files are already downloaded, they are not
            downloaded again.
    """
    folder = ''
    download_url_prefix = 'https://dl.dropboxusercontent.com/s/ihznm23bu11dr2f/split-omniglot.zip'
    zip_md5 = 'ea35dcac6a3fd2de6c07ced48cc9ddbd'

    def __init__(self, root, num_tasks=50, classes_per_task=5, batch_size=1, augmentations=None,
                 seed=None, incremental_class=False, image_size=None, download=False):
        self.root = join(os.path.expanduser(root), self.folder)
        self.num_tasks = num_tasks
        self.classes_per_task = classes_per_task
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.seed = seed
        self.incremental_class = incremental_class
        self.image_size = image_size
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.target_folder = join(self.root, self._get_target_folder())
        self._alphabets = list_dir(self.target_folder)
        self._tasks_loaders = []
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._alphabets)
        for task_id in range(self.num_tasks):
            alphabet = self._alphabets[task_id % self.num_tasks]
            characters = list_dir(join(self.target_folder, alphabet))
            random.shuffle(characters)
            characters = characters[:self.classes_per_task]
            train_loader = data.DataLoader(SplitOmniglotTask(self.target_folder, alphabet, characters,
                                                             self.augmentations, incremental_class=incremental_class, task_id=task_id,
                                                             image_size=image_size, dataset_type='train'),
                                           batch_size=self.batch_size, shuffle=True)
            valid_loader = data.DataLoader(SplitOmniglotTask(self.target_folder, alphabet, characters,
                                                             self.augmentations, incremental_class=incremental_class, task_id=task_id,
                                                             image_size=image_size, dataset_type='validation'),
                                           batch_size=self.batch_size, shuffle=True)
            test_loader = data.DataLoader(SplitOmniglotTask(self.target_folder, alphabet, characters,
                                                            self.augmentations, incremental_class=incremental_class, task_id=task_id,
                                                            image_size=image_size, dataset_type='test'),
                                           batch_size=self.batch_size, shuffle=True)
            self._tasks_loaders.append({'train': train_loader, 'validation': valid_loader, 'test': test_loader})

        print("Total Tasks = ", len(self._tasks_loaders))

    def __len__(self):
        return len(self._tasks_loaders)

    def __getitem__(self, index):
        """
        Args:
            index (int): Task index
        Returns:
            dictionary: {"train":task_train_data, "valid":task_valid_data, "test":task_test_data) where each item is a
                dataloader for a SplitOmniglotTask Dataset object
        """
        # image_name, character_class = self._flat_character_images[index]

        return self._tasks_loaders[index]

    def _check_integrity(self):
        zip_filename = self._get_target_folder()
        if not check_integrity(join(self.root, zip_filename + '.zip'), self.zip_md5):
            return False
        return True

    def download(self):
        import zipfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        filename = self._get_target_folder()
        zip_filename = filename + '.zip'
        url = self.download_url_prefix
        download_url(url, self.root, zip_filename, self.zip_md5)
        print('Extracting downloaded file: ' + join(self.root, zip_filename))
        with zipfile.ZipFile(join(self.root, zip_filename), 'r') as zip_file:
            zip_file.extractall(self.root)

    def _get_target_folder(self):
        return 'split-omniglot'


class SplitOmniglotTask(data.Dataset):
    """`Omniglot <https://github.com/brendenlake/omniglot>`_ Dataset.
    in the continual learning setup (no background set and evaluation set, just train/validation/test set per character)
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        task_folder (string): The task folder, in this case the alphabet folder
        characters (list): The list of characters(classes) to be used in the task
        augmentations (int): The number of data augmentations (random rotations and translations) to apply to each image
        incremental_class (bool): Whether to treat it as a multi-task (where the target labels starts from 0 for each task) or
            as an incremental class setting (where the target labels keeps incrementing with each task)
        task_id (int): the index of this task
        image_size (tuple or int): The resized image size (None for no resizing)
        dataset_type (string): train, validation or test set
    """
    def __init__(self, root, task_folder, characters, augmentations=None,
                 incremental_class=False, task_id = 0, image_size=None, dataset_type="train"):
        self.root = root
        self.task_folder = task_folder
        self._characters = characters
        self.augmentations = augmentations
        self.incremental_class = incremental_class
        self.task_id = task_id
        self.image_size = image_size
        self.dataset_type = dataset_type
        self._resize = self.image_size is not None
        self._augment = augmentations is not None
        self.images_cached = {}

        self.target_folder = join(self.root, self.task_folder)
        if dataset_type == "train":
            self._character_images = [[(image, idx) for image in list_files(join(self.target_folder, character, 'train'), '.png')]
                                      for idx, character in enumerate(self._characters)]
        elif dataset_type == "validation":
            self._character_images = [[(image, idx) for image in list_files(join(self.target_folder, character, 'validation'), '.png')]
                                      for idx, character in enumerate(self._characters)]
        elif dataset_type == "test":
            self._character_images = [[(image, idx) for image in list_files(join(self.target_folder, character, 'test'), '.png')]
                                      for idx, character in enumerate(self._characters)]
        self._flat_character_images = sum(self._character_images, [])
        self.data = [x[0] for x in self._flat_character_images]
        self._targets = [x[1] for x in self._flat_character_images]
        if incremental_class:
            self.targets = [target + len(characters) * self.task_id for target in self._targets]
            self.targets = self._targets
        else:
            self.targets = self._targets

        if self._resize:
            self.transform_resize = transforms.Resize(self.image_size)
        if self._augment:
            self.transform_augment = transforms.RandomAffine(degrees=(-20,20), translate=(0.1,0.2))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        if self._augment:
            return len(self.data) * self.augmentations
        else:
            return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of the task id and the index of the target character class.
        """
        # As index is larger than the data size due to augmentations, this way augmentations are done on the fly
        index = index % len(self.data)
        image_name = self.data[index]
        character_class = self.targets[index]
        image_path = join(self.target_folder, self._characters[self._targets[index]], self.dataset_type, image_name)
        if image_path not in self.images_cached:
            image = Image.open(image_path, mode='r').convert('L')
            if self._resize:
                image = self.transform_resize(image)
            self.images_cached[image_path] = image
        else:
            image = self.images_cached[image_path]

        if self._augment:
            image = self.transform_augment(image)
        image = self.to_tensor(image)
        return image, character_class

    def empty_cache(self):
        self.images_cached = {}

# class InfoDataset(MyPyShim):
#     train: data.Dataset# = attr.ib()
#     valid: data.Dataset# = attr.ib() #can be None
#     test: data.Dataset# = attr.ib() #can be None
#     #task_name: str# = attr.ib()
#
# class MultitaskOmniglot():
#     folder = 'omniglot-py'
#     download_url_prefix = 'https://github.com/brendenlake/omniglot/raw/master/python'
#     zips_md5 = {
#         'images_background': '68d2efa1b9178cc56df9314c21c6e718',
#         'images_evaluation': '6b91aef0f799c5bb55b94e3f2daec811'
#     }
#
#     def __init__(self, root, background=True,
#                  transform=None, target_transform=None, augment_transform=None,
#                  download=False, train=True, all_=False, redo_augment=False, augment=False):
#
#         self.root = join(os.path.expanduser(root), self.folder)
#         self.transform = transform
#         self.target_transform = target_transform
#         self.images_cached = {}
#         self._alphabets=[]
#         self.datasets = []
#         self.augment_transform = augment_transform
#         self.redo_augment=redo_augment
#         self.augment=augment
#
#         for self.background in [True,False]:
#             print(self.background)
#             if download:
#                 self.download()
#
#             if not self._check_integrity():
#                 raise RuntimeError('Dataset not found or corrupted.' +
#                                    ' You can use download=True to download it')
#
#             self.target_folder = join(self.root, self._get_target_folder())
#             self._alphabets = list_dir(self.target_folder)
#
#
#             for alphabet in self._alphabets:
#                 alp_characters = [ join(alphabet,c) for c in list_dir(join(self.target_folder, alphabet))]
#                 if self.augment:
#                     self.augment_omniglot(alp_characters)
#
#                 character_images_train = []
#                 character_images_test = []
#                 character_images_valid = []
#                 for idx, character in enumerate(alp_characters):
#                     images = list_files(join(self.target_folder, character), '.png')
#                     images_train = images[:12]
#                     images_valid = images[12:16]
#                     images_test = images[16:]
#                     #images_augmented = None
#                     #if self.augment:
#                     #    images_augmented = list_files(join(self.target_folder, character, 'augmented'), '.png')
#
#                     for image in images_train:
#                         character_images_train.append((image, idx))
#                         if self.augment:
#                             for image_augmented in list_files(join(self.target_folder, character, 'augmented',image[:-4]), '.png'):
#                                 character_images_train.append(('augmented/'+image[:-4]+'/'+image_augmented, idx))
#
#                     for image in images_test:
#                         character_images_test.append((image, idx))
#                         if self.augment:
#                             for image_augmented in list_files(join(self.target_folder, character, 'augmented',image[:-4]), '.png'):
#                                 character_images_test.append(('augmented/'+image[:-4]+'/'+image_augmented, idx))
#
#                     for image in images_valid:
#                         character_images_valid.append((image, idx))
#                         if self.augment:
#                             for image_augmented in list_files(join(self.target_folder, character, 'augmented',image[:-4]), '.png'):
#                                 character_images_valid.append(('augmented/'+image[:-4]+'/'+image_augmented, idx))
#
#
#
#                 #character_images = [[(image, idx) for image in list_files(join(self.target_folder, character), '.png')]
#                 #                    for idx, character in enumerate(alp_characters)]
#
#
#                 flat_character_images_train = character_images_train #sum(character_images_train, [])
#                 flat_character_images_test = character_images_test #sum(character_images_test, [])
#                 flat_character_images_valid = character_images_valid #sum(character_images_valid, [])
#
#                 data_train = [x[0] for x in flat_character_images_train]
#                 targets_train = [x[1] for x in flat_character_images_train]
#
#                 data_test = [x[0] for x in flat_character_images_test]
#                 targets_test = [x[1] for x in flat_character_images_test]
#
#                 data_valid = [x[0] for x in flat_character_images_valid]
#                 targets_valid = [x[1] for x in flat_character_images_valid]
#                 dataset_train = OmniglotTask(data_train,targets_train,self.target_folder,alp_characters,self.transform, self.target_transform)
#                 dataset_valid = OmniglotTask(data_test,targets_test,self.target_folder,alp_characters,self.transform, self.target_transform)
#                 dataset_test = OmniglotTask(data_valid,targets_valid,self.target_folder,alp_characters,self.transform, self.target_transform)
#                 self.datasets.append((dataset_train, dataset_valid, dataset_test, alphabet))
#
#
#         #self._characters = sum([[join(a, c) for c in list_dir(join(self.target_folder, a))]
#         #                        for a in self._alphabets], [])
#
#
#         #self._character_images = [[(image, idx) for image in list_files(join(self.target_folder, character), '.png')]
#         #                          for idx, character in enumerate(self._characters)]
#         #self._flat_character_images = sum(self._character_images, [])
#         #self.data = [x[0] for x in self._flat_character_images]
#         #self.targets = [x[1] for x in self._flat_character_images]
#         #self.data2 = []
#         #self.targets2 = []
#         #self.new_flat = []
#         #if not all_:
#
#     def download(self):
#         import zipfile
#
#         if self._check_integrity():
#             print('Files already downloaded and verified')
#             return
#
#         filename = self._get_target_folder()
#         zip_filename = filename + '.zip'
#         url = self.download_url_prefix + '/' + zip_filename
#         download_url(url, self.root, zip_filename, self.zips_md5[filename])
#         print('Extracting downloaded file: ' + join(self.root, zip_filename))
#         with zipfile.ZipFile(join(self.root, zip_filename), 'r') as zip_file:
#             zip_file.extractall(self.root)
#
#     def _check_integrity(self):
#         zip_filename = self._get_target_folder()
#         if not check_integrity(join(self.root, zip_filename + '.zip'), self.zips_md5[zip_filename]):
#             return False
#         return True
#
#     def _get_target_folder(self):
#         return 'images_background' if self.background else 'images_evaluation'
#
#     def augment_omniglot(self, characters):
#         for character in characters:
#             character_path = join(self.target_folder, character)
#             if not os.path.isdir(character_path+'/augmented/') or self.redo_augment:
#                 if os.path.isdir(character_path+'/augmented/'): shutil.rmtree(character_path+'/augmented/')
#                 os.mkdir(character_path+'/augmented/')
#                 character_images = list_files(character_path, '.png')
#                 for image_name in character_images:
#                     os.mkdir(character_path + '/augmented/'+image_name[:-4])
#                     image_path = join(character_path, image_name)
#                     image = Image.open(image_path, mode='r').convert('L')
#                     for i in range(20):
#                         image_augmented = self.augment_transform(image)
#                         image_augmented.save(character_path+'/augmented/'+'/'+image_name[:-4]+'/'+str(i)+".png")
#
#
# class OmniglotTask(data.Dataset):
#     def __init__(self, data, target, target_folder, characters, transform=None, target_transform=None):
#         self.data = data
#         self.targets = target
#
#         self.transform = transform
#         self._characters = characters
#         self.target_transform = target_transform
#         self.images_cached = {}
#         self.target_folder = target_folder
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#         Returns:
#             tuple: (image, target) where target is index of the target character class.
#         """
#         # image_name, character_class = self._flat_character_images[index]
#         image_name = self.data[index]
#         character_class = self.targets[index]
#         image_path = join(self.target_folder, self._characters[character_class], image_name)
#         if image_path not in self.images_cached:
#
#             image = Image.open(image_path, mode='r').convert('L')
#             if self.transform:
#                 image = self.transform(image)
#
#             self.images_cached[image_path] = image
#         else:
#             image = self.images_cached[image_path]
#
#         # if self.transform:
#         #     image = self.transform(image)
#
#         if self.target_transform:
#             character_class = self.target_transform(character_class)
#
#         return image, character_class

# class Omniglot(data.Dataset):
#     """`Omniglot <https://github.com/brendenlake/omniglot>`_ Dataset.
#     Args:
#         root (string): Root directory of dataset where directory
#             ``omniglot-py`` exists.
#         background (bool, optional): If True, creates dataset from the "background" set, otherwise
#             creates from the "evaluation" set. This terminology is defined by the authors.
#         transform (callable, optional): A function/transform that  takes in an PIL image
#             and returns a transformed version. E.g, ``transforms.RandomCrop``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#         download (bool, optional): If true, downloads the dataset zip files from the internet and
#             puts it in root directory. If the zip files are already downloaded, they are not
#             downloaded again.
#     """
#     folder = 'omniglot-py'
#     download_url_prefix = 'https://github.com/brendenlake/omniglot/raw/master/python'
#     zips_md5 = {
#         'images_background': '68d2efa1b9178cc56df9314c21c6e718',
#         'images_evaluation': '6b91aef0f799c5bb55b94e3f2daec811'
#     }
#
#     def __init__(self, root, background=True,
#                  transform=None, target_transform=None,
#                  download=False, train=True, all_=False):
#         self.root = join(os.path.expanduser(root), self.folder)
#         self.background = background
#         self.transform = transform
#         self.target_transform = target_transform
#         self.images_cached = {}
#
#         if download:
#             self.download()
#
#         if not self._check_integrity():
#             raise RuntimeError('Dataset not found or corrupted.' +
#                                ' You can use download=True to download it')
#
#         self.target_folder = join(self.root, self._get_target_folder())
#         self._alphabets = list_dir(self.target_folder)
#         self._characters = sum([[join(a, c) for c in list_dir(join(self.target_folder, a))]
#                                 for a in self._alphabets], [])
#         self._character_images = [[(image, idx) for image in list_files(join(self.target_folder, character), '.png')]
#                                   for idx, character in enumerate(self._characters)]
#         self._flat_character_images = sum(self._character_images, [])
#         self.data = [x[0] for x in self._flat_character_images]
#         self.targets = [x[1] for x in self._flat_character_images]
#         self.data2 = []
#         self.targets2 = []
#         self.new_flat = []
#         #if not all_:
#         for a in range(int(len(self.targets) / 20)):
#             start = a * 20
#             if train:
#                 for b in range(start, start + 15):
#                     self.data2.append(self.data[b])
#                     self.targets2.append(self.targets[b])
#                     self.new_flat.append(self._flat_character_images[b])
#                     # print(self.targets[start+b])
#             else:
#                 for b in range(start + 15, start + 20):
#                     self.data2.append(self.data[b])
#                     self.targets2.append(self.targets[b])
#                     self.new_flat.append(self._flat_character_images[b])
#
#         if all_:
#             pass
#         else:
#             print("downloader")
#             self._flat_character_images = self.new_flat
#             self.targets = self.targets2
#             print(self.targets[0:30])
#             self.data = self.data2
#
#         print("Total classes = ", np.max(self.targets))
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#         Returns:
#             tuple: (image, target) where target is index of the target character class.
#         """
#         # image_name, character_class = self._flat_character_images[index]
#         image_name = self.data[index]
#         character_class = self.targets[index]
#         image_path = join(self.target_folder, self._characters[character_class], image_name)
#         if image_path not in self.images_cached:
#             image = Image.open(image_path, mode='r').convert('L')
#             if self.transform:
#                 image = self.transform(image)
#
#             self.images_cached[image_path] = image
#         else:
#             image = self.images_cached[image_path]
#
#         # if self.transform:
#         #     image = self.transform(image)
#
#         if self.target_transform:
#             character_class = self.target_transform(character_class)
#
#         return image, character_class
#
#     def _cache_data(self):
#         pass
#
#     def _check_integrity(self):
#         zip_filename = self._get_target_folder()
#         if not check_integrity(join(self.root, zip_filename + '.zip'), self.zips_md5[zip_filename]):
#             return False
#         return True
#
#     def download(self):
#         import zipfile
#
#         if self._check_integrity():
#             print('Files already downloaded and verified')
#             return
#         filename = self._get_target_folder()
#         zip_filename = filename + '.zip'
#         url = self.download_url_prefix + '/' + zip_filename
#         download_url(url, self.root, zip_filename, self.zips_md5[filename])
#         print('Extracting downloaded file: ' + join(self.root, zip_filename))
#         with zipfile.ZipFile(join(self.root, zip_filename), 'r') as zip_file:
#             zip_file.extractall(self.root)
#
#     def _get_target_folder(self):
#         return 'images_background' if self.background else 'images_evaluation'