import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import pandas as pd
import glob
import os
import json
from constants import _IMAGENET_CLASSNAMES, _CACHED_DATA_ROOT, _IMAGENET_DATA_ROOT, _IMAGENET_LABELS_TXT, _IMAGENETV2_DATA_ROOT
from datasets.base_dataset import ClassificationDset
from typing import List, Dict

standard_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

class ImageNet(ClassificationDset):
    """
    ImageNet. Non-attributed dataset
    """

    has_gt_attrs = False

    def __init__(
        self, 
        dsetname: str = 'imagenet', 
        transform = standard_transform, 
        data_dir: str = _IMAGENET_DATA_ROOT, 
        inet_split: str = 'val'
    ):
        ''' data_dir should correspond to your ImageNet path, as Living17 is a subset of it. '''
        self.imagenet_dir = data_dir
        self.inet_split = inet_split
        self.dsetname = dsetname
        if self.inet_split == 'train':
            self.dsetname += '_train'

        # with open(os.path.join(self.imagenet_dir, 'labels.txt')) as f:
        with open(_IMAGENET_LABELS_TXT) as f:
            inet_wnid_and_labels = f.readlines()

        self.wnid_to_label = dict()
        for i, wnid_and_label in enumerate(inet_wnid_and_labels):
            wnid, label = wnid_and_label.split(',')
            # label = label.split('\n')[0]
            label = _IMAGENET_CLASSNAMES[i]
            self.wnid_to_label[wnid] = label

        self.classnames = list(self.wnid_to_label.values())
        self.transform = transform

        self.attrs = ['']
        self.attrs_by_class = dict({classname:[''] for classname in self.classnames})

        self.collect_instances()

        ### We will save the list of identifier strings (image_paths) now, at initialization.
        ### THIS SHOULD REMAIN STATIC. Same with self.classnames 
        self.static_img_path_list = self.data_df.index.tolist()

    def collect_instances(self):
        img_paths, valid_classnames, attrs = [], [], []

        # Recall, each Breeds class consists of the union of images from a set of ImageNet classes
        # inet_cls_idx_in_class is the set of ImageNet classes that compose the current Breeds class
        # 'classname' is the name of the current *Breeds* class
        for wnid, classname in self.wnid_to_label.items():
            curr_img_paths = glob.glob(os.path.join(self.imagenet_dir, self.inet_split, wnid, '*'))
            img_paths.extend(curr_img_paths)
            N_imgs_in_class = len(curr_img_paths)

            valid_classnames.extend([[classname]]*N_imgs_in_class)
            attrs.extend(['']*N_imgs_in_class)

        data_df = pd.DataFrame(list(zip(img_paths, valid_classnames, attrs)), 
                                    columns=['img_path', 'valid_classnames', 'attr'])
        self.data_df = data_df.set_index('img_path')

    def caption_gt_subpop(self, classname: str, attr: str) -> str:
        return classname



class ImageNetv2(ClassificationDset):
    """
    ImageNetv2. Non-attributed dataset
    """

    has_gt_attrs = False

    def __init__(
        self, 
        dsetname: str = 'imagenetv2', 
        transform = standard_transform, 
        data_dir: str = _IMAGENETV2_DATA_ROOT, 
    ):
        ''' data_dir should correspond to your ImageNet path, as Living17 is a subset of it. '''
        self.imagenetv2_dir = data_dir
        self.dsetname = dsetname

        self.transform = transform
        self.classnames = self.collect_instances()

        self.attrs = ['']
        self.attrs_by_class = dict({classname:[''] for classname in self.classnames})

        ### We will save the list of identifier strings (image_paths) now, at initialization.
        ### THIS SHOULD REMAIN STATIC. Same with self.classnames 
        self.static_img_path_list = self.data_df.index.tolist()

    def collect_instances(self):
        img_paths, valid_classnames, attrs = [], [], []

        # This is hard-coded, but I'm not sure there is another way to go about it ...
        for i, classname in enumerate(_IMAGENET_CLASSNAMES):
            curr_img_paths = glob.glob(os.path.join(self.imagenetv2_dir, str(i), '*'))
            img_paths.extend(curr_img_paths)
            N_imgs_in_class = len(curr_img_paths)

            valid_classnames.extend([[classname]]*N_imgs_in_class)
            attrs.extend(['']*N_imgs_in_class)

        data_df = pd.DataFrame(list(zip(img_paths, valid_classnames, attrs)), 
                                    columns=['img_path', 'valid_classnames', 'attr'])
        self.data_df = data_df.set_index('img_path')
        return _IMAGENET_CLASSNAMES

    def caption_gt_subpop(self, classname: str, attr: str) -> str:
        return classname



