import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import pandas as pd
import glob
import os
import json
from constants import _IMAGENET_CLASSNAMES, _CACHED_DATA_ROOT
from datasets.base_dataset import ClassificationDset
from typing import List, Dict

standard_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

class Breeds(ClassificationDset):
    """
    Breeds is a family of datasets, where each consists of sets supercategories of ImageNet classes.
    For example, Living17 consists of 17 supercategories from ImageNet, with four ImageNet classes making up each supercategory. 
    Thus, each Breeds class has a number of ImageNet classes within them, which we take as its gt subpopuplations (e.g. 'black bear' is a subpopulation within 'bear' class.)

    Options for dsetname are entity13, entity30, living17, nonliving26
    """

    # is_multilabel = False
    has_gt_attrs = True

    def __init__(
        self, 
        dsetname='living17', 
        transform=standard_transform, 
        data_dir='/datasets01/imagenet_full_size/061417/', 
        inet_split='val', 
        breeds_info_path='/private/home/mazda/multiple_cls_vecs/datasets/breeds_info.json'
    ):
        ''' data_dir should correspond to your ImageNet path, as Living17 is a subset of it. '''
        self.dsetname = dsetname
        
        self.imagenet_dir = data_dir
        self.inet_split = inet_split
        with open(os.path.join(self.imagenet_dir, 'labels.txt')) as f:
            inet_wnid_and_labels = f.readlines()
        self.inet_wnids = [wnid_and_label.split(',')[0] for wnid_and_label in inet_wnid_and_labels]

        self.transform = transform

        # breeds_info will have a dictionary w/ key for each dataset within breeds (e.g. 'living17' is a key)
        # breeds_info['living17'] will return another dictionary with living17 classnames as keys and 
        # ImageNet class idx as values (specifically, the classes in ImageNet that make up the Breeds class)
        with open(breeds_info_path, 'r') as f:
            breeds_info = json.load(f)
        self.breeds_classes_to_inet_cls_idx = breeds_info[dsetname]

        self.classnames = list(self.breeds_classes_to_inet_cls_idx.keys())
        # self.cls_to_ind = dict({c:i for i,c in enumerate(self.classnames)})

        self.attrs = []
        self.attrs_by_class = dict({classname:[] for classname in self.classnames})

        for classname, inet_cls_idx in self.breeds_classes_to_inet_cls_idx.items():
            
            for inet_cls_ind in inet_cls_idx:
                attr = _IMAGENET_CLASSNAMES[inet_cls_ind]
                self.attrs.append(attr)
                self.attrs_by_class[classname].append(attr)#_ind)

        self.collect_instances()

        ### We will save the list of identifier strings (image_paths) now, at initialization.
        ### THIS SHOULD REMAIN STATIC. Same with self.classnames 
        self.static_img_path_list = self.data_df.index.tolist()

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, ind: int):
        img_path = self.static_img_path_list[ind]
        row = self.data_df.loc[img_path]
        # row = self.data_df.loc[self.data_df['img_path'] == identifier].iloc[0]
        valid_classnames, attr = [row[x] for x in ['valid_classnames', 'attr']]

        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        label_dict = dict({'valid_classnames': valid_classnames, 'attr': attr})
        
        return img, img_path, label_dict

    def collect_instances(self):
        img_paths, valid_classnames, attrs, class_idx = [], [], [], []

        # Recall, each Breeds class consists of the union of images from a set of ImageNet classes
        # inet_cls_idx_in_class is the set of ImageNet classes that compose the current Breeds class
        # 'classname' is the name of the current *Breeds* class
        for classname, inet_cls_idx_in_class in self.breeds_classes_to_inet_cls_idx.items():
            for inet_cls_ind in inet_cls_idx_in_class:

                curr_img_paths = glob.glob(os.path.join(self.imagenet_dir,self.inet_split,self.inet_wnids[inet_cls_ind],'*'))
                img_paths.extend(curr_img_paths)
                N_imgs_in_subpop = len(curr_img_paths)

                valid_classnames.extend([[classname]]*N_imgs_in_subpop)
                # class_idx.extend([self.cls_to_ind[classname]]*N_imgs_in_subpop)

                attr = _IMAGENET_CLASSNAMES[inet_cls_ind]
                attrs.extend([attr]*N_imgs_in_subpop)

        data_df = pd.DataFrame(list(zip(img_paths, valid_classnames, attrs)), 
                                    columns=['img_path', 'valid_classnames', 'attr'])
        self.data_df = data_df.set_index('img_path')
        # self.data_df = pd.DataFrame(list(zip(img_paths, classnames, attrs, class_idx)), columns=['img_path', 'class', 'attr', 'cls_idx'])

    def gt_attrs_by_class(self, classname) -> Dict[str, List[str]]:
        return self.attrs_by_class[classname]

    def mask_for_class(self, classname: str):
        in_class_fn = lambda valid_classnames: classname in valid_classnames
        mask = self.data_df['valid_classnames'].apply(in_class_fn)
        return mask

    def ids_for_class(self, classname: str) -> pd.Series:
        return self.data_df.index[self.mask_for_class(classname)]

    def ids_for_subpop(self, classname: str, attr: str) -> pd.Series:
        mask1 = self.mask_for_class(classname)
        mask2 = self.data_df['attr'] == attr
        return self.data_df.index[mask1 & mask2]

    def get_dsetname(self) -> str:
        return self.dsetname

    def valid_classnames_for_id(self, identifier: str) -> List[str]:
        row = self.data_df.loc[identifier]
        return row['valid_classnames']