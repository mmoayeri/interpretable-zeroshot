from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import Dict, List, Union
import torch
import numpy as np
import pandas as pd
from PIL import Image
import os
from torchvision import transforms


class ClassificationDset(ABC, Dataset):
    '''
    Each dataset in our study will implement this abstract class.
    Majority of functionality is already implemented here.

    When implementing a new dataset, the following are required:
    1. self.dsetname: a str name of the dataset; used in caching image embeddings.
    2. self.data_df: a dataframe where
        a. 'img_path' is the index
        b. We have columns for:
            i. valid_classnames: List of strings per row, corresponding to classes present in image
            ii. attr: a str corresponding to the attribute present in the image
    3. self.gt_attrs_by_class: dictionary mapping classname to list of groundtruth attributes for that class
    4. self.classnames: list of classnames
    5. has_gt_attrs: a bool property that indicates whether the dataset is attributed
            Note: if false, 2bii and 3 are not needed ...
    
    These things should be made upon initialization.
    '''
    def __init__(self,
        dsetname: str
    ):
        self.collect_instances()

        pass

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, ind: int):
        img_path = self.static_img_path_list[ind]
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        if img.shape[0] == 1:
            img = torch.cat([img]*3, axis=0)

        return img, img_path
    
    def gt_attrs_by_class(self, classname) -> Dict[str, List[str]]:
        if self.has_gt_attrs:
            return self.attrs_by_class[classname]
        else:
            raise Exception(f'Dataset {self.dsetname} does not have ground truth attributes.') 

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

    @property
    @abstractmethod
    def has_gt_attrs(self) -> bool:
        # if the dataset has ground truth attribute labels
        pass

    # TODO: move this guy to his own class; should be separate as this pertains to
    #       VLM prompting, and also may be affected by attribute category
    def subpop_descriptions_from_attrs(
        self,
        attrs_by_class: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:

        subpop_descriptions_by_cls = dict()
        for classname, attrs in attrs_by_class.items():
            subpop_descriptions_by_cls[classname] = []
            for attr in attrs:
                if attr is None:
                    # This is for our Vanilla case, where we just pass the classname
                    subpop_descriptions_by_cls[classname].append(classname)
                else:
                    # TODO: think about this more
                    # other options include {classname} that is/has {attr} (like Vondrick)
                    # or perhaps something that is prompt specific: e.g. {attr}, a kind of {classname}
                    # subpop_descriptions_by_cls[classname].append(f'{attr} {classname}')
                    # subpop_descriptions_by_cls[classname].append(f'{attr}, a kind of {classname}')
                    subpop_descriptions_by_cls[classname].append(f'{classname}: from {attr}') # like in waffleclip
                    # subpop_descriptions_by_cls[classname].append(f'{classname} from {attr}') # like in waffleclip
        return subpop_descriptions_by_cls