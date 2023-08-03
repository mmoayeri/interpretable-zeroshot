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
    Dataset needs to implement the regular torch dataset functions, and must also 
    have ways to access attribute list, class list, and attributes per class

    Needed fields:
    - dset.classes : list of class names
    - dset.attrs : list of attribute names
    - dset.attrs_by_class : list of attribute indices for each class
    - dset.class_to_ind, attr_to_ind: dictionaries mapping names of classes/attrs to index in respective lists

    Other considerations:
    - separate class for non-attributed datasets?
    - field encoding type of diversity dataset contains (i.e. 'types', 'states', 'regions', etc?)
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
        row = self.data_df.loc[img_path]
        # row = self.data_df.loc[self.data_df['img_path'] == identifier].iloc[0]
        # valid_classnames, attr = [row[x] for x in ['valid_classnames', 'attr']]

        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        # label_dict = dict({'valid_classnames': valid_classnames, 'attr': attr})
        return img, img_path#, label_dict
    
    # @abstractmethod
    def gt_attrs_by_class(self, classname) -> Dict[str, List[str]]:
        return self.attrs_by_class[classname]

    # @abstractmethod
    def mask_for_class(self, classname: str):
        in_class_fn = lambda valid_classnames: classname in valid_classnames
        mask = self.data_df['valid_classnames'].apply(in_class_fn)
        return mask

    # @abstractmethod
    def ids_for_class(self, classname: str) -> pd.Series:
        return self.data_df.index[self.mask_for_class(classname)]

    # @abstractmethod
    def ids_for_subpop(self, classname: str, attr: str) -> pd.Series:
        mask1 = self.mask_for_class(classname)
        mask2 = self.data_df['attr'] == attr
        return self.data_df.index[mask1 & mask2]

    # @abstractmethod
    def get_dsetname(self) -> str:
        return self.dsetname

    # @abstractmethod
    def valid_classnames_for_id(self, identifier: str) -> List[str]:
        row = self.data_df.loc[identifier]
        return row['valid_classnames']

    # def __len__(self):
    #     return len(self.data_df)

    # @abstractmethod
    # def __getitem__(self, ind):
    #     # The first two items that are returned MUST be img, ind
    #     # We'll cache ind along with all embeddings, just in case the order gets messed up
    #     # ind will be consistent with idx_in_class / idx_in_subpop
    #     raise NotImplementedError

    # @abstractmethod
    # def get_dsetname(self) -> str:
    #     # return a string for the dataset, used in caching + results table
    #     raise NotImplementedError

    # @abstractmethod
    # def ids_for_class(self, classname: str) -> pd.Series:
    #     # Should return indices of images within a class
    #     raise NotImplementedError

    # @abstractmethod
    # def ids_for_subpop(self, classname: str, attr: str) -> pd.Series:
    #     # Should return indices of images within a subpopulation, defined by a (class, attributes) pair
    #     raise NotImplementedError

    # # @property
    # # Place holder for now because I couldn't get the property tag to work
    # @abstractmethod
    # def gt_attrs_by_class(self) -> Dict[str, List[str]]:
    #     raise NotImplementedError('Dataset is missing attrs_by_class, dict containing list of attributes per classname key')

    # @abstractmethod
    # def valid_classnames_for_id(self, identifier: str) -> List[str]:
    #     # Returns list of valid classnames for a sample w/ given identifier 
    #     # Note that each sample's identifier is just it's img_path, so to be interpretable
    #     raise NotImplementedError

    @property
    @abstractmethod
    def has_gt_attrs(self) -> bool:
        # if the dataset has ground truth attribute labels
        pass


    # @abstractmethod
    # def get_labels_in_given_order(self, idx: np.array) -> Union[np.array, List[List[int]]]:
    #     # Returns labels reordered based on idx. Idx is saved from when data is originally loaded,
    #     # and provides indices to rows in dset.data_df, which is our ultimate source of info for each dset. 
    #     raise NotImplementedError

    # @property
    # @abstractmethod
    # def is_multilabel(self) -> bool:
    #     # if the dataset is multilabel, we expect get_label() to return a list of valid class idx
    #     pass
    

    # @abstractmethod
    # def collect_instances(self):
    #     # This method is called upon initialization and constructs the dataframe self.data_df,
    #     # which has (at least) img path, classname, attr in each row

    #     pass

    # def cache_features(self, image_encoder, save_path):
    #     '''
    #     Given an image
    #     '''

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
                    subpop_descriptions_by_cls[classname].append(f'{classname}: a {attr}') # like in waffleclip
        return subpop_descriptions_by_cls
    #     # Generate caption given classname and attribute name

    #     # QUESTION: Perhaps this should be implemented by the attr inferer?
    #     pass