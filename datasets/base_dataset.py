from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import Dict, List, Union
import torch
import numpy as np


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

    @abstractmethod
    def __getitem__(self, ind):
        # The first two items that are returned MUST be img, ind
        # We'll cache ind along with all embeddings, just in case the order gets messed up
        # ind will be consistent with idx_in_class / idx_in_subpop
        raise NotImplementedError

    @abstractmethod
    def get_dsetname(self) -> str:
        # return a string for the dataset, used in caching + results table
        raise NotImplementedError

    @abstractmethod
    def idx_in_class(self, classname: str) -> np.array:
        # Should return indices of images within a class
        raise NotImplementedError

    @abstractmethod
    def idx_in_subpop(self, classname: str, attr: str) -> np.array:
        # Should return indices of images within a subpopulation, defined by a (class, attributes) pair
        raise NotImplementedError

    # @property
    # Place holder for now because I couldn't get the property tag to work
    @abstractmethod
    def gt_attrs_by_class(self) -> Dict[str, List[str]]:
        raise NotImplementedError('Dataset is missing attrs_by_class, dict containing list of attributes per class')

    @abstractmethod
    def get_labels_in_given_order(self, idx: np.array) -> Union[np.array, List[List[int]]]:
        # Returns labels reordered based on idx. Idx is saved from when data is originally loaded,
        # and provides indices to rows in dset.data_df, which is our ultimate source of info for each dset. 
        raise NotImplementedError

    @property
    @abstractmethod
    def is_multilabel(self) -> bool:
        # if the dataset is multilabel, we expect get_label() to return a list of valid class idx
        pass

    @property
    @abstractmethod
    def has_gt_attrs(self) -> bool:
        # if the dataset has ground truth attribute labels
        pass
    

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
                    subpop_descriptions_by_cls[classname].append(f'{attr}, a kind of {classname}')
        return subpop_descriptions_by_cls
    #     # Generate caption given classname and attribute name

    #     # QUESTION: Perhaps this should be implemented by the attr inferer?
    #     pass