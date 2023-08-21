import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import pandas as pd
import glob
import os
from datasets import ClassificationDset
from my_utils import load_cached_data

standard_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

class MITStates(ClassificationDset):
    """
    MIT States Dataset. It consists of a image sets corresponding to (noun, adj) pairs, where
    adjectives contextualize the noun to various diverse states (e.g. sliced tomato vs. pureed tomato)
    """
    has_gt_attrs = True

    # def __init__(self, data_dir='/private/home/mazda/data/mit_states', 
    def __init__(
            self, 
            data_dir: str='/checkpoint/mazda/data/mit_states', 
            max_allowable_sim_of_classnames: float=0.9,
            transform=standard_transform
        ):
        """
        max_allowable_sim_of_classnames is the max CLIP cos-sim allowed bw two classnames. We perform this filtering
        step since MIT States was not originally designed as a classification dataset.
        """
        self.data_dir = data_dir
        self.dsetname = f'mit_states_thresh_{max_allowable_sim_of_classnames}'

        if max_allowable_sim_of_classnames < 1:
            disallowed_classes_path = f'{self.data_dir}/problem_classes/thresh_{max_allowable_sim_of_classnames}.pkl'
            assert os.path.exists(disallowed_classes_path), 'Make sure to run record_mit_states_problem_classes with same max_allowable_sim_of_classnames in my_utils first'
            self.disallowed_classes = load_cached_data(disallowed_classes_path)
        else:
            self.disallowed_classes = []

        self.transform = transform
        adjs_per_noun = self.collect_instances()
        self.classnames = list(adjs_per_noun.keys())
        self.attrs_by_class = adjs_per_noun

        ### We will save the list of identifier strings (image_paths) now, at initialization.
        ### THIS SHOULD REMAIN STATIC. Same with self.classnames 
        self.static_img_path_list = self.data_df.index.tolist()


    def collect_instances(self):
        img_paths, valid_classnames, attrs = [], [], []
        adjs_per_noun = dict()
        for adj_noun_path in glob.glob(f'{self.data_dir}/images/*'):
            curr_img_paths = glob.glob(adj_noun_path+'/*')

            adj, noun = adj_noun_path.split(f'{self.data_dir}/images/')[-1].split(' ')
            
            if noun in self.disallowed_classes:
                continue                

            if adj == 'adj':
                adj = 'typical'

            img_paths.extend(curr_img_paths)
            valid_classnames.extend([noun]*len(curr_img_paths))
            attrs.extend([adj]*len(curr_img_paths))

            if noun not in adjs_per_noun:
                adjs_per_noun[noun] = []
            adjs_per_noun[noun].append(adj)
        
        data_df = pd.DataFrame(list(zip(img_paths, valid_classnames, attrs)), 
                                    columns=['img_path', 'valid_classnames', 'attr'])
        self.data_df = data_df.set_index('img_path')

        return adjs_per_noun

    def caption_gt_subpop(self, classname: str, attr: str) -> str:
        return f'{attr} {classname}'