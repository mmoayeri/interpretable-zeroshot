import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import pandas as pd
import glob
import os
from my_utils import load_cached_results

standard_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

class MITStates(Dataset):
    """
    MIT States Dataset. It consists of a image sets corresponding to (noun, adj) pairs, where
    adjectives contextualize the noun to various diverse states (e.g. sliced tomato vs. pureed tomato)
    """

    def __init__(self, data_dir='/private/home/mazda/data/mit_states', 
                transform=standard_transform, min_cnt_in_group=10, filter_classes=True):
        self.min_cnt_in_group = min_cnt_in_group

        if filter_classes:
            assert os.path.exists('results/mit_states_problem_classes.pkl'), 'Make sure to run record_mit_states_problem_classes in analyze_mit_states.py first'
            self.disallowed_classes = load_cached_results('mit_states_problem_classes')
        else:
            self.disallowed_classes = []

        self.data_dir = data_dir
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
        
        self.data_df = pd.DataFrame(list(zip(img_paths, valid_classnames, attrs)), 
                                    columns=['img_path', 'valid_classnames', 'attr'])
        self.data_df = data_df.set_index('img_path')

        return adjs_per_noun

    def caption_gt_subpop(self, classname: str, attr: str) -> str:
        return f'{attr} {classname}'