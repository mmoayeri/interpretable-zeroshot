from PIL import Image
import os
from torchvision import transforms
import pandas as pd
from torch.utils.data import Dataset
from ast import literal_eval
import numpy as np
from datasets.base_dataset import ClassificationDset
from constants import _DSTREET_DATA_ROOT, _DSTREET_INFO_FPATH
from my_utils import load_cached_data

class DollarstreetDataset(ClassificationDset):
    has_gt_attrs = True

    def __init__(
        self,
        attr_col = "region",
        og_meta_data_path: str = _DSTREET_INFO_FPATH,
        data_dir: str = _DSTREET_DATA_ROOT,
        max_allowable_sim_of_classnames: float = 1,
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ]
        )
    ):
        self.dsetname = 'dollarstreet_full'
        self.og_meta_data = pd.read_csv(og_meta_data_path, index_col=0).reset_index()
        self.data_dir = data_dir
        self.transform = transform

        if max_allowable_sim_of_classnames < 1:
            disallowed_classes_path = f'/checkpoint/mazda/data/meta_files/dollarstreet_problem_classes_thresh_{max_allowable_sim_of_classnames}.pkl'
            assert os.path.exists(disallowed_classes_path), 'For now, only use 0.8 or 0.9 for max_allowable_sim_of_classnames. If you absolutely need to try something else, ping Mazda. He will run record_mit_states_problem_classes (a simple fn, it just lives somewhere else right now .. too low priority to fix atm).'
            self.disallowed_classes = load_cached_data(disallowed_classes_path)
            self.dsetname = f'dollarstreet_thresh_{max_allowable_sim_of_classnames}'
        else:
            self.disallowed_classes = []

        ### self.collect_instances is responsible for creating self.data_df and classnames list
        self.classnames = self.collect_instances()

        ### We will save the list of identifier strings (image_paths) now, at initialization.
        ### THIS SHOULD REMAIN STATIC. Same with self.classnames 
        self.static_img_path_list = self.data_df.index.tolist()

        self.allowed_attr_cols = ['region', 'country_name', 'income_level']
        self.set_attribute_column(attr_col)

    def collect_instances(self):
        img_paths, valid_classnames = [], []
        regions, country_names, income_groups = [], [], []

        classnames = []

        for i, row in self.og_meta_data.iterrows():
            img_rel_path, curr_valid_classnames_str = [row[x] for x in ['imageRelPath', 'topics']]
            img_path = os.path.join(self.data_dir, img_rel_path)
            
            # The og_meta_data dataframe saves the curr_valid_classnames list as a string
            curr_valid_classnames = literal_eval(curr_valid_classnames_str)
            # let's take out classes that have too much overlap
            curr_valid_classnames = [c for c in curr_valid_classnames if c not in self.disallowed_classes]
            if len(curr_valid_classnames) == 0:
                # we do not include entries that whose only labels are disallowed classes
                continue
            # The next line appends a list of valid_classnames for our current sample to the running list
            valid_classnames.append(curr_valid_classnames)

            img_paths.append(img_path)

            for attr_list, attr_name in zip([regions, country_names, income_groups], 
                                            ['region', 'country_name', 'income_level']):
                attr_list.append(row[attr_name])

            # we'll also update attrs_by_class for each valid_classname for the sample now
            classnames.extend(curr_valid_classnames)

        data_df = pd.DataFrame(list(zip(img_paths, valid_classnames, regions, country_names, income_groups)),
                               columns=['img_path', 'valid_classnames', 'region', 'country_name', 'income_level'])
        self.data_df = data_df.set_index('img_path')

        ### Order of classnames doesn't really matter anymore btw. 
        classnames = list(set(classnames))
        return classnames

    def set_attribute_column(self, attr_col: str) -> None:
        assert attr_col in self.allowed_attr_cols, f'Invalid attr_col: {attr_col}. Must be one of {", ".join(self.allowed_attr_cols)}.'
        print(f'Setting {attr_col} as attribute for Dollarstreet (i.e. subpopulations will be defined by a classname and {attr_col})')
        self.data_df['attr'] = self.data_df[attr_col]
        self.attr_col = attr_col

        # build attr_by_class
        attrs_by_class = dict({classname:[] for classname in self.classnames})

        for i, row in self.data_df.iterrows():
            curr_valid_classnames, attr = [row[x] for x in ['valid_classnames', 'attr']]
            for classname in curr_valid_classnames:
                attrs_by_class[classname].append(attr)            

        attrs_by_class = dict({classname:list(set(attrs)) for classname, attrs in attrs_by_class.items()})
        self.attrs_by_class = attrs_by_class
        print(f'Updated dset.attrs_by_class dictionary for attribute {attr_col}')

    def caption_gt_subpop(self, classname: str, attr: str) -> str:
        if self.attr_col == 'region':
            caption = f'{classname} from the region {attr}'
        elif self.attr_col == 'income_level':
            caption = f'{classname} from a {attr} country'
        elif self.attr_col == 'country_name':
            caption = f'{classname} from the country {attr}'
        else:
            raise ValueError(f'Invalid attr_col {self.attr_col}. Must be one of {", ".join(self.allowed_attr_cols)}.')    
        return caption