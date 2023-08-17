from PIL import Image
import os
from torchvision import transforms
import pandas as pd
from torch.utils.data import Dataset
from ast import literal_eval
import numpy as np
from datasets.base_dataset import ClassificationDset

class DollarstreetDataset(ClassificationDset):
    has_gt_attrs = True

    def __init__(
        self,
        attr_col = "region",
        og_meta_data_path: str = "/checkpoint/mazda/dollarstreet_test_house_separated_with_metadata.csv",
        data_dir: str = "/checkpoint/meganrichards/datasets/dollarstreet_kaggle/dataset_dollarstreet/",
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ]
        )
    ):
        self.dsetname = 'dollarstreet'
        self.og_meta_data = pd.read_csv(og_meta_data_path, index_col=0).reset_index()
        self.data_dir = data_dir
        self.transform = transform

        ### self.collect_instances is responsible for creating self.data_df and classnames list
        self.classnames = self.collect_instances()

        ### We will save the list of identifier strings (image_paths) now, at initialization.
        ### THIS SHOULD REMAIN STATIC. Same with self.classnames 
        self.static_img_path_list = self.data_df.index.tolist()

        self.allowed_attr_cols = ['region', 'country.name', 'income_group']
        self.set_attribute_column(attr_col)

    def collect_instances(self):
        img_paths, valid_classnames = [], []
        regions, country_names, income_groups = [], [], []

        classnames = []

        for i, row in self.og_meta_data.iterrows():
            img_rel_path, curr_valid_classnames_str = [row[x] for x in ['imageRelPath', 'topics']]
            img_path = os.path.join(self.data_dir, img_rel_path)
            img_paths.append(img_path)

            # The og_meta_data dataframe saves the curr_valid_classnames list as a string
            curr_valid_classnames = literal_eval(curr_valid_classnames_str)
            # The next line appends a list of valid_classnames for our current sample to the running list
            valid_classnames.append(curr_valid_classnames)

            for attr_list, attr_name in zip([regions, country_names, income_groups], 
                                            ['region', 'country.name', 'income_group']):
                attr_list.append(row[attr_name])

            # we'll also update attrs_by_class for each valid_classname for the sample now
            classnames.extend(curr_valid_classnames)

        data_df = pd.DataFrame(list(zip(img_paths, valid_classnames, regions, country_names, income_groups)),
                               columns=['img_path', 'valid_classnames', 'region', 'country.name', 'income_group'])
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
        elif self.attr_col == 'country.name':
            caption = f'{classname} from the country {attr}'
        else:
            raise ValueError(f'Invalid attr_col {self.attr_col}. Must be one of {", ".join(self.allowed_attr_cols)}.')    
        return caption