from PIL import Image
import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from datasets.base_dataset import ClassificationDset
from typing import List, Dict
from constants import _GEODE_INFO_FPATH

### Thanks a million to Megan who provided great starter code!
### Link here: https://github.com/fairinternal/Interplay_of_Model_Properties/blob/main/datasets/geode.py
class GeodeDataset(ClassificationDset):
    has_gt_attrs = True

    def __init__(
        self,
        attr_col: str="region",
        og_meta_data_path: str = _GEODE_INFO_FPATH,
        data_dir: str = "/checkpoint/meganrichards/datasets/geode/images/",
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ]
        )
    ):
        self.dsetname = 'geode'
        self.og_meta_data = pd.read_csv(og_meta_data_path, index_col=0).reset_index()
        self.data_dir = data_dir
        self.transform = transform
        self.attr_col = attr_col

        ### self.collect_instances is responsible for creating self.data_df and classnames list
        self.classnames = self.collect_instances()

        ### We will save the list of identifier strings (image_paths) now, at initialization.
        ### THIS SHOULD REMAIN STATIC. Same with self.classnames 
        self.static_img_path_list = self.data_df.index.tolist()

        self.allowed_attr_cols = ['region', 'country']
        self.set_attribute_column(attr_col)

    def collect_instances(self) -> List[str]:
        # kind of weird but we return the list of unique classnames
        img_paths, valid_classnames = [], []
        regions, country_names = [], []

        classnames = []

        for i, row in self.og_meta_data.iterrows():
            img_rel_path, classname = [row[x] for x in ['file_path', 'object']]
            img_path = os.path.join(self.data_dir, img_rel_path)
            img_paths.append(img_path)

            # GeoDE is single label, but we save it as multilabel for consistency
            valid_classnames.append([classname])

            for attr_list, attr_name in zip([regions, country_names], 
                                            ['region', 'ip_country']):
                attr_list.append(row[attr_name])

            # we also keep track of all classnames we've seen, which this function returns after getting unique set
            if classname not in classnames:
                classnames.append(classname)

        data_df = pd.DataFrame(list(zip(img_paths, valid_classnames, regions, country_names)),
                               columns=['img_path', 'valid_classnames', 'region', 'country'])
        self.data_df = data_df.set_index('img_path')

        return classnames

    def set_attribute_column(self, attr_col: str) -> None:
        assert attr_col in self.allowed_attr_cols, f'Invalid attr_col: {attr_col}. Must be one of {", ".join(self.allowed_attr_cols)}.'
        print(f'Setting {attr_col} as attribute for GeoDE (i.e. subpopulations will be defined by a classname and {attr_col})')
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
        elif self.attr_col == 'country.name':
            caption = f'{classname} from the country {attr}'
        else:
            raise ValueError(f'Invalid attr_col {self.attr_col}. Must be one of {", ".join(self.allowed_attr_cols)}.')    
        return caption