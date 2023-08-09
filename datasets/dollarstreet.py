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
        attr_col = "Region",
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

        if attr_col == "region":
            print('Using region as attribute for Dollarstreet')
        elif attr_col == "country.name":
            print('Using country name as attribute for Dollarstreet')
        elif attr_col == "income_group":
            print('Using income group as attribute for Dollarstreet')
        else:
            raise ValueError(f"Invalid attr_col: {attr_col}. Must be one of Region, country.name, or Income_Group.")

        self.attr_col = attr_col
        ### self.collect_instances is responsible for creating self.data_df and attrs_by_class
        self.attrs_by_class = self.collect_instances()
        ### Order of classnames doesn't really matter anymore btw. 
        self.classnames = list(self.attrs_by_class.keys())

        ### We will save the list of identifier strings (image_paths) now, at initialization.
        ### THIS SHOULD REMAIN STATIC. Same with self.classnames 
        self.static_img_path_list = self.data_df.index.tolist()


    def collect_instances(self):
        img_paths, valid_classnames, attrs = [], [], []

        attrs_by_class = dict()

        for i, row in self.og_meta_data.iterrows():
            img_rel_path, curr_valid_classnames_str, attr = [row[x] for x in ['imageRelPath', 'topics', self.attr_col]]
            img_path = os.path.join(self.data_dir, img_rel_path)
            img_paths.append(img_path)

            # The og_meta_data dataframe saves the curr_valid_classnames list as a string
            curr_valid_classnames = literal_eval(curr_valid_classnames_str)
            # The next line appends a list of valid_classnames for our current sample to the running list
            valid_classnames.append(curr_valid_classnames)
            # we'll also update attrs_by_class for each valid_classname for the sample now
            for classname in curr_valid_classnames:
                if classname not in attrs_by_class:
                    attrs_by_class[classname] = [attr]
                elif attr not in attrs_by_class[classname]:
                    attrs_by_class[classname].append(attr)

            attrs.append(attr)

        data_df = pd.DataFrame(list(zip(img_paths, valid_classnames, attrs)),
                                    columns=['img_path', 'valid_classnames', 'attr'])
        self.data_df = data_df.set_index('img_path')

        return attrs_by_class
