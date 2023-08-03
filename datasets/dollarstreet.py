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
        # og_meta_data_path: str = "/checkpoint/meganrichards/datasets/dollarstreet_kaggle/dataset_dollarstreet/house_separated_with_metadata/test.csv",
        og_meta_data_path: str = "/checkpoint/mazda/dollarstreet_test_house_separated_with_metadata_copy.csv",
        data_dir: str = "/checkpoint/meganrichards/datasets/dollarstreet_kaggle/dataset_dollarstreet/",
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        ),
        attr_col = "Region",
        # label_col="imagenet_synset_id",  # replace with 'topic_indicies' to get original dollarstreet label index instead of ImageNet
        # label_col="topics",  # replace with 'topic_indicies' to get original dollarstreet label index instead of ImageNet
    ):
        self.dsetname = 'dollarstreet'
        self.og_meta_data = pd.read_csv(og_meta_data_path, index_col=0).reset_index()
        self.data_dir = data_dir
        self.transform = transform
        # self.label_col = label_col
        # self.og_meta_data[label_col] = self.file[label_col].apply(literal_eval)

        # if "imagenet" in label_col:
        #     print("Using 1k mapping for DollarStreet")
        # else:
        #     print("Using DollarStreet original labels")

        if attr_col == "Region":
            print('Using Region as attribute for Dollarstreet')
        elif attr_col == "country.name":
            print('Using Country Name as attribute for Dollarstreet')
        elif attr_col == "Income_Group":
            print('Using Income Group as attribute for Dollarstreet')
        else:
            raise ValueError(f"Invalid attr_col: {attr_col}. Must be one of Region, country.name, or Income_Group.")

        self.attr_col = attr_col
        ### self.collect_instances is responsible for creating self.data_df and self.attrs_by_class
        self.collect_instances()

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

        self.attrs_by_class = attrs_by_class
        ### Order of classnames doesn't really matter anymore btw. 
        self.classnames = list(self.attrs_by_class.keys())
