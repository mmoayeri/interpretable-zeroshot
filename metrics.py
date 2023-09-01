import numpy as np
import torch
from torch import Tensor
from typing import List, Dict
from constants import _REGIONS, _INCOME_LEVELS

def mark_as_correct(pred_classnames: List[str], ids: List[str], dset) -> Dict[str, bool]:
    '''
    Returns a dictionary with (key, value) pairs as follows:
        - key: identifier for a sample (i.e. it's full image path)
        - value: a boolean, which is True if the sample was predicted correctly. Specifically, 
                 if the single predicted classname was amongst valid classnames for that sample.
    '''
    is_correct_by_id = dict({identifier: (pred_classname in dset.valid_classnames_for_id(identifier))
                    for pred_classname, identifier in zip(pred_classnames, ids)})
    
    return is_correct_by_id


def acc_by_class_and_subpop(is_correct_by_id: Dict[str, bool], dset, min_cnt: int=10):

    acc_by_class, acc_by_subpop = dict(), dict()

    for classname in dset.classnames:
        
        ids_for_class = dset.ids_for_class(classname)
        if len(ids_for_class) < min_cnt:
            continue
        correct_for_class = [is_correct_by_id[x] for x in ids_for_class]
        acc_by_class[classname] = np.mean(correct_for_class) * 100

        if dset.has_gt_attrs: # We can only compute subpopulation accuracy if dset has gt attrs
            for attr in dset.gt_attrs_by_class(classname):
                ids_for_subpop = dset.ids_for_subpop(classname, attr)
                if len(ids_for_subpop) < min_cnt:
                    continue

                if classname not in acc_by_subpop:
                    acc_by_subpop[classname] = dict()
                correct_for_subpop = [is_correct_by_id[x] for x in ids_for_subpop]
                acc_by_subpop[classname][attr] = np.mean(correct_for_subpop) * 100
    
    return acc_by_class, acc_by_subpop

def accuracy_by_region_and_income_level(is_correct_by_id: Dict[str, bool], dset):
    accs_by_attr = dict()
    df = dset.data_df.reset_index()
    df['is_correct'] = df['img_path'].apply(lambda x: is_correct_by_id[x])
    if 'region' in df.columns:
        for region, sub_df in df.groupby('region'):
            accs_by_attr[region] = sub_df['is_correct'].mean() * 100
    if 'income_level' in df.columns:
        for income_level, sub_df in df.groupby('income_level'):
            accs_by_attr[income_level] = sub_df['is_correct'].mean() * 100
    return accs_by_attr

def accuracy_metrics(pred_classnames: List[str], ids: List[str], dset, verbose: bool=False) -> Dict[str, float]:
    is_correct_by_id = mark_as_correct(pred_classnames, ids, dset)
    
    # Overall accuracy
    acc = np.mean(list(is_correct_by_id.values())) * 100
    # We also want worst class and avg worst subpop accuracy
    acc_by_class, acc_by_subpop = acc_by_class_and_subpop(is_correct_by_id, dset)

    if verbose: 
        print('CLASSES BY ACCURACY')
        sorted_acc_by_class = dict(sorted(acc_by_class.items(), key=lambda x:x[1]))
        for c, class_acc in sorted_acc_by_class.items():
            print(c, class_acc)
        print('\nCLASS and GT SUBPOPS w/ ACCURACIES')
        for c in acc_by_subpop:
            print(c, acc_by_subpop[c])

    worst_class_acc = np.min(list(acc_by_class.values()))

    if dset.has_gt_attrs:
        worst_subpop_accs = [min(subpop_accs_for_class.values()) for subpop_accs_for_class in acc_by_subpop.values()]
        avg_worst_subpop_acc = np.mean(worst_subpop_accs)
        std_worst_subpop_acc = np.std(worst_subpop_accs)
    else:
        avg_worst_subpop_acc = np.nan

    metrics_dict = dict({
        'accuracy': acc, 
        'worst class accuracy': worst_class_acc,
        'average worst subpop accuracy': avg_worst_subpop_acc,
    })

    ### Let's try some less sensitive metrics, by inspecting the bottom x^th percentile instead of just worst class
    sorted_accs = np.sort(list(acc_by_class.values()))
    num_classes_in_bot_xth_percentile = dict({x:max(1, int(np.round(x/100 * len(acc_by_class)))) for x in [1,5,10,20]})
    for x, num in num_classes_in_bot_xth_percentile.items():
        metrics_dict[f'avg worst {x}th percentile class accs'] = np.mean(sorted_accs[:num])

    # And what we really care about is accuracy by region / income_level
    accs_by_attr = accuracy_by_region_and_income_level(is_correct_by_id, dset)
    attrs_we_care_about = _REGIONS+_INCOME_LEVELS
    for attr in attrs_we_care_about:
        if attr in accs_by_attr:
            metrics_dict[attr] = accs_by_attr[attr]
        else:
            metrics_dict[attr] = np.nan

    return metrics_dict

def dollarstreet_worst_subpop_accs_all_attrs(pred_classnames, identifiers, dset):
    # This works for both dollarstreet and geode, where we want to inspect wsa over diff gt attrs
    wsa_by_attr = dict()
    for i, attr in enumerate(dset.allowed_attr_cols):
        dset.set_attribute_column(attr)
        metrics_dict = accuracy_metrics(pred_classnames, identifiers, dset)
        wsa_by_attr[attr] = metrics_dict['average worst subpop accuracy']
    return wsa_by_attr