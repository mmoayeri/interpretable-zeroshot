import numpy as np
import torch
from torch import Tensor
from typing import List, Dict


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
        avg_worst_subpop_acc = -1
    
    metrics_dict = dict({
        'accuracy': acc, 
        'worst class accuracy': worst_class_acc,
        'average worst subpop accuracy': avg_worst_subpop_acc,
        'std dev worst subpop accuracy': std_worst_subpop_acc
    })

    return metrics_dict