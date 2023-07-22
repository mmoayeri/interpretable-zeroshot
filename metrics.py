import numpy as np
import torch
from torch import Tensor
from typing import List


def mark_as_correct(predictions: Tensor, idx: np.array, dset):
# def mark_as_correct(pred_classnames: List[str], idx: np.array, dset):

    # we pass idx as well just in case there was funny business with the ordering
    # idx was the identifiers seen during caching of image features

    labels = dset.get_labels_in_given_order(idx)

    if dset.is_multilabel:
        is_correct = [pred.item() in labels[i] for i, pred in enumerate(predictions)]
    else:
        is_correct = (predictions == torch.tensor(labels).cuda()).detach().cpu()
    
    return np.array(is_correct)


def acc_by_class_and_subpop(is_correct: np.array, dset):

    acc_by_class, acc_by_subpop = dict(), dict()

    for classname in dset.classes:
        
        idx_in_class = dset.idx_in_class(classname)
        acc_by_class[classname] = np.mean(is_correct[idx_in_class]) * 100

        if dset.has_gt_attrs:
            acc_by_subpop[classname] = dict()
            for attr in dset.gt_attrs_by_class(classname):
                idx_in_subpop = dset.idx_in_subpop(classname, attr)
                acc_by_subpop[classname][attr] = np.mean(is_correct[idx_in_subpop]) * 100
    
    return acc_by_class, acc_by_subpop


def accuracy_metrics(predictions: Tensor, idx: np.array, dset):
    is_correct = mark_as_correct(predictions, idx, dset)
# def accuracy_metrics(pred_classnames: List[str], idx: np.array, dset):
#     is_correct = mark_as_correct(pred_classnames, idx, dset)
    
    # Overall accuracy
    acc = np.mean(is_correct) * 100
    # We also want worst class and avg worst subpop accuracy
    acc_by_class, acc_by_subpop = acc_by_class_and_subpop(is_correct, dset)

    # print(acc_by_class)
    # for c in acc_by_subpop:
    #     print(c, acc_by_subpop[c])
    # print(acc_by_subpop)

    worst_class_acc = np.min(list(acc_by_class.values()))

    if dset.has_gt_attrs:
        worst_subpop_accs = [min(subpop_accs_dict.values()) for subpop_accs_dict in acc_by_subpop.values()]
        avg_worst_subpop_acc = np.mean(worst_subpop_accs)
    else:
        avg_worst_subpop_acc = -1
    
    return acc, worst_class_acc, avg_worst_subpop_acc