import torch
from sklearn.metrics import confusion_matrix
import pandas as pd
from metrics import mark_as_correct


def compute_avg_cls_embeddings(text_embeddings_dict):

    avg_cls_embeddings = dict()
    for cls in text_embeddings_dict.keys():
        avg_cls_embeddings[cls] = text_embeddings_dict[cls].mean(0)

    return avg_cls_embeddings


def confusion_matrix_computation(dset, predictions, identifiers):
    is_correct_by_id = mark_as_correct(predictions, identifiers, dset)
    df_correct = pd.DataFrame(is_correct_by_id, index=["correct"]).transpose()
    df_predictions = pd.DataFrame(predictions, index=identifiers, columns=["pred"])
    df = df_correct.join(df_predictions)
    mat = torch.zeros(size=(len(dset.classnames), len(dset.classnames)))

    for c, cls in enumerate(dset.classnames):
        cls_identifiers = dset.ids_for_class(cls)
        df_cls = df.loc[cls_identifiers]
        str_labels = [cls] * len(df_cls)
        str_predictions = df_cls["pred"]
        # Take only the c-th row (all labels are cls)
        mat[c, :] += confusion_matrix(
            str_labels, str_predictions, labels=dset.classnames
        )[c]

    return mat
