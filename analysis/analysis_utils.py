import torch
import pandas as pd
from metrics import mark_as_correct
import numpy as np


def compute_avg_cls_embeddings(text_embeddings_dict):

    avg_cls_embeddings = dict()
    for cls in text_embeddings_dict.keys():
        avg_cls_embeddings[cls] = text_embeddings_dict[cls].mean(0)

    return avg_cls_embeddings


def return_df_of_cosine(dset, llm, vlm, llm_answers, vlm_prompts):

    attrs_by_class = llm.infer_attrs(dset, llm_answers)
    subpops_by_class = dset.subpop_descriptions_from_attrs(attrs_by_class)

    text_embeddings_by_cls = vlm.embed_subpopulation_descriptions(
        subpops_by_class, vlm_prompts
    )
    cossim = torch.nn.CosineSimilarity(dim=2)
    mat = dict()

    for cls1 in dset.classnames:
        mat[cls1] = dict()
        for cls2 in dset.classnames:
            mat[cls1][cls2] = dict()
    class_indices = range(len(dset.classnames))
    for c1 in class_indices:
        cls1 = dset.classnames[c1]
        subpops_cl1 = text_embeddings_by_cls[cls1]
        subpops_names_cl1 = subpops_by_class[cls1]

        for c2 in class_indices:
            cls2 = dset.classnames[c2]
            subpops_cl2 = text_embeddings_by_cls[cls2]
            subpops_names_cl2 = subpops_by_class[cls2]
            if c2 >= c1:
                s = cossim(subpops_cl1.unsqueeze(1), subpops_cl2.unsqueeze(0))
                mat[cls1][cls2] = pd.DataFrame(
                    s.cpu(), index=subpops_names_cl1, columns=subpops_names_cl2
                )
                mat[cls2][cls1] = pd.DataFrame(
                    s.transpose(1, 0).cpu(),
                    index=subpops_names_cl2,
                    columns=subpops_names_cl1,
                )
    return mat


def return_df_C_by_CK(dset, llm, vlm, llm_answers, vlm_prompts):

    vanilla_attrs_by_class = llm.infer_attrs(dset, [("classname", None)])
    vanilla_descriptions = dset.subpop_descriptions_from_attrs(vanilla_attrs_by_class)

    attrs_by_class = llm.infer_attrs(dset, llm_answers)
    subpops_by_class = dset.subpop_descriptions_from_attrs(attrs_by_class)

    text_embeddings_vanilla = vlm.embed_subpopulation_descriptions(
        vanilla_descriptions, ["a photo of a {}"]
    )

    text_embeddings_by_cls = vlm.embed_subpopulation_descriptions(
        subpops_by_class, vlm_prompts
    )
    cossim = torch.nn.CosineSimilarity(dim=2)
    mat = dict()
    mat_vanilla = dict()

    for c1 in range(len(dset.classnames)):
        cls1 = dset.classnames[c1]
        baseline_cl1 = text_embeddings_vanilla[cls1]

        for c2 in range(len(dset.classnames)):
            cls2 = dset.classnames[c2]
            subpops_cl2 = text_embeddings_by_cls[cls2]
            subpops_names_cl2 = subpops_by_class[cls2]

            s = cossim(baseline_cl1.unsqueeze(1), subpops_cl2.unsqueeze(0))
            mi = [(cls2, i) for i in subpops_names_cl2]
            index = pd.MultiIndex.from_tuples(mi)
            if cls1 not in mat.keys():
                mat[cls1] = pd.DataFrame(s.cpu(), index=[cls1], columns=index)
            else:
                tmp = pd.DataFrame(s.cpu(), index=[cls1], columns=index)
                mat[cls1] = pd.concat([mat[cls1], tmp], axis=1)

            baseline_cl2 = text_embeddings_vanilla[cls2]
            s = cossim(baseline_cl1.unsqueeze(1), baseline_cl2.unsqueeze(0))
            if cls1 not in mat_vanilla.keys():
                mat_vanilla[cls1] = pd.DataFrame(s.cpu(), index=[cls1], columns=[cls2])
            else:
                tmp = pd.DataFrame(s.cpu(), index=[cls1], columns=[cls2])
                mat_vanilla[cls1] = pd.concat([mat_vanilla[cls1], tmp], axis=1)
    mat = pd.concat(mat, axis=0)
    class_subpop_mat = mat.droplevel(level=0).groupby(level=0, axis=1).mean()
    for cls in class_subpop_mat.index:
        class_subpop_mat.loc[cls][cls] = 0.0  # set to 0 the same to same class

    class_mat = class_subpop_mat.mean(
        axis=1
    )  # average over subpops, per class. This gives CxC matrix
    mat_vanilla = pd.concat(mat_vanilla).droplevel(level=0)
    for cls in mat_vanilla.index:
        mat_vanilla.loc[cls][cls] = 0.0  # set to 0 the same to same class

    class_mat_vanilla = mat_vanilla.mean(axis=1)
    import pdb

    pdb.set_trace()


def return_df_of_avg(dset, llm, vlm, llm_answers, vlm_prompts):

    attrs_by_class = llm.infer_attrs(dset, llm_answers)
    subpops_by_class = dset.subpop_descriptions_from_attrs(attrs_by_class)

    text_embeddings_by_cls = vlm.embed_subpopulation_descriptions(
        subpops_by_class, vlm_prompts
    )

    avg_per_class = compute_avg_cls_embeddings(text_embeddings_by_cls)

    df_avg = pd.DataFrame(
        dict([(k, v.cpu()) for k, v in avg_per_class.items()])
    ).transpose()

    return df_avg


def confusion_matrix_computation(dset, predictions, identifiers):
    is_correct_by_id = mark_as_correct(predictions, identifiers, dset)
    df_correct = pd.DataFrame(is_correct_by_id, index=["correct"]).transpose()
    df_predictions = pd.DataFrame(predictions, index=identifiers, columns=["pred"])
    df = df_correct.join(df_predictions)

    mat = dict()
    for cls1 in dset.classnames:
        mat[cls1] = dict()
        cls_identifiers = dset.ids_for_class(cls1)
        df_cls = df.loc[cls_identifiers]
        n_correct = (df_cls["correct"] == True).sum()
        mat[cls1][cls1] = n_correct
        df_mistakes = df_cls[df_cls["correct"] == False]
        predicted_classes = np.unique(df_mistakes["pred"])
        for cls2 in dset.classnames:
            if cls2 in predicted_classes:
                # if cls1 == 'toilet' and cls2 == 'using toilet':
                #     import pdb; pdb.set_trace()
                confusion_cls2_cls1 = (df_mistakes["pred"] == cls2).sum()
                mat[cls1][cls2] = confusion_cls2_cls1
            else:
                mat[cls1][cls2] = 0
    return pd.DataFrame.from_dict(mat, orient="index")
