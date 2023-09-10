import argparse
import sys

if "/private/home/dianeb/pretty-mmmd/" not in sys.path:
    sys.path.append("/private/home/dianeb/pretty-mmmd/")

from datasets import Breeds, DollarstreetDataset, GeodeDataset, MITStates
from models.vlm import CLIP, InstructBLIP, BLIP2
from models.llm import Vicuna
from models.predictor import init_predictor, init_vlm_prompt_dim_handler
from models.attributer import init_attributer, infer_attrs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from metrics import mark_as_correct, acc_by_class_and_subpop


def return_df_CK_by_CK(dset, vlm):

    # 3a. Set up LLM object.
    llm = Vicuna(model_key = args.llm)
    # WARNING : this is uper slow if we have tons of subpop
    attributer = init_attributer("llm_kinds", dset, llm=None)

    texts_by_subpop_by_class = infer_attrs(
        dset.classnames, [attributer], ["A photo of a {}"]
    )
    text_embeddings_by_subpop_by_cls = vlm.embed_subpopulation_descriptions(
        texts_by_subpop_by_class
    )

    vlm_handler = init_vlm_prompt_dim_handler("stack_all")
    text_embeddings_by_cls, subpops = vlm_handler.convert_to_embeddings_by_cls(
        text_embeddings_by_subpop_by_cls
    )
    cossim = torch.nn.CosineSimilarity(dim=2)
    mat = dict()
    for c1 in range(len(dset.classnames)):
        cls1 = dset.classnames[c1]
        for c2 in range(len(dset.classnames)):
            cls2 = dset.classnames[c2]

            s = cossim(
                text_embeddings_by_cls[cls1].unsqueeze(1),
                text_embeddings_by_cls[cls2].unsqueeze(0),
            )
            # set to 0 the already parsed ones
            if c2 <= c1:
                s[:] = 0.0

            df = pd.DataFrame(s.cpu(), index=subpops[cls1], columns=subpops[cls2])

            if cls1 not in mat.keys():
                mat[cls1] = df
            else:
                mat[cls1] = pd.concat([mat[cls1], df], axis=1)
    mat = pd.concat(mat).droplevel(level=0)
    for c in range(len(dset.classnames)):
        cls = dset.classnames[c]
        for subpop1 in subpops[cls]:
            for subpop2 in subpops[cls]:
                mat.loc[subpop1][
                    subpop2
                ] = 0.0  # set to 0 the same to same class we want to compare confusing subpop of different classes)
    return mat, text_embeddings_by_subpop_by_cls, subpops


def main(args):

    ### Experiment mode: we want the output for a whole dataset, not just a single image
    # 1a. Load dataset
    if args.dsetname in ["living17", "entity30", "entity13", "nonliving26"]:
        dset = Breeds(dsetname=args.dsetname)
    elif "dollarstreet" in args.dsetname:
        attr = args.dsetname.split("__")[-1]
        dset = DollarstreetDataset(attr_col=attr)
    elif "geode" in args.dsetname:
        attr = args.dsetname.split("__")[-1]
        dset = GeodeDataset(attr_col=attr)
    elif "mit_states" in args.dsetname:
        thresh = float(args.dsetname.split("mit_states_")[-1])
        dset = MITStates(max_allowable_sim_of_classnames=thresh)
    else:
        raise ValueError(
            f"Dataset {args.dsetname} not recognized. Is it implemented? Should be in ./dataset/ directory."
        )

    if "clip" in args.vlm:
        model_key = args.vlm.split("clip_")[-1]
        vlm = CLIP(model_key=model_key)
    elif "instruct_blip" in args.vlm:
        vlm = InstructBLIP()
    elif args.vlm == "blip2":
        vlm = BLIP2()
    else:
        raise ValueError(
            f"VLM {args.vlm} not recognized. Is it implemented? Should be in in ./models/vlm.py"
        )

    ###### IMAGES EMBEDDINGS
    image_embeddings, identifiers = vlm.embed_all_images(dset)

    ###### PCA
    df_images = pd.DataFrame(image_embeddings.cpu(), index=identifiers)
    # pca = PCA(n_components=2)
    # TODO: random state setting?
    pca = TSNE(n_components=2, random_state=0)
    #### IF WE DO PCA/TSNE WITH EVERYBODY
    df_scaled = StandardScaler().fit_transform(df_images)
    pca_features = pca.fit_transform(df_scaled)
    pca_df = pd.DataFrame(data=pca_features, columns=["PC1", "PC2"], index=identifiers)

    # TODO: keep "a kind of" in prompts? cf bow behavior
    mat, text_embeddings_by_subpop_by_cls, subpops = return_df_CK_by_CK(dset, vlm)

    ##### DO THE PREDICTIONS
    ###### VANILLA MODEL
    vlm_prompt_dim_handler = init_vlm_prompt_dim_handler(args.vlm_prompt_dim_handler)
    vanilla_predictor = init_predictor("average_vecs", 1.0)
    vanilla_pred_classnames, _ = vanilla_predictor.predict(
        image_embeddings,
        text_embeddings_by_subpop_by_cls,
        dset.classnames,
        vlm_prompt_dim_handler,
    )
    vanilla_is_correct_by_id = mark_as_correct(
        vanilla_pred_classnames, identifiers, dset
    )
    _, vanilla_acc = acc_by_class_and_subpop(vanilla_is_correct_by_id, dset, min_cnt=10)

    ###### OUR MODEL (TODO: wouldn't max of max be more illustrative?) What is interpol between max of max and avg sims?
    predictor = init_predictor("average_top_1_vecs", 0.5)
    pred_classnames, _ = predictor.predict(
        image_embeddings,
        text_embeddings_by_subpop_by_cls,
        dset.classnames,
        vlm_prompt_dim_handler,
    )
    is_correct_by_id = mark_as_correct(pred_classnames, identifiers, dset)
    _, our_acc = acc_by_class_and_subpop(is_correct_by_id, dset, min_cnt=10)

    idxs = np.argsort(mat.to_numpy().ravel())[-10:]
    rows, cols = idxs // mat.shape[0], idxs % mat.shape[0]
    for i in range(10):
        str1 = mat.index[rows[i]]
        str2 = mat.index[cols[i]]
        print(str1, str2)

        cls1 = str1.split(" ")[-1]
        cls2 = str2.split(" ")[-1]
        subpop1 = str1.split(",")[0]
        subpop2 = str2.split(",")[0]
        print(vanilla_acc[cls1][subpop1], vanilla_acc[cls2][subpop2])
        print(our_acc[cls1][subpop1], our_acc[cls2][subpop2])

        cls1_identifiers = dset.ids_for_class(cls1)
        cls2_identifiers = dset.ids_for_class(cls2)
        df_cls1 = pca_df.loc[cls1_identifiers]
        df_cls2 = pca_df.loc[cls2_identifiers]
        df_cls1["target"] = cls1
        df_cls2["target"] = cls2
        df = pd.concat([df_cls1, df_cls2], axis=0)

        df_attr = pd.DataFrame(dset.data_df["attr"])
        df_confusion = dset.data_df["attr"].isin([subpop1, subpop2])
        df = pd.concat([df, df_attr], axis=1)
        df = df.rename(columns={"attr": "subpop"})
        df = pd.concat([df, df_confusion], axis=1)
        df_two_classes = df[df["target"].isin([cls1, cls2])]
        # sns.set(rc={"figure.figsize": (10, 10)})
        # sns.scatterplot(
        #     x="PC1", y="PC2", data=df_two_classes, hue="target", style="attr",
        # )
        # plt.title("2D PCA Graph")
        # plt.legend(bbox_to_anchor=(1.25, 1), borderaxespad=0)
        # plt.show()

        sns.set(rc={"figure.figsize": (10, 10)})
        ax = sns.scatterplot(
            x="PC1",
            y="PC2",
            hue="target",
            data=df_two_classes[df_two_classes.attr == False],
            alpha=0.5,
        )
        sns.scatterplot(
            x="PC1",
            y="PC2",
            hue="target",
            data=df_two_classes[df_two_classes.attr == True],
            alpha=1.0,
            ax=ax,
        )
        ax.legend(bbox_to_anchor=(1.25, 1), borderaxespad=0)
        plt.show()
    ##### IF WE DO IT WITH WOLF AND FOX ONLY
    # df_images = pd.concat([df_images.loc[cls1_identifiers],df_images.loc[cls2_identifiers]],axis=0)
    # df_scaled = StandardScaler().fit_transform(df_images)
    # pca_features = pca.fit_transform(df_scaled)

    # df = pd.DataFrame(
    #     data=pca_features,
    #     columns=['PC1', 'PC2'],
    #     index=cls1_identifiers.union(cls2_identifiers))
    # df_target = pd.DataFrame([cls1]*len(cls1_identifiers), columns=['target'], index=cls1_identifiers)
    # df_target = pd.concat([df_target,pd.DataFrame([cls2]*len(cls2_identifiers), columns=['target'], index=cls2_identifiers)],axis=0)
    # df = pd.concat([df,df_target],axis=1)


if __name__ == "__main__":
    main(sys.argv[1:])
