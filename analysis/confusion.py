import argparse
import sys

if "/private/home/dianeb/pretty-mmmd/" not in sys.path:
    sys.path.append("/private/home/dianeb/pretty-mmmd/")

from models.vlm import CLIP
from models.llm import Vicuna
from main import load_dataset
from models.predictor import AverageTopK
from analysis_utils import (
    confusion_matrix_computation,
    return_df_of_avg,
    return_df_C_by_CK,
)
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from models.predictor import init_predictor, init_vlm_prompt_dim_handler
from models.attributer import init_attributer, infer_attrs

def NoneOrStr(value):
    if value == "None":
        return None
    else:
        return str(value)


def main_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsetname", type=str, default="dollarstreet__region")
    parser.add_argument("--vlm", type=str, default="clip_ViT-B/16")
    parser.add_argument("--llm", type=str, default="vicuna-13b-v1.3")
    parser.add_argument(
        "--llm_prompts",
        type=lambda a: tuple(map(NoneOrStr, a.split(","))),
        nargs="+",
        default=[("classname", None)],
    )
    parser.add_argument(
        "--vlm_prompts", type=str, nargs="+", default=["a photo of a {}"]
    )

    return parser.parse_args(args)


def main(args):

    dset = load_dataset(args)

    # STEP 1: get the vanilla model predictions
    if "clip" in args.vlm:
        model_key = args.vlm.split("clip_")[-1]
        vlm = CLIP(model_key=model_key)
    else:
        raise ValueError(
            f"VLM {args.vlm} not recognized. Is it implemented? Should be in in ./models/vlm.py"
        )

    image_embeddings, identifiers = vlm.embed_all_images(dset)
    if "vicuna" in args.llm:
        llm = Vicuna(model_key=args.llm)
    else:
        raise ValueError(
            f"LLM {args.llm} not recognized. Is it implemented? Should be in in ./models/llm.py"
        )
    # STEP 1: confusion matrix of vanilla model
    attributer = init_attributer('vanilla', dset, llm)
    texts_by_subpop_by_class = infer_attrs(dset.classnames, [attributer], args.vlm_prompts)
    text_embeddings_by_subpop_by_cls = vlm.embed_subpopulation_descriptions(texts_by_subpop_by_class)
    vlm_prompt_dim_handler = init_vlm_prompt_dim_handler('average_and_norm_then_stack')
    predictor = AverageTopK(mode='vecs', lamb=1)
    predictions, confidences = predictor.predict(
        image_embeddings, 
        text_embeddings_by_subpop_by_cls, 
        dset.classnames,
        vlm_prompt_dim_handler 
        )

    confusion_mat, per_class_accuracy = confusion_matrix_computation(
        dset, predictions, identifiers)

    dset_prompt = 'llm_kinds'
    # STEP 2: study subpopulations text vectors
    class_mat, class_mat_vanilla = return_df_C_by_CK(
        dset, llm, dset_prompt, vlm, args.vlm_prompts
    )
    return class_mat, class_mat_vanilla, per_class_accuracy


if __name__ == "__main__":
    args = main_arguments(sys.argv[1:])
    main(args)
