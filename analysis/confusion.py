import argparse
import sys

if "/private/home/dianeb/pretty-mmmd/" not in sys.path:
    sys.path.append("/private/home/dianeb/pretty-mmmd/")

from models.vlm import CLIP
from models.llm import Vicuna
from main import load_dataset
from models.predictor import AverageVecs
from analysis_utils import confusion_matrix_computation, compute_avg_cls_embeddings
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


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
        "--vlm_prompts", type=str, nargs="+", default=["USE OPENAI IMAGENET TEMPLATES"]
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

    attrs_by_class = llm.infer_attrs(dset, [("classname", None)])
    subpops_by_class = dset.subpop_descriptions_from_attrs(attrs_by_class)
    text_embeddings_by_cls = vlm.embed_subpopulation_descriptions(
        subpops_by_class, ["a photo of a {}."]
    )

    predictor = AverageVecs()
    predictions, _ = predictor.predict(
        image_embeddings, text_embeddings_by_cls, dset.classnames
    )

    confusion_mat = confusion_matrix_computation(dset, predictions, identifiers)

    # STEP 2: study subpopulations text vectors

    # CLASS ONLY
    # args.llm_prompts = [("classname", None)]

    # # ORACLE
    # args.llm_prompts = [("classname", None), ("groundtruth", None)]

    # CLASS + LLM
    args.llm_prompts = [
        ("classname", None), 
        (
            "kinds_regions_incomes",
            "List 16 ways in which a {} appear differently across diverse incomes and geographic regions. Use up to three words per list item.",
        )
    ]

    attrs_by_class = llm.infer_attrs(dset, args.llm_prompts)
    subpops_by_class = dset.subpop_descriptions_from_attrs(attrs_by_class)
    
    text_embeddings_by_cls = vlm.embed_subpopulation_descriptions(
        subpops_by_class, args.vlm_prompts
    )

    avg_per_class = compute_avg_cls_embeddings(text_embeddings_by_cls)

    df_avg = pd.DataFrame(
        dict([(k, v.cpu()) for k, v in avg_per_class.items()])
    ).transpose()
    cossim_avg = cosine_similarity(df_avg)
    print(f"Average cosine similarity over all classes is {cossim_avg.mean()}")


if __name__ == "__main__":
    args = main_arguments(sys.argv[1:])
    main(args)
