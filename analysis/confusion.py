import argparse
import sys

if "/private/home/dianeb/pretty-mmmd/" not in sys.path:
    sys.path.append("/private/home/dianeb/pretty-mmmd/")

from models.vlm import CLIP
from models.llm import Vicuna
from main import load_dataset
from models.predictor import AverageVecs
from analysis_utils import confusion_matrix_computation, return_df_of_avg, return_df_C_by_CK
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
    # attrs_by_class = llm.infer_attrs(dset, [("classname", None)])
    # subpops_by_class = dset.subpop_descriptions_from_attrs(attrs_by_class)
    # text_embeddings_by_cls = vlm.embed_subpopulation_descriptions(
    #     subpops_by_class, ["a photo of a {}."]
    # )

    # predictor = AverageVecs()
    # predictions, _ = predictor.predict(
    #     image_embeddings, text_embeddings_by_cls, dset.classnames
    # )

    # confusion_mat = confusion_matrix_computation(dset, predictions, identifiers)
    # top_mistakes = confusion_mat.stack().nlargest(20)
    # for row in top_mistakes.index:
    #     cl1,cl2 = row
    #     print(f"{cl1} VS {cl2}")
    
    # STEP 2: study subpopulations text vectors
    for llm_answers in [
        [(
            "kinds_regions_incomes",
            "List 16 ways in which a {} appear differently across diverse incomes and geographic regions. Use up to three words per list item.",
        )],
    ]:   
        mat = return_df_C_by_CK(dset, llm, vlm, llm_answers, args.vlm_prompts)
        import pdb; pdb.set_trace()

if __name__ == "__main__":
    args = main_arguments(sys.argv[1:])
    main(args)
