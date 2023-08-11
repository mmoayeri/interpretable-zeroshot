import argparse
import sys

if "/private/home/dianeb/pretty-mmmd/" not in sys.path:
    sys.path.append("/private/home/dianeb/pretty-mmmd/")

from datasets.breeds import Breeds
from models.vlm import CLIP
from models.llm import Vicuna
from main import load_dataset
from models.predictor import AverageVecs
from analysis_utils import confusion_matrix_computation


def NoneOrStr(value):
    if value == "None":
        return None
    else:
        return str(value)


def main_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsetname", type=str, default="dollarstreet")
    parser.add_argument("--vlm", type=str, default="clip_ViT-B/16")
    parser.add_argument("--llm", type=str, default="vicuna-13b-v1.3")
    parser.add_argument(
        "--llm_prompts",
        type=lambda a: tuple(map(NoneOrStr, a.split(","))),
        nargs="+",
        default=[("classname", None)],
    )
    parser.add_argument(
        "--vlm_prompts", type=str, nargs="+", default=["a photo of a {}."]
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
    # 3a. Set up LLM object.
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
    predictions, confidences = predictor.predict(
        image_embeddings, text_embeddings_by_cls, dset.classnames
    )

    confusion_mat = confusion_matrix_computation(dset, predictions, identifiers)


if __name__ == "__main__":
    args = main_arguments(sys.argv[1:])
    main(args)
