# import argparse
from datasets import Breeds, DollarstreetDataset, GeodeDataset, MITStates
from models.vlm import CLIP, InstructBLIP, BLIP2
from models.llm import Vicuna
from models.predictor import init_predictor, init_vlm_prompt_dim_handler
from models.attributer import init_attributer, infer_attrs
from metrics import accuracy_metrics
from tqdm import tqdm


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

    # 1b. Load VLM
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
    # 2. Get image embeddings
    image_embeddings, identifiers = vlm.embed_all_images(dset)
    # image_embeddings = image_embeddings.detach().cpu() # adding this support in case I run into memory issues next week...

    # 3. Get descriptions per class. Note that we can (and usually do) have multiple descriptions per class,
    #    intended to refer to distinct subpopulations with the class.

    # 3a. Set up LLM object.
    if "vicuna" in args.llm:
        llm = Vicuna(model_key=args.llm)
    else:
        raise ValueError(
            f"LLM {args.llm} not recognized. Is it implemented? Should be in in ./models/llm.py"
        )

    # 3b. Set up attributers
    attributers = []
    for attributer_key in args.attributer_keys:
        attributers.append(init_attributer(attributer_key, dset, llm))

    # 3c. Get 3-dim structure: Dict[classname, Dict[subpop caption, List[subpop caption in various vlm prompt templates]]
    texts_by_subpop_by_class = infer_attrs(
        dset.classnames, attributers, args.vlm_prompts
    )

    # 4. Embed the subpopulation descriptions to build multiple-vec-per-class classification head.
    text_embeddings_by_subpop_by_cls = vlm.embed_subpopulation_descriptions(
        texts_by_subpop_by_class
    )

    # 5. Make predictions
    predictor = init_predictor(args.predictor, args.lamb)

    # 5b. Make VLMPromptDim handler
    vlm_prompt_dim_handler = init_vlm_prompt_dim_handler(args.vlm_prompt_dim_handler)

    # Now we predict
    pred_classnames, confidences = predictor.predict(
        image_embeddings,
        text_embeddings_by_subpop_by_cls,
        dset.classnames,
        vlm_prompt_dim_handler,
    )

    # 6. Compute metrics.
    metric_dict = accuracy_metrics(pred_classnames, identifiers, dset)
    print(f"Dataset: {args.dsetname}, VLM: {args.vlm}, LLM: {args.llm}")
    print(f"Attributer keys: {', '.join(args.attributer_keys)}")
    print(f"VLM prompts: {args.vlm_prompts}")
    print(f"Prediction consolidation strategy: {args.predictor}")
    print(f"VLM Prompt dim handling: {args.vlm_prompt_dim_handler}")
    print(f"Lambda: {args.lamb}")
    print(
        ", ".join(
            [
                f"{metric_name}: {metric_val:.2f}%"
                for (metric_name, metric_val) in metric_dict.items()
            ]
        )
    )

    output_dict = dict(
        {
            "pred_classnames": pred_classnames,
            "identifiers": identifiers,
            "metric_dict": metric_dict,
            "dset": dset,
        }
    )
    return output_dict


# Look away! I don't feel like setting up argparse rn, especially if we might replace it (e.g. w hydra).
# Yes this is ridiculous and yes fixing it is something TODO
class Config(object):
    def __init__(self, d):
        self.__dict__ = d


def test_run_full_pipeline():
    args_as_dict = dict(
        {
            "dsetname": "nonliving26",  # other options: see _ALL_DSETNAMES in constants.py
            "vlm": "blip2",  #'clip_ViT-B/16', # other option: 'blip2'
            "llm": "vicuna-13b-v1.5",
            "attributer_keys": [
                "vanilla",
                "llm_kinds",
            ],  # 'llm_dclip', 'llm_states', 'auto_global', 'income_level'],#, 'region', 'llm_co_occurring_objects', 'llm_backgrounds'],
            "vlm_prompt_dim_handler": "average_and_norm_then_stack",  #'stack_all',
            "vlm_prompts": [
                "USE OPENAI IMAGENET TEMPLATES"
            ],  # other options: ['a photo of a {}.'], ['USE CONDENSED OPENAI TEMPLATES']
            # 'predictor': 'average_top_8_sims',
            "predictor": "chils",
            "lamb": 0.0,  # this parameter only goes into effect when using average_top_k_{sims or vecs}
        }
    )

    args = Config(args_as_dict)
    _ = main(args)


def test_waffle():
    args_as_dict = dict(
        {
            "dsetname": "nonliving26",  # other options: see _ALL_DSETNAMES in constants.py
            "vlm": "clip_ViT-B/16",  # other option: 'blip2'
            "llm": "vicuna-13b-v1.5",
            "attributer_keys": [
                "vanilla",
                "waffle",
            ],  # 'llm_dclip', 'llm_states', 'auto_global', 'income_level'],#, 'region', 'llm_co_occurring_objects', 'llm_backgrounds'],
            "vlm_prompt_dim_handler": "average_and_norm_then_stack",  #'stack_all',
            "vlm_prompts": [
                "USE OPENAI IMAGENET TEMPLATES"
            ],  # other options: ['a photo of a {}.'], ['USE CONDENSED OPENAI TEMPLATES']
            # 'predictor': 'average_top_8_sims',
            "predictor": "average_sims",
            "lamb": 0.0,  # this parameter only goes into effect when using average_top_k_{sims or vecs}
        }
    )

    args = Config(args_as_dict)
    _ = main(args)


if __name__ == "__main__":
    test_run_full_pipeline()
    # test_waffle()
