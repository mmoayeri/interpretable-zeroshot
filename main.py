# import argparse
from datasets import Breeds, DollarstreetDataset
from models.vlm import CLIP
from models.llm import Vicuna
from models.predictor import init_predictor, init_vlm_prompt_dim_handler
from models.attributer import init_attributer, infer_attrs
from metrics import accuracy_metrics

def main(args):
    ### Experiment mode: we want the output for a whole dataset, not just a single image
    # 1a. Load dataset
    if args.dsetname in ['living17', 'entity30', 'entity13', 'nonliving26']:
        dset = Breeds(dsetname=args.dsetname)
    elif 'dollarstreet' in args.dsetname:
        attr = args.dsetname.split('__')[-1]
        dset = DollarstreetDataset(attr_col = attr)
    elif args.dsetname == 'mit_states':
        dset = MITStates()
    else:
        raise ValueError(f'Dataset {args.dsetname} not recognized. Is it implemented? Should be in ./dataset/ directory.')


    # 1b. Load VLM
    if 'clip' in args.vlm:
        model_key = args.vlm.split('clip_')[-1]
        vlm = CLIP(model_key=model_key)
    else:
        raise ValueError(f'VLM {args.vlm} not recognized. Is it implemented? Should be in in ./models/vlm.py')
    # 2. Get image embeddings
    image_embeddings, identifiers = vlm.embed_all_images(dset)


    # 3. Get descriptions per class. Note that we can (and usually do) have multiple descriptions per class, 
    #    intended to refer to distinct subpopulations with the class. 

    # 3a. Set up LLM object.
    if 'vicuna' in args.llm:
        llm = Vicuna(model_key = args.llm)
    else:
        raise ValueError(f'LLM {args.llm} not recognized. Is it implemented? Should be in in ./models/llm.py')

    # 3b. Set up attributers
    attributers = []
    for attributer_key in args.attributer_keys:
        attributers.append(init_attributer(attributer_key, dset, llm))
    
    # 3c. Get 3-dim structure: Dict[classname, Dict[subpop caption, List[subpop caption in various vlm prompt templates]]
    texts_by_subpop_by_class = infer_attrs(dset.classnames, attributers, args.vlm_prompts)

    # 4. Embed the subpopulation descriptions to build multiple-vec-per-class classification head.
    text_embeddings_by_subpop_by_cls = vlm.embed_subpopulation_descriptions(texts_by_subpop_by_class)

    # 5. Make predictions
    predictor = init_predictor(args.predictor, args.lamb)

    # 5b. Make VLMPromptDim handler
    vlm_prompt_dim_handler = init_vlm_prompt_dim_handler(args.vlm_prompt_dim_handler)

    # Now we predict
    pred_classnames, confidences = predictor.predict(
        image_embeddings, 
        text_embeddings_by_subpop_by_cls, 
        dset.classnames,
        vlm_prompt_dim_handler 
        )


    # 6. Compute metrics. 
    metric_dict = accuracy_metrics(pred_classnames, identifiers, dset)
    print(f'Dataset: {args.dsetname}, VLM: {args.vlm}, LLM: {args.llm}')
    print(f"Attributer keys: {', '.join(args.attributer_keys)}")
    print(f'VLM prompts: {args.vlm_prompts}')
    print(f'Prediction consolidation strategy: {args.predictor}')
    print(f'VLM Prompt dim handling: {args.vlm_prompt_dim_handler}')
    print(', '.join([f'{metric_name}: {metric_val:.2f}%' for (metric_name, metric_val) in metric_dict.items()]))
    
    output_dict = dict({'pred_classnames':pred_classnames, 'identifiers': identifiers, 'metric_dict':metric_dict, 'dset': dset})
    return output_dict

# Look away! I don't feel like setting up argparse rn, especially if we might replace it (e.g. w hydra).
# Yes this is ridiculous and yes fixing it is something TODO
class Config(object):
    def __init__(self, d):
        self.__dict__ = d

def test_run_full_pipeline():
    # These args will be a pure vanilla case
    args_as_dict = dict({
        'dsetname': 'dollarstreet__income_group',
        # 'dsetname': 'living17',
        'vlm': 'clip_ViT-B/16',
        'llm': 'vicuna-13b-v1.5',
        # 'llm_prompts': [('classname', None)],
        'attributer_keys': ['vanilla', 'income_level', 'country', 'region'], #'income_level'],
        'vlm_prompt_dim_handler': 'average_and_norm_then_stack',
        'vlm_prompts': ['a photo of a {}.'],
        'predictor': 'average_top_2',
        'lamb': 0.5
    })

    ## To get our oracle case, you can uncomment this
    # args_as_dict['llm_prompts'] = [('classname', None), ('groundtruth', None)]
    ### Or to use LLM prompts, you can try one of these!
    # args_as_dict['llm_prompts'] = [('classname', None), ('kinds', 'List 16 different kinds of {}. Only use up to three words per list item.')]
    # args_as_dict['llm_prompts'] = [('classname', None), ('kinds_regions_incomes', 'List 16 different ways in which a {} may appear across diverse geographic regions and incomes. Only use up to three words per list item.')]

    ### And to play with prediction consolidation strategy, you can do one of these
    # args_as_dict['predictor'] = 'max_of_max'
    # args_as_dict['predictor'] = 'average_top_6'

    ### You can also use the ImageNet VLM prompts that were handcrafted for CLIP
    args_as_dict['vlm_prompts'] = ['USE OPENAI IMAGENET TEMPLATES']
    # args_as_dict['vlm_prompts'] = ['USE CONDENSED OPENAI TEMPLATES']

    args = Config(args_as_dict)
    _ = main(args)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument()


    test_run_full_pipeline()