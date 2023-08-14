# import argparse
from datasets import Breeds, DollarstreetDataset
from models.vlm import CLIP
from models.llm import Vicuna
from models.predictor import MaxOfMax, AverageSims, AverageVecs, AverageTopKSims, LinearInterpolationAverageSimsTopK, LinearInterpolationAverageVecsTopk
from metrics import accuracy_metrics

def load_dataset(args):

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

    return dset

def main(args):
    '''
    Needed arguments for a single run:
    - dataset name : str
    - LLM name : str
    - VLM name : str
    - LLM prompt(s) : List(Tuple(str, str)) [e.g. ('kinds', 'List different kinds of {clsname}')]
    - VLM prompt/template (s): List(str)
    - Attribute refinement : str (bool?)
    - Prediction consolidation method : str
    - Metrics : List(str)


    Full pipeline:
    1. Set up: dataset, vlm

    2. Compute image embeddings for all images in dataset using vlm.image_encoder
    
       Caching: Save image embeddings to cached/image_embeddings/{vlm}/{dataset}.pkl

    3. Get subpopulation names (list of strings) per class, as a dictionary with 
    key-value pairs of (classname, list of subpopulation names)
        If LLM prompt is empty:
            We are in vanilla setting. Resulting dict: for each class, we have a 
            one element list (just the classname)
        Elif LLM prompt is 'GT':
            We are in oracle setting. Resulting dict: for each class, we have a 
            list of the ground truth subpopulations for that class. 
        Else:
            a. Set up LLM
            b. Resulting dict: for each class, we have a list that contains responses 
            for LLM_prompt.format(clsname) aggregated over every LLM_prompt (so we can
            ask LLM for different kinds of attrs).
            c. Refine attributes if attribute refinement is turned on

            Caching: Save LLM outputs to cached/subpops_from_llm/{llm_prompt}/{llm}/{dataset}.json 
            for each LLM prompt. 

    3.5 instance based adaptivity

    4. Embed subpopulations to VLM space, resulting in dictionary of (cls_ind, tensor of subpop embeddings)
    key-value pairs. Note that we embed a string s by averaging VLM.encode_text(VLM_prompt.format(s)) over VLM_prompts
    
    5. Make pred_classnames, following prediction consolidation strategy. Returns prediction and confidence per image.

    6. Compute metrics. Inputs are dataset object and pred_classnames (and confidences, for calibration error)

    '''

    ### Experiment mode: we want the output for a whole dataset, not just a single image
    # 1a. Load dataset
    dset = load_dataset(args)

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
    # 3b. Get LLM-inferred (or groundtruth / classname-only) attributes
    attrs_by_class = llm.infer_attrs(dset, args.llm_prompts)
    # 3c. Use dataset object to create subpopulation descriptions for each attribute
    subpops_by_class = dset.subpop_descriptions_from_attrs(attrs_by_class)


    # 4. Embed the subpopulation descriptions to build multiple-vec-per-class classification head.
    text_embeddings_by_cls = vlm.embed_subpopulation_descriptions(subpops_by_class, args.vlm_prompts)


    # 5. Make predictions
    if args.predictor == 'max_of_max':
        predictor = MaxOfMax()
    elif args.predictor == 'average_vecs':
        predictor = AverageVecs()
    elif args.predictor == 'average_sims':
        predictor = AverageSims()
    elif 'average_top_' in args.predictor:
        k = int(args.predictor.split('_')[-1])
        predictor = AverageTopKSims(k=k)
    elif 'interpol_vecs_top_' in args.predictor:
        k = int(args.predictor.split('_')[-1])
        predictor = LinearInterpolationAverageVecsTopk(k=k, lamb=args.lamb)
    elif 'interpol_sims_top_' in args.predictor:
        k = int(args.predictor.split('_')[-1])
        predictor = LinearInterpolationAverageSimsTopK(k=k, lamb=args.lamb)
    else:
        raise ValueError(f'Predictor {args.predictor} not recognized. Is it implemented? Should be in ./models/predictor.py')

    pred_classnames, confidences = predictor.predict(image_embeddings, text_embeddings_by_cls, dset.classnames)


    # 6. Compute metrics. 
    metric_dict = accuracy_metrics(pred_classnames, identifiers, dset)
    print(f'Dataset: {args.dsetname}, VLM: {args.vlm}, LLM: {args.llm}')
    print('LLM prompts (in (nickname, full prompt) format): \n' + "\n".join([f'({k}, {v})' for (k,v) in args.llm_prompts]))
    print(f'VLM prompts: {args.vlm_prompts}')
    print(f'Prediction consolidation strategy: {args.predictor}')
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
        'dsetname': 'dollarstreet__country.name',
        # 'dsetname': 'living17',
        'vlm': 'clip_ViT-B/16',
        'llm': 'vicuna-13b-v1.5',
        'llm_prompts': [('classname', None)],
        'vlm_prompts': ['a photo of a {}.'],
        'predictor': 'interpol_sims_top_2',
        'lamb': 0.5
    })

    ## To get our oracle case, you can uncomment this
    args_as_dict['llm_prompts'] = [('classname', None), ('groundtruth', None)]
    ### Or to use LLM prompts, you can try one of these!
    # args_as_dict['llm_prompts'] = [('classname', None), ('kinds', 'List 16 different kinds of {}. Only use up to three words per list item.')]
    # args_as_dict['llm_prompts'] = [('classname', None), ('kinds_regions_incomes', 'List 16 different ways in which a {} may appear across diverse geographic regions and incomes. Only use up to three words per list item.')]

    ### And to play with prediction consolidation strategy, you can do one of these
    # args_as_dict['predictor'] = 'max_of_max'
    args_as_dict['predictor'] = 'average_top_6'

    ### You can also use the ImageNet VLM prompts that were handcrafted for CLIP
    args_as_dict['vlm_prompts'] = ['USE OPENAI IMAGENET TEMPLATES']
    # args_as_dict['vlm_prompts'] = ['USE CONDENSED OPENAI TEMPLATES']

    args = Config(args_as_dict)
    _ = main(args)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument()


    test_run_full_pipeline()