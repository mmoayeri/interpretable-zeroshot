import argparse
import sys 
if "/private/home/dianeb/pretty-mmmd/" not in sys.path:
    sys.path.append("/private/home/dianeb/pretty-mmmd/")

from datasets.breeds import Breeds
from models.vlm import CLIP
from models.llm import Vicuna

def main_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dsetname', type=str, default='living17')
    parser.add_argument('--vlm', type=str, default='clip_ViT-B/16')
    parser.add_argument('--llm', type=str, default= 'vicuna-13b-v1.3')
    parser.add_argument('--llm_prompts', type=lambda a: tuple(map(str, a.split(','))), nargs='+', default= [('classname', None)])
    parser.add_argument('--vlm_prompts', type=str, nargs='+', default=['a photo of a {}.'])

    return parser.parse_args(args)

def main(args):
    # 1a. Load dataset
    if args.dsetname in ['living17', 'entity30', 'entity13', 'nonliving26']:
        dset = Breeds(dsetname=args.dsetname)
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
    image_embeddings, identifier_idx = vlm.embed_all_images(dset)
    import pdb; pdb.set_trace()
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

if __name__ == '__main__':
    args = main_arguments(sys.argv[1:])
    main(args)