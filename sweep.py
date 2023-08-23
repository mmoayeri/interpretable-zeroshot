from main import Config, main
from metrics import accuracy_metrics, acc_by_class_and_subpop, mark_as_correct
from constants import _CACHED_DATA_ROOT, _REGIONS, _INCOME_LEVELS
from models.llm import LLM, Vicuna
import pandas as pd
import os
from typing import List, Tuple, Optional
import submitit
import json
from tqdm import tqdm

### Now let's utilize the cluster like real scientists

def single_pipeline_call(args):
    config = Config(args)
    output_dict = main(config)
    metric_dict = output_dict['metric_dict']

    args_and_outputs = args
    for key in metric_dict:
        args_and_outputs[key] = metric_dict[key]
    
    args_and_outputs['pred_classnames'] = output_dict['pred_classnames']
    args_and_outputs['identifiers'] = output_dict['identifiers']

    keys_to_save = ['dsetname', 'attributer_keys', 'vlm_prompts', 'predictor', 'vlm_prompt_dim_handler', 
                    'vlm', 'lamb', 'accuracy', 'worst class accuracy', 'avg worst 20th percentile class accs', 
                    'average worst subpop accuracy', 'pred_classnames', 'identifiers']

    keys_to_save = keys_to_save + _REGIONS + _INCOME_LEVELS

    results_dict = dict({
        key: args_and_outputs[key] for key in keys_to_save
    })
    with open(args['save_path'], 'w') as f:
        json.dump(results_dict, f)


def cluster_sweep(
    all_dsetnames: List[str], 
    all_attributer_keys: List[List[str]], 
    all_vlm_prompts: List[List[str]], 
    all_predictors: List[str], 
    all_vlm_prompt_dim_handlers: List[str],
    all_vlms: List[str],
    all_lambs: List[float],
    log_dir: str,
    ):

    log_dir_path = os.path.join(_CACHED_DATA_ROOT, 'experiments', log_dir)
    log_folder = log_dir_path+'/jobs/%j'
    results_path = log_dir_path + '/results/'
    os.makedirs(results_path, exist_ok=True)
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(timeout_min=180, slurm_partition="learnlab,devlab", gpus_per_node=1, tasks_per_node=1, slurm_constraint='volta32gb,ib4')
    jobs = []
    with executor.batch():
        ctr = 0
        for dsetname in all_dsetnames:
            for attributer_keys in all_attributer_keys:
                for vlm in all_vlms:
                    for vlm_prompts in all_vlm_prompts:
                        for vlm_prompt_dim_handler in all_vlm_prompt_dim_handlers:
                            # In the one-vec-per-class setting, nearly all our predictors are equivalent
                            if attributer_keys == ['vanilla'] and (('average' in vlm_prompt_dim_handler) or (vlm_prompts == ['a photo of a {}'])):
                                curr_predictors = ['average_vecs']  
                            else:
                                curr_predictors = all_predictors
                            for predictor in curr_predictors:
                                curr_all_lambs = all_lambs if 'interpol' in predictor else [1]
                                for lamb in curr_all_lambs:
                                    if predictor == 'chils' and 'vanilla' not in attributer_keys:
                                        continue # such a job would fail
                                    args = dict({
                                        'dsetname': dsetname,
                                        'vlm': vlm,
                                        'llm': 'vicuna-13b-v1.5',
                                        'attributer_keys': attributer_keys,
                                        'vlm_prompts': vlm_prompts,
                                        'vlm_prompt_dim_handler': vlm_prompt_dim_handler,
                                        'predictor': predictor,
                                        'lamb': lamb,
                                        'save_path': f'{results_path}/{ctr}.json'
                                    })
                                    job = executor.submit(single_pipeline_call, args)
                                    jobs.append(job)
                                    ctr += 1


"""
TODO: Shell files for each of these.

What are different kinds of sweeps I want to run. 

1) Get results for CHiLS
    --> fixed predictor, vlm_prompt_dim_handler, attributer_keys
    Cycle through all datasets, vlms, and both vlm_prompt options

    Note: perhaps also worth including ['vanilla', 'groundtruth'] ...

2) Get results for DCLIP
    --> fixed predictor, vlm_prompt_dim_handler, attributer_keys
    Cycle through dsets, vlms, and vlm_prompts

3) Pure (vanilla) baseline on all datasets
    --> fixed predictor, vlm_prompt_dim_handler, attributer_keys
    Cycle through all datasets, vlms, and both vlm_prompts

4) Try vanilla attributer w/ different consolidations
    --> fixed attributer_keys (['vanilla'])
    Cycle through all datasets, vlms, vlm_prompts, and most importantly, predictors 
    [we can cycle over lambda too, but only for interpol...]

5) Try it all... (main comparisons are predictor and attributer keys (amongst our methods) 
   and also how we do compared to baselines; 1-3 above)
"""

def sweep_chils():
    # things that are fixed for CHiLS
    all_predictors = ['chils']
    all_attributer_keys = [['vanilla', 'groundtruth'], ['vanilla', 'llm_kinds_chils']]
    all_vlm_prompt_dim_handlers = ['average_and_norm_then_stack']
    # things we sweep over
    dsetnames = ['living17', 'nonliving26', 'entity13', 'entity30', 'mit_states_0.8', 'mit_states_0.9',
        'dollarstreet__region', 'geode__region']
    vlms = ['clip_ViT-B/16', 'blip2']
    all_vlm_prompts = [['a photo of a {}'], ['USE OPENAI IMAGENET TEMPLATES']]
    lambs = [0.5]

    cluster_sweep(dsetnames, all_attributer_keys, all_vlm_prompts, all_predictors, 
            all_vlm_prompt_dim_handlers, vlms, lambs, log_dir='aug23_chils_2')

def sweep_dclip():
    # things that are fixed for DCLIP
    all_predictors = ['average_sims']
    all_attributer_keys = [['vanilla', 'llm_dclip']]
    all_vlm_prompt_dim_handlers = ['average_and_norm_then_stack']
    # things we sweep over
    dsetnames = ['living17', 'nonliving26', 'entity13', 'entity30', 'mit_states_0.8', 'mit_states_0.9',
                 'dollarstreet__region', 'geode__region']
    vlms = ['clip_ViT-B/16', 'blip2']
    all_vlm_prompts = [['a photo of a {}'], ['USE OPENAI IMAGENET TEMPLATES']]
    lambs = [0.5]

    cluster_sweep(dsetnames, all_attributer_keys, all_vlm_prompts, all_predictors, 
        all_vlm_prompt_dim_handlers, vlms, lambs, log_dir='aug23_dclip_2')

def sweep_vanilla():
    dsetnames = ['living17', 'nonliving26', 'entity13', 'entity30', 'mit_states_0.8', 'mit_states_0.9',
                 'dollarstreet__region', 'geode__region']
    all_attributer_keys = [['vanilla']]
    all_vlm_prompts = [['USE OPENAI IMAGENET TEMPLATES'], ['a photo of a {}']]
    all_predictors = ['average_sims']
    vlms = ['clip_ViT-B/16', 'blip2']
    all_vlm_prompt_dim_handlers = ['average_and_norm_then_stack']
    all_lambs = [1]

    cluster_sweep(dsetnames, all_attributer_keys, all_vlm_prompts, all_predictors, all_vlm_prompt_dim_handlers, vlms, all_lambs,
        log_dir = 'aug23_pure_vanilla_2')


def sweep_all():
    dsetnames = ['dollarstreet__region', 'geode__region', 'living17', 'nonliving26', 'entity13', 'entity30', 'mit_states_0.8', 'mit_states_0.9']
    all_attributer_keys = [['vanilla', 'income_level'], ['vanilla', 'llm_kinds'], ['vanilla', 'llm_kinds_chils'], ['vanilla', 'llm_states'], 
                       ['vanilla', 'llm_kinds_regions_incomes'], ['vanilla', 'llm_dclip']]
    all_vlm_prompts = [['USE OPENAI IMAGENET TEMPLATES'], ['a photo of a {}']]
    all_predictors = ['average_sims', 'average_vecs', 'max_of_max', 'chils', 'interpol_sims_top_2', 'interpol_sims_top_4', 
                      'interpol_sims_top_8', 'interpol_sims_top_16', 'interpol_sims_top_32', 'interpol_sims_top_64']
    vlms = ['clip_ViT-B/16', 'blip2']
    all_vlm_prompt_dim_handlers = ['average_and_norm_then_stack', 'stack_all']
    all_lambs = [0, 0.25, 0.5, 0.75]

    cluster_sweep(dsetnames, all_attributer_keys, all_vlm_prompts, all_predictors, all_vlm_prompt_dim_handlers, vlms, all_lambs,
          log_dir='aug23_all_2')

def sweep_geographic():
    dsetnames = ['dollarstreet__region', 'geode__region']
    all_attributer_keys = [['vanilla', 'income_level'], ['vanilla', 'region'], ['vanilla', 'country'], ['vanilla', 'llm_kinds_regions_incomes']]
    predictors = ['average_sims', 'average_vecs', 'max_of_max', 'interpol_sims_top_2', 'interpol_sims_top_8', 'interpol_sims_top_32', 'chils']
    vlms = ['clip_ViT-B/16', 'blip2']
    all_vlm_prompts = [['USE OPENAI IMAGENET TEMPLATES'], ['a photo of a {}']]
    all_vlm_prompt_dim_handlers = ['average_and_norm_then_stack', 'stack_all']
    all_lambs = [0, 0.25, 0.5, 0.75]

    cluster_sweep(dsetnames, all_attributer_keys, all_vlm_prompts, predictors, all_vlm_prompt_dim_handlers, vlms, all_lambs,
        log_dir = 'aug23_geographic')


if __name__ == '__main__':
    # run_sweep()
    # sweep_chils()
    # sweep_dclip()
    # sweep_vanilla()
    sweep_geographic()
    # sweep_all()