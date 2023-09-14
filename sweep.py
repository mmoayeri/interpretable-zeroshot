from main import Config, main
from metrics import accuracy_metrics, acc_by_class_and_subpop, mark_as_correct
from constants import _CACHED_DATA_ROOT, _REGIONS, _INCOME_LEVELS, _INPUTS, _METRICS
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

    # keys_to_save = ['dsetname', 'attributer_keys', 'vlm_prompts', 'predictor', 'vlm_prompt_dim_handler', 
    #                 'vlm', 'lamb', 'accuracy', 'worst class accuracy', 'avg worst 20th percentile class accs', 
    #                 'average worst subpop accuracy', 'pred_classnames', 'identifiers']

    # keys_to_save = keys_to_save + _REGIONS + _INCOME_LEVELS
    keys_to_save = _INPUTS + _METRICS

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
    executor.update_parameters(timeout_min=180, slurm_partition="learnlab,devlab", mem_gb=80, gpus_per_node=1, tasks_per_node=1, slurm_constraint='volta32gb,ib4')
    # executor.update_parameters(timeout_min=180, slurm_partition="cml-dpart", slurm_qos="cml-default", slurm_account="cml-sfeizi", mem_gb=32, gpus_per_node=1, tasks_per_node=1)#, slurm_constraint='volta32gb,ib4')
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
                                curr_predictors = ['average_sims']  
                            else:
                                curr_predictors = all_predictors
                            for predictor in curr_predictors:
                                if 'average_top' in predictor or 'knn' in predictor:
                                    curr_all_lambs = all_lambs
                                # these next two don't affect the run; but makes more sense (lamb=1 corresponds to average_vecs/sims, lamb=0 is more like max of max)
                                # in either case, we only loop over one value of lamb bc it is never used (you get identical runs w/ diff lambs)
                                elif predictor == 'average_sims' or predictor == 'average_vecs':
                                    curr_all_lambs = [1]
                                else:
                                    curr_all_lambs = [0]
                                for lamb in curr_all_lambs:
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
    all_attributer_keys = [['vanilla', 'llm_kinds_chils']]
    all_vlm_prompt_dim_handlers = ['average_and_norm_then_stack']
    # things we sweep over
    dsetnames = ['living17', 'nonliving26', 'entity13', 'entity30', 'mit_states_0.8', 'mit_states_0.9',
        'dollarstreet__region', 'geode__region']
    vlms = ['clip_ViT-B/16', 'blip2']
    # all_vlm_prompts = [['a photo of a {}'], ['USE OPENAI IMAGENET TEMPLATES']]
    all_vlm_prompts = [['USE OPENAI IMAGENET TEMPLATES']]
    lambs = [0]

    cluster_sweep(dsetnames, all_attributer_keys, all_vlm_prompts, all_predictors, 
            all_vlm_prompt_dim_handlers, vlms, lambs, log_dir='sep13_chils')
            # all_vlm_prompt_dim_handlers, vlms, lambs, log_dir='sept5_chils_3')
            # all_vlm_prompt_dim_handlers, vlms, lambs, log_dir='aug29_chils')

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
    lambs = [1]

    cluster_sweep(dsetnames, all_attributer_keys, all_vlm_prompts, all_predictors, 
        all_vlm_prompt_dim_handlers, vlms, lambs, log_dir='sep13_dclip')
        # all_vlm_prompt_dim_handlers, vlms, lambs, log_dir='aug23_dclip_2')

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
        log_dir = 'sep13_pure_vanilla')

def sweep_waffle():
    dsetnames = ['living17', 'nonliving26', 'entity13', 'entity30', 'mit_states_0.8', 'mit_states_0.9',
                 'dollarstreet__region', 'geode__region']
    all_attributer_keys = [['vanilla', 'waffle']]
    all_vlm_prompts = [['USE OPENAI IMAGENET TEMPLATES'], ['a photo of a {}']]
    all_predictors = ['average_sims']
    vlms = ['clip_ViT-B/16', 'blip2']
    all_vlm_prompt_dim_handlers = ['average_and_norm_then_stack']
    all_lambs = [1]

    for i in range(5):
        cluster_sweep(dsetnames, all_attributer_keys, all_vlm_prompts, all_predictors, all_vlm_prompt_dim_handlers, vlms, all_lambs,
            log_dir = f'sep13_waffle_{i+1}')

def stack_all_vlm_prompts():
    dsetnames = ['living17', 'nonliving26', 'entity13', 'entity30', 'mit_states_0.8', 'mit_states_0.9',
                 'dollarstreet__region', 'geode__region']
    all_attributer_keys = [['vanilla']]
    all_vlm_prompts = [['USE OPENAI IMAGENET TEMPLATES'], ['a photo of a {}']]
    all_predictors = ['average_top_8_sims', 'average_top_4_sims', 'average_top_16_sims']
    vlms = ['clip_ViT-B/16', 'blip2']
    all_vlm_prompt_dim_handlers = ['stack_all']
    all_lambs = [0, 0.25, 0.5]

    cluster_sweep(dsetnames, all_attributer_keys, all_vlm_prompts, all_predictors, all_vlm_prompt_dim_handlers, vlms, all_lambs,
        log_dir = 'sep13_stack_vlm_prompts')


def our_best():
    dsetnames = ['living17', 'nonliving26', 'entity13', 'entity30', 'mit_states_0.8', 'mit_states_0.9',
                 'dollarstreet__region', 'geode__region']
    all_attributer_keys = [['vanilla', 'auto_global', 'income_level', 'llm_dclip', 'country', 'llm_states', 'llm_kinds', 'region', 'llm_co_occurring_objects']]
    all_vlm_prompts = [['USE OPENAI IMAGENET TEMPLATES']]
    all_predictors = ['average_top_8_sims', 'average_top_16_sims']
    vlms = ['clip_ViT-B/16', 'blip2']
    all_vlm_prompt_dim_handlers = ['average_and_norm_then_stack']
    all_lambs = [0, 0.25]

    cluster_sweep(dsetnames, all_attributer_keys, all_vlm_prompts, all_predictors, all_vlm_prompt_dim_handlers, vlms, all_lambs,
        log_dir = 'sep14_our_bests')


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


def new_llm_queries():
    dsetnames = ['dollarstreet__region', 'geode__region', 'living17', 'nonliving26', 'entity13', 'entity30', 'mit_states_0.8', 'mit_states_0.9']
    # all_attributer_keys = [['vanilla', 'llm_backgrounds'], ['vanilla', 'llm_co_occurring_objects'], ['vanilla', 'llm_kinds_chils', 'llm_co_occurring_objects', 'llm_backgrounds']]
    all_attributer_keys = [['vanilla', 'llm_kinds_chils', 'llm_co_occurring_objects', 'llm_backgrounds', 'auto_global', 'income_level', 'region', 'country']]
    predictors = ['chils', 'interpol_sims_top_4', 'interpol_sims_top_16', 'average_sims', 'average_vecs']
    vlms = ['clip_ViT-B/16', 'blip2']
    all_vlm_prompts = [['a photo of a {}'], ['USE OPENAI IMAGENET TEMPLATES']]
    all_vlm_prompt_dim_handlers = ['average_and_norm_then_stack', 'stack_all']
    all_lambs = [0, 0.5]
    cluster_sweep(dsetnames, all_attributer_keys, all_vlm_prompts, predictors, all_vlm_prompt_dim_handlers, vlms, all_lambs,
        log_dir ='aug24_kitchen_sink')   
        # log_dir ='aug24_auto_global_with_income')   

def sweep_soft_chils():
    # things that are fixed for CHiLS
    all_predictors = [f'chils_{k}' for k in [2,4,8,16,32]]
    all_attributer_keys = [['vanilla', 'llm_kinds_chils'], ['vanilla', 'llm_kinds_chils', 'llm_co_occurring_objects', 'llm_backgrounds', 'auto_global', 'income_level']]
    all_vlm_prompt_dim_handlers = ['average_and_norm_then_stack', 'stack_all']
    # things we sweep over
    dsetnames = ['living17', 'nonliving26', 'entity13', 'entity30', 'mit_states_0.8', 'mit_states_0.9',
        'dollarstreet__region', 'geode__region']
    vlms = ['clip_ViT-B/16', 'blip2']
    all_vlm_prompts = [['a photo of a {}'], ['USE OPENAI IMAGENET TEMPLATES']]
    lambs = [1]

    cluster_sweep(dsetnames, all_attributer_keys, all_vlm_prompts, all_predictors, 
            all_vlm_prompt_dim_handlers, vlms, lambs, log_dir='aug24_soft_chils')

def sweep_attributers():
    dsetnames = ['living17', 'entity30', 'mit_states_0.8', 'mit_states_0.9', 'dollarstreet__region', 'geode__region']
    all_predictors = ['interpol_sims_top_16', 'chils', 'average_sims']
    all_vlm_prompt_dim_handlers = ['average_and_norm_then_stack']
    vlms = ['clip_ViT-B/16', 'blip2']
    all_vlm_prompts = [['USE OPENAI IMAGENET TEMPLATES']]
    lambs = [0]
    all_attributer_keys = [['vanilla', x] for x in ['llm_kinds_chils', 'llm_co_occurring_objects', 'llm_backgrounds', 
        'auto_global', 'income_level', 'region', 'country', 'llm_states', 'llm_dclip', 'llm_kinds']] + ['vanilla']
    cluster_sweep(dsetnames, all_attributer_keys, all_vlm_prompts, all_predictors, all_vlm_prompt_dim_handlers,
        vlms, lambs, log_dir='aug25_attributers')

def sweep_k_and_lamb():
    dsetnames = ['dollarstreet__region', 'geode__region', 'living17', 'nonliving26', 'entity13', 'entity30', 'mit_states_0.8', 'mit_states_0.9']
    all_predictors = [f'average_top_{k}_sims' for k in [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 32, 64, 128]]
    all_predictors += [f'average_top_{k}_vecs' for k in [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 32, 64, 128]]
    # all_predictors += [f'new_average_top_{k}_vecs' for k in [2,8,16]]
    # all_predictors += [f'new_chils_top_{k}_vecs' for k in [1,4,16]]
    # all_predictors += [f'new_chils_top_{k}_sims' for k in [1,4,16]]
    all_attributer_keys = [['vanilla', 'income_level', 'country', 'auto_global', 'llm_co_occurring_objects', 'llm_dclip', 'llm_kinds', 'region']]
    vlms = ['blip2', 'clip_ViT-B/16']
    all_vlm_prompts = [['USE OPENAI IMAGENET TEMPLATES']]
    all_vlm_prompt_dim_handlers = ['average_and_norm_then_stack']
    lambs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    cluster_sweep(dsetnames, all_attributer_keys, all_vlm_prompts, all_predictors, all_vlm_prompt_dim_handlers,
        vlms, lambs, log_dir='sep14_k_and_lamb_super_fine')

def sweep_knn():
    # dsetnames = ['dollarstreet__region', 'geode__region', 'living17', 'nonliving26', 'entity13', 'entity30', 'mit_states_0.8', 'mit_states_0.9']
    dsetnames = ['mit_states_0.9']
    all_attributer_keys = [['vanilla', 'income_level', 'country', 'auto_global', 'llm_co_occurring_objects', 'llm_dclip', 'llm_kinds_chils']]
    vlms = ['clip_ViT-B/16', 'blip2']
    all_vlm_prompts = [['USE OPENAI IMAGENET TEMPLATES']]
    all_vlm_prompt_dim_handlers = ['average_and_norm_then_stack']
    lambs = [0]
    all_predictors = [f'adaptive_knn_sims_{z}' for z in [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3]]
    all_predictors += [f'adaptive_knn_avgsims_{z}' for z in [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3]]
    # all_predictors = [f'knn_sims_{k}' for k in [2,4,16,64]]
    # all_predictors += [f'knn_voting_{k}' for k in [2,4,16,64]]
    cluster_sweep(dsetnames, all_attributer_keys, all_vlm_prompts, all_predictors, all_vlm_prompt_dim_handlers,
        vlms, lambs, log_dir='aug29_adaptive_knn_rest')


def sweep_preds_w_best_attrs():
    # dsetnames = ['dollarstreet__region', 'geode__region', 'living17', 'nonliving26', 'entity13', 'entity30', 'mit_states_0.8', 'mit_states_0.9']
    dsetnames = ['dollarstreet__region', 'nonliving26', 'mit_states_0.9']
    all_attributer_keys = [['vanilla', 'income_level', 'country', 'auto_global', 'llm_co_occurring_objects', 'llm_dclip', 'llm_kinds_chils']]
    vlms = ['clip_ViT-B/16', 'blip2']
    all_vlm_prompts = [['USE OPENAI IMAGENET TEMPLATES']]
    all_vlm_prompt_dim_handlers = ['stack_all']
    lambs = [0,0.5]
    # predictors = ['chils', 'average_sims', 'average_vecs', 'new_average_top_8_sims', 'new_average_top_8_vecs', 'new_average_top_16_sims', 
    #               'new_average_top_16_vecs', 'new_chils_top_4_sims', 'knn_sims_16', 'knn_sims_32', 'knn_sims_8']
    predictors = ['knn_sims_16', 'knn_sims_32', 'knn_sims_8']
    cluster_sweep(dsetnames, all_attributer_keys, all_vlm_prompts, predictors, all_vlm_prompt_dim_handlers,
        vlms, lambs, log_dir='aug26_preds_w_best_attrs_rest')

def sweep_add_in_attributes():
    dsetnames = ['dollarstreet__region', 'geode__region', 'living17', 'nonliving26', 'entity13', 'entity30', 'mit_states_0.8', 'mit_states_0.9']
    vlms = ['clip_ViT-B/16', 'blip2']
    all_vlm_prompts = [['USE OPENAI IMAGENET TEMPLATES']]
    all_vlm_prompt_dim_handlers = ['average_and_norm_then_stack']
    lambs = [0,0.25]
    all_attributer_keys = [['vanilla'], ['vanilla', 'auto_global'], ['vanilla', 'auto_global', 'income_level'],
        ['vanilla', 'auto_global', 'income_level', 'region'], ['vanilla', 'auto_global', 'income_level', 'region', 'country'],
        ['vanilla', 'auto_global', 'income_level', 'region', 'country', 'llm_kinds'], ['vanilla', 'auto_global', 'income_level', 'region', 'country', 'llm_kinds', 'llm_states'],
        ['vanilla', 'auto_global', 'income_level', 'region', 'country', 'llm_kinds', 'llm_states', 'llm_dclip'],
        ['vanilla', 'auto_global', 'income_level', 'region', 'country', 'llm_kinds', 'llm_states', 'llm_dclip', 'llm_co_occurring_objects'],
        ['vanilla', 'auto_global', 'income_level', 'region', 'country', 'llm_kinds', 'llm_states', 'llm_dclip', 'llm_co_occurring_objects', 'llm_backgrounds']]#,
        # 'llm_dclip', 'country'], ['vanilla', 'auto_global', 'income_level', 'llm_dclip', 'country', 'llm_states'],
        # ['vanilla', 'auto_global', 'income_level', 'llm_dclip', 'country', 'llm_states', 'llm_kinds'], 
        # ['vanilla', 'auto_global', 'income_level', 'llm_dclip', 'country', 'llm_states', 'llm_kinds', 'region'], 
        # ['vanilla', 'auto_global', 'income_level', 'llm_dclip', 'country', 'llm_states', 'llm_kinds', 'region', 'llm_co_occurring_objects'], 
        # ['vanilla', 'auto_global', 'income_level', 'llm_dclip', 'country', 'llm_states', 'llm_kinds', 'region', 'llm_co_occurring_objects', 'llm_backgrounds']] 
    # predictors = ['average_vecs', 'average_sims', 'chils', 'new_average_top_1_sims', 'new_average_top_16_sims', 'new_average_top_16_vecs', 'knn_sims_16']
    predictors = ['average_sims', 'average_vecs', 'average_top_16_sims', 'average_top_8_sims', 'average_top_8_vecs', 'average_top_16_vecs', 'chils']
    cluster_sweep(dsetnames, all_attributer_keys, all_vlm_prompts, predictors, all_vlm_prompt_dim_handlers,
        vlms, lambs, log_dir='sep14_add_in_attrs')
        # vlms, lambs, log_dir='aug28_add_in_attrs')


def sweep_add_in_attributes_new():
    dsetnames = ['dollarstreet__region', 'geode__region', 'living17', 'nonliving26', 'entity13', 'entity30', 'mit_states_0.8', 'mit_states_0.9']
    vlms = ['clip_ViT-B/16', 'blip2']
    all_vlm_prompts = [['USE OPENAI IMAGENET TEMPLATES']]
    all_vlm_prompt_dim_handlers = ['average_and_norm_then_stack']
    lambs = [0, 0.25]#, 0.5]
    all_attributer_keys = [['vanilla'], ['vanilla', 'llm_kinds'], ['vanilla', 'llm_kinds', 'llm_dclip'], ['vanilla', 'llm_kinds', 'llm_dclip', 'llm_states'],
        ['vanilla', 'llm_kinds', 'llm_dclip', 'llm_states', 'auto_global'], ['vanilla', 'llm_kinds', 'llm_dclip', 'llm_states', 'auto_global', 'income_level'],
        ['vanilla', 'llm_kinds', 'llm_dclip', 'llm_states', 'auto_global', 'income_level', 'region'],
        ['vanilla', 'llm_kinds', 'llm_dclip', 'llm_states', 'auto_global', 'income_level', 'region', 'llm_co_occurring_objects'],
        ['vanilla', 'llm_kinds', 'llm_dclip', 'llm_states', 'auto_global', 'income_level', 'region', 'llm_co_occurring_objects', 'llm_backgrounds']] 
    predictors = ['chils', 'average_sims', 'average_vecs', 'new_average_top_8_sims', 'new_average_top_16_vecs']
    cluster_sweep(dsetnames, all_attributer_keys, all_vlm_prompts, predictors, all_vlm_prompt_dim_handlers,
        vlms, lambs, log_dir='aug30_add_in_attrs_new_order_3')


def sweep_attrs_delete_one():
    dsetnames = ['dollarstreet__region', 'geode__region', 'living17', 'nonliving26', 'entity13', 'entity30', 'mit_states_0.8', 'mit_states_0.9']
    vlms = ['clip_ViT-B/16', 'blip2']
    all_vlm_prompts = [['USE OPENAI IMAGENET TEMPLATES']]
    all_vlm_prompt_dim_handlers = ['average_and_norm_then_stack']
    lambs = [0, 0.25]
    all_attributer_keys = [
        ['vanilla', 'llm_kinds', 'llm_dclip', 'llm_states', 'auto_global', 'income_level', 'region', 'llm_co_occurring_objects', 'llm_backgrounds', 'country'],
        ['llm_kinds', 'llm_dclip', 'llm_states', 'auto_global', 'income_level', 'region', 'llm_co_occurring_objects', 'llm_backgrounds', 'country'],
        ['vanilla', 'llm_dclip', 'llm_states', 'auto_global', 'income_level', 'region', 'llm_co_occurring_objects', 'llm_backgrounds', 'country'],
        ['vanilla', 'llm_kinds', 'llm_states', 'auto_global', 'income_level', 'region', 'llm_co_occurring_objects', 'llm_backgrounds', 'country'],
        ['vanilla', 'llm_kinds', 'llm_dclip', 'auto_global', 'income_level', 'region', 'llm_co_occurring_objects', 'llm_backgrounds', 'country'],
        ['vanilla', 'llm_kinds', 'llm_dclip', 'llm_states', 'income_level', 'region', 'llm_co_occurring_objects', 'llm_backgrounds', 'country'],
        ['vanilla', 'llm_kinds', 'llm_dclip', 'llm_states', 'auto_global', 'region', 'llm_co_occurring_objects', 'llm_backgrounds', 'country'],
        ['vanilla', 'llm_kinds', 'llm_dclip', 'llm_states', 'auto_global', 'income_level', 'llm_co_occurring_objects', 'llm_backgrounds', 'country'],
        ['vanilla', 'llm_kinds', 'llm_dclip', 'llm_states', 'auto_global', 'income_level', 'region', 'llm_co_occurring_objects']
    ]
    predictors = ['chils', 'average_sims', 'average_vecs', 'new_average_top_8_sims', 'new_average_top_8_vecs']
    cluster_sweep(dsetnames, all_attributer_keys, all_vlm_prompts, predictors, all_vlm_prompt_dim_handlers,
        vlms, lambs, log_dir='aug30_attr_deletion')

def sweep_attrs_add_one():
    dsetnames = ['dollarstreet__region', 'geode__region', 'living17', 'nonliving26', 'entity13', 'entity30', 'mit_states_0.8', 'mit_states_0.9']
    vlms = ['clip_ViT-B/16', 'blip2']
    all_vlm_prompts = [['USE OPENAI IMAGENET TEMPLATES']]
    all_vlm_prompt_dim_handlers = ['average_and_norm_then_stack']
    lambs = [0, 0.25]
    all_attributer_keys = ['vanilla'] + [['vanilla', x] for x in ['llm_kinds','llm_dclip', 'llm_states', 'auto_global', 'income_level', 
                                                                  'region', 'llm_co_occurring_objects', 'llm_backgrounds', 'country']]
    predictors = ['chils', 'average_sims', 'average_vecs', 'new_average_top_8_sims', 'new_average_top_8_vecs']
    cluster_sweep(dsetnames, all_attributer_keys, all_vlm_prompts, predictors, all_vlm_prompt_dim_handlers,
        vlms, lambs, log_dir='aug30_attr_insertion')

def sweep_try_out_orders():
    dsetnames = ['dollarstreet__region', 'geode__region', 'living17', 'nonliving26', 'entity13', 'entity30', 'mit_states_0.8', 'mit_states_0.9']
    vlms = ['clip_ViT-B/16', 'blip2']
    all_vlm_prompts = [['USE OPENAI IMAGENET TEMPLATES']]
    all_vlm_prompt_dim_handlers = ['average_and_norm_then_stack']
    lambs = [0, 0.25]
    predictors = ['average_sims', 'new_average_top_8_vecs', 'new_average_top_8_sims', 'chils']

    attr_groups = [['auto_global'], ['income_level', 'region', 'country'], ['llm_kinds', 'llm_dclip', 'llm_states'], ['llm_co_occurring_objects', 'llm_backgrounds']]
    from itertools import permutations
    orders = list(permutations([0,1,2,3]))
    for j, order in enumerate(orders):
        curr_attr_keys = []
        all_attributer_keys = []
        for i in order:
            curr_attr_keys.extend(attr_groups[i])
            all_attributer_keys.append(curr_attr_keys.copy())
        all_attributer_keys = [x+['vanilla'] for x in all_attributer_keys]
        cluster_sweep(dsetnames, all_attributer_keys, all_vlm_prompts, predictors, all_vlm_prompt_dim_handlers,
            vlms, lambs, log_dir=f'aug31_new_orders_{j}')



if __name__ == '__main__':
    # run_sweep()
    # sweep_chils()
    # sweep_dclip()
    # sweep_waffle()
    # sweep_vanilla()
    # stack_all_vlm_prompts()
    # our_best()
    # sweep_geographic()
    # sweep_all()
    # new_llm_queries()
    # sweep_soft_chils()
    # sweep_attributers()
    # sweep_k_and_lamb()
    # sweep_knn()
    # sweep_preds_w_best_attrs()
    sweep_add_in_attributes()
    # sweep_attrs_delete_one()
    # sweep_attrs_add_one()

    # sweep_try_out_orders()