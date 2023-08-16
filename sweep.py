from main import Config, main
from metrics import accuracy_metrics, acc_by_class_and_subpop, mark_as_correct
from constants import _CACHED_DATA_ROOT
from models.llm import LLM, Vicuna
import pandas as pd
import os
from typing import List, Tuple, Optional

def sweep(
    all_dsetnames: List[str], 
    all_attributer_keys: List[List[str]], 
    all_vlm_prompts: List[List[str]], 
    all_predictors: List[str], 
    all_vlm_prompt_dim_handlers: List[str],
    save_path: Optional[str]=None) -> pd.DataFrame:
    '''
    Simple function to run a ton of experiments
    Will need to be changed a bit when we revamp the attribute inference logic. 
    '''

    keys = ['dsetname', 'attributer_keys', 'vlm_prompts', 'predictor', 'vlm_prompt_dim_handler', 'accuracy', 
            'worst class accuracy', 'average worst subpop accuracy', 'std dev worst subpop accuracy']

    results_dict = dict({key:[] for key in keys})

    for dsetname in all_dsetnames:
        for attributer_keys in all_attributer_keys:
            for vlm_prompts in all_vlm_prompts:
                for predictor in all_predictors:
                    for vlm_prompt_dim_handler in all_vlm_prompt_dim_handlers:
                        args = dict({
                            'dsetname': dsetname,
                            'vlm': 'clip_ViT-B/16',
                            'llm': 'vicuna-13b-v1.5',
                            'attributer_keys': attributer_keys,
                            'vlm_prompts': vlm_prompts,
                            'vlm_prompt_dim_handler': vlm_prompt_dim_handler,
                            'predictor': predictor,
                            'lamb': 0.5
                        })
                        config = Config(args)
                        output_dict = main(config)
                        metric_dict = output_dict['metric_dict']


                        args_and_outputs = args
                        for key in metric_dict:
                            args_and_outputs[key] = metric_dict[key]

                        for key in results_dict:
                            results_dict[key].append(args_and_outputs[key])

                        results_df = pd.DataFrame.from_dict(results_dict)
                        if save_path:
                            results_df.to_csv(save_path)
    return results_df

def run_sweep():
    ### Testing to see if our sweep code works ... it did!
    dsetnames = ['dollarstreet__country.name']#, 'dollarstreet__country.name', 'dollarstreet__income_group']
    all_predictors = ['max_of_max', 'average_top_2', 'average_top_4', 'average_top_6', 'average_vecs',
                       'average_sims', 'interpol_sims_top_2', 'interpol_sims_top_4']

    all_attributer_keys = [['vanilla'], ['income_level'], ['country'], ['region'], ['income_level', 'country', 'region', 'vanilla'], ['income_level', 'country', 'region']]
    all_vlm_prompts = [['a photo of a {}'], ['USE OPENAI IMAGENET TEMPLATES']]
    all_vlm_prompt_dim_handlers = ['average_and_norm_then_stack', 'stack_all'] # 'average_and_stack'

    sweep(dsetnames, all_attributer_keys, all_vlm_prompts, all_predictors, all_vlm_prompt_dim_handlers,
          save_path='/checkpoint/mazda/mmmd_results/experiments/aug15_dollarstreet_global_attrs.csv')

if __name__ == '__main__':
    run_sweep()