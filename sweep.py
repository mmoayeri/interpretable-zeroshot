from main import Config, main
from metrics import accuracy_metrics, acc_by_class_and_subpop, mark_as_correct
from constants import _CACHED_DATA_ROOT
from models.llm import LLM, Vicuna
import pandas as pd
import os

def sweep(all_dsetnames, all_llm_prompts, all_vlm_prompts, all_predictors, save_path=None):
    '''
    Simple function to run a ton of experiments
    Will need to be changed a bit when we revamp the attribute inference logic. 
    '''

    keys = ['dsetname', 'llm_prompts', 'vlm_prompts', 'predictor', 'accuracy', 'worst class accuracy', 
            'average worst subpop accuracy', 'std dev worst subpop accuracy']

    results_dict = dict({key:[] for key in keys})

    for dsetname in all_dsetnames:
        for llm_prompts in all_llm_prompts:
            for vlm_prompts in all_vlm_prompts:
                for predictor in all_predictors:
                    args = dict({
                        'dsetname': dsetname,
                        'vlm': 'clip_ViT-B/16',
                        'llm': 'vicuna-13b-v1.5',
                        'llm_prompts': llm_prompts,
                        'vlm_prompts': vlm_prompts,
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
    dsetnames = ['dollarstreet__region', 'dollarstreet__country.name', 'dollarstreet__income_group']
    all_predictors = ['max_of_max', 'average_top_2', 'average_top_4', 'average_top_6', 'average_vecs',
                      'average_sims', 'interpol_sims_top_2', 'interpol_sims_top_4']
    all_llm_prompts = [[('classname', None)], [('classname', None), ('groundtruth', None)]]
    all_vlm_prompts = [['a photo of a {}'], ['USE OPENAI IMAGENET TEMPLATES'], ['USE CONDENSED OPENAI TEMPLATES']]

    sweep(dsetnames, all_llm_prompts, all_vlm_prompts, all_predictors, 
          save_path='/checkpoint/mazda/mmmd_results/experiments/aug11_simple_predictors_vlm_prompts_dollarstreet.csv')

if __name__ == '__main__':
    run_sweep()