from constants import _CACHED_DATA_ROOT, _METRICS, _INPUTS, _IMPORTANT_METRICS
import pandas as pd
from typing import List, Optional
from glob import glob
import os
import json
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "cmss10",
    "axes.formatter.use_mathtext": "True"
})

"""
First thing: a class to consolidate + process results
Will be able to...:
    - create a dataframe combining many jsons from a sweep (or multiple)
    - give baseline numbers (e.g. for pure vanilla, dclip, or chils)
    - let us 'slice and dice' ...

This will be flexible -- we'll add to this as needed
The point is that we have a clear and consistent methodology for processing results.
"""

class Analyze:

    def collect_jsons_for_sweep(self, log_dir: str, do_not_return_preds: bool=True) -> pd.DataFrame:
        results_dir_path = os.path.join(_CACHED_DATA_ROOT, 'experiments', log_dir, 'results', '*.json')    
        results_json_paths_for_all_runs = glob(results_dir_path)

        # Everything in a sweep will have the same saved keys; we get this list by accessing the first json
        with open(results_json_paths_for_all_runs[0], 'r') as f:
            eg_results = json.load(f)
        keys = list(eg_results.keys())
        if do_not_return_preds:
            if 'pred_classnames' in keys:
                keys.remove('pred_classnames')
            if 'identifiers' in keys:
                keys.remove('identifiers')

        all_results = dict({k:[] for k in keys})

        for json_for_one_run in tqdm(results_json_paths_for_all_runs):
            with open(json_for_one_run, 'r') as f:
                single_run_results = json.load(f)

            for key in keys: 
                val = single_run_results[key]
                if type(val) is list: 
                    val = sorted(val) # alphabetize in case there was any inconsistency
                    val = str(val) # so that we can groupby if desired; list type is unhashable
                all_results[key].append(val)
        
        df = pd.DataFrame(zip(*[all_results[key] for key in keys]), columns=keys)
        return df
    
    def consolidate_sweeps(self, log_dirs: List[str], return_preds: bool=False) -> pd.DataFrame:
        """
        Here, we'll combine results of various sweeps to one big dataframe. 
        Final dataframe columns will be union of all columns seen. 
        If one dataframe lacks a column, we simply place nan. 
            use case: I added region + income_level info, but didn't want to rerun massive sweep
                when it would be nan for all non-geo datasets anyway. 
        """
        dfs = []
        all_columns = []
        for log_dir in log_dirs:
            sweep_df = self.collect_jsons_for_sweep(log_dir, return_preds)
            dfs.append(sweep_df)
            all_columns.extend(list(sweep_df.columns))
        
        all_columns = list(set(all_columns))
        
        dfs_all_cols = []
        for df in dfs:
            empty_col = [np.nan] * len(df)
            cols_to_add = [col for col in all_columns if col not in df.columns]
            for col in cols_to_add:
                df[col] = empty_col
            dfs_all_cols.append(df)

        return pd.concat(dfs_all_cols)

    def load_baseline(self, method: str, no_groundtruth: bool=True):
        if method == 'vanilla':
            df = self.collect_jsons_for_sweep('aug23_pure_vanilla')
        elif method == 'dclip':
            df = self.collect_jsons_for_sweep('aug23_dclip')
        elif method == 'chils':
            df = self.collect_jsons_for_sweep('aug23_chils')
            if no_groundtruth:
                df = df[df['attributer_keys'] == "['llm_kinds_chils', 'vanilla']"]
        else:
            raise ValueError(f'Baseline {method} not recognized / not yet supported. Run the appropriate sweep and update this function.')
        return df

    def baseline_numbers(self, vlm_prompts: str = "['USE OPENAI IMAGENET TEMPLATES']") -> pd.DataFrame:
        """
        This function will return a dataframe with metrics for dclip, chils, and vanilla across all datasets.
        
        Some details:
        - we'll average over the two VLMs (clip and blip)
        - we'll take as an argument the vlm_prompts - must be one of "['USE OPENAI IMAGENET TEMPLATES']" or "['a photo of a {}']" or 'mean' in which case we average
        """
        allowable_vlm_prompts = ["['USE OPENAI IMAGENET TEMPLATES']", "['a photo of a {}']", 'mean']
        assert vlm_prompts in allowable_vlm_prompts, f"vlm_prompts must be in {', '.join(allowable_vlm_prompts)}. You passed: {vlm_prompts}"

        baselines_dict = dict({k:[] for k in ['dsetname', 'method']+_METRICS})
        for method in ['vanilla', 'dclip', 'chils']:
            df = self.load_baseline(method)

            if vlm_prompts != 'mean':
                df = df[df['vlm_prompts'] == vlm_prompts]

            for dsetname, sub_df in df.groupby('dsetname'):
                # let's only keep metrics and avg out remaining dimensions (just VLM, and perhaps vlm_prompt if vlm_prompt arg to this fn was 'mean')
                sub_df = sub_df[_METRICS].mean()
                for metric in _METRICS:
                    baselines_dict[metric].append(sub_df[metric])
                baselines_dict['dsetname'].append(dsetname)
                baselines_dict['method'].append(method)
        
        baselines_df = pd.DataFrame.from_dict(baselines_dict)
        return baselines_df


    def best_runs_given_metric(
        self,
        results_df: pd.DataFrame, 
        inputs_to_show: List[str] = ['attributer_keys', 'vlm_prompts', 'predictor', 'vlm_prompt_dim_handler', 'lamb'],
        metric_to_sort_by: str='accuracy', 
        metrics_to_show: List[str]=_METRICS, 
        num_to_show: int=20,
        dsetname: str='mean'
    ) -> pd.DataFrame:
        if dsetname != 'mean':
            results_df = results_df[results_df['dsetname'] == dsetname]

        results_df = results_df.groupby(inputs_to_show).mean('accuracy').reset_index()
        return results_df.sort_values(metric_to_sort_by, ascending=False).head(num_to_show)[inputs_to_show + metrics_to_show]
         

    def general_plotter(
        self,
        results_df: pd.DataFrame,
        col_to_compare: str,
        save_fname: str, 
        col_to_compare_vals: List[str] = None,
        metrics: List[str] = _METRICS,
        n_subplots_per_row: int = 5
    ):
        """
        col_to_compare is something like method, predictor, attributer_keys. Importantly, we need the
        columns of results_df to include (1) dsetname (2) each of metrics (3) col_to_compare

        col_to_compare_vals is just in case you want the vals to be in a specific order (e.g. baselines first)
                            or if you don't care about nor wish to visualize certain values.

        This guy is gonna make two plots:
        1. Show everything: subplot per metric * set of bars per dset * bar per col_to_compare
        2. Show averages: Subplot per metric * bar per col_to_compare showing average val over dsetss
        """
        n_subplots_per_row = min(len(metrics), n_subplots_per_row)
        n_rows = int(np.ceil(len(metrics) / n_subplots_per_row))
        f, axs = plt.subplots(n_rows, n_subplots_per_row, figsize = (4*n_subplots_per_row, 4*n_rows))
        avg_f, avg_axs = plt.subplots(n_rows, n_subplots_per_row, figsize = (3*n_subplots_per_row, 4*n_rows))
        if n_rows == 1:
            axs = [axs]
            avg_axs = [avg_axs]

        # we'll have a color for each of the things we want to compare
        if col_to_compare_vals is None:
            col_to_compare_vals = list(set(results_df[col_to_compare]))
        else:
            results_df = results_df[results_df[col_to_compare].isin(col_to_compare_vals)]
        cmap = mpl.colormaps['plasma']
        colors = [cmap((i+1)/len(col_to_compare_vals)) for i in range(len(col_to_compare_vals))]

        results_df['dsetname'] = results_df['dsetname'].apply(lambda x: x.split('__')[0].title())
        avg_df = results_df.groupby(col_to_compare).mean('dsetname').reset_index()

        for i, metric in enumerate(metrics):
            ax, avg_ax = [axs_set[i // n_subplots_per_row][i % n_subplots_per_row] for axs_set in [axs, avg_axs]]
            sns.barplot(data=results_df, x="dsetname", y=metric, hue=col_to_compare, ax=ax,
                        palette=colors, hue_order=col_to_compare_vals)

            sns.barplot(data=avg_df, x=col_to_compare, y=metric, ax=avg_ax, palette=colors,
                        order=col_to_compare_vals)

            for curr_ax in [ax, avg_ax]:
                for container in curr_ax.containers:
                    curr_ax.bar_label(container, fmt='%.2f', fontsize=8, rotation=90)
                curr_ax.tick_params(axis='x', rotation=90)
                curr_ax.set_xlabel(None)
                curr_ax.spines[['right', 'top']].set_visible(False)
            ax.legend(loc='lower right')

        f.tight_layout(); f.savefig(f'plots/{save_fname}.jpg', dpi=300, bbox_inches='tight')
        avg_f.tight_layout(); avg_f.savefig(f'plots/{save_fname}_avg.jpg', dpi=300, bbox_inches='tight')

    def baselines_summarize_stats(self, important_only: bool=True) -> pd.DataFrame:
        # with important_only, we add focus on fairness metrics + overall acc 
        baselines = self.baseline_numbers()
        summarized = baselines.groupby('method').mean('accuracy')
        if important_only:
            summarized = summarized[_IMPORTANT_METRICS]
        return summarized
