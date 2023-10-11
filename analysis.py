from constants import _CACHED_DATA_ROOT, _METRICS, _INPUTS, _IMPORTANT_METRICS, _ALL_DSETNAMES
import pandas as pd
from typing import List, Optional, Tuple
from glob import glob
import os
import json
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ast import literal_eval

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": "cmss10",
        "axes.formatter.use_mathtext": "True",
        "mathtext.fontset": "stixsans"
    }
)

"""
First thing: a class to consolidate + process results
Will be able to... 
    - create a dataframe combining many jsons from a sweep (or multiple)
    - give baseline numbers (e.g. for pure vanilla, dclip, or chils)
    - let us 'slice and dice' ...

This will be flexible -- we'll add to this as needed
The point is that we have a clear and consistent methodology for processing results.
"""


class Analyze:
    def collect_jsons_for_sweep(
        self,
        log_dir: str,
        do_not_return_preds: bool = True,
        save_df: bool = False,
        save_df_root: str = "./mazda_analysis/experiment_dfs/",
    ) -> pd.DataFrame:
        results_dir_path = os.path.join(
            _CACHED_DATA_ROOT, "experiments", log_dir, "results", "*.json"
        )
        results_json_paths_for_all_runs = glob(results_dir_path)

        # Everything in a sweep will have the same saved keys; we get this list by accessing the first json
        with open(results_json_paths_for_all_runs[0], "r") as f:
            eg_results = json.load(f)
        keys = list(eg_results.keys())
        if do_not_return_preds:
            if "pred_classnames" in keys:
                keys.remove("pred_classnames")
            if "identifiers" in keys:
                keys.remove("identifiers")

        all_results = dict({k: [] for k in keys})

        for json_for_one_run in tqdm(results_json_paths_for_all_runs):
            with open(json_for_one_run, "r") as f:
                single_run_results = json.load(f)

            for key in keys:
                val = single_run_results[key]
                if type(val) is list:
                    val = sorted(val)  # alphabetize in case there was any inconsistency
                    val = str(
                        val
                    )  # so that we can groupby if desired; list type is unhashable
                all_results[key].append(val)
            all_results["json"] = json_for_one_run

        df = pd.DataFrame(zip(*[all_results[key] for key in keys]), columns=keys)

        if save_df:
            save_df_path = f"{save_df_root}{log_dir}.csv"
            df.to_csv(save_df_path, index=False)
        return df

    def consolidate_sweeps(
        self, log_dirs: List[str], return_preds: bool = False
    ) -> pd.DataFrame:
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

    def load_baseline(self, method: str, no_groundtruth: bool = True):
        if method == "vanilla":
            # df = self.collect_jsons_for_sweep('aug23_pure_vanilla')
            df = self.collect_jsons_for_sweep("sep13_pure_vanilla")
        elif method == "dclip":
            # df = self.collect_jsons_for_sweep('aug23_dclip')
            df = self.collect_jsons_for_sweep("sep13_dclip")
        elif method == "chils":
            # df = self.collect_jsons_for_sweep('aug29_chils')
            df = self.collect_jsons_for_sweep("sep13_chils")
            if no_groundtruth:
                df = df[df["attributer_keys"] == "['llm_kinds_chils', 'vanilla']"]
        elif method == "waffle":
            all_dfs = []
            for i in range(1, 6):
                # all_dfs.append(self.collect_jsons_for_sweep(f'sep12_waffle_{i}'))
                all_dfs.append(self.collect_jsons_for_sweep(f"sep13_waffle_{i}"))
            df = pd.concat(all_dfs)
        else:
            raise ValueError(
                f"Baseline {method} not recognized / not yet supported. Run the appropriate sweep and update this function."
            )
        return df

    def baseline_numbers(
        self, vlm_prompts: str = "['USE OPENAI IMAGENET TEMPLATES']"
    ) -> pd.DataFrame:
        """
        This function will return a dataframe with metrics for dclip, chils, and vanilla across all datasets.

        Some details:
        - we'll average over the two VLMs (clip and blip)
        - we'll take as an argument the vlm_prompts - must be one of "['USE OPENAI IMAGENET TEMPLATES']" or "['a photo of a {}']" or 'mean' in which case we average
        """
        allowable_vlm_prompts = [
            "['USE OPENAI IMAGENET TEMPLATES']",
            "['a photo of a {}']",
            "mean",
        ]
        # allowable_vlms = ['clip__ViT-B/16', 'blip-2', 'mean']
        assert (
            vlm_prompts in allowable_vlm_prompts
        ), f"vlm_prompts must be in {', '.join(allowable_vlm_prompts)}. You passed: {vlm_prompts}"
        # assert vlm in allowable_vlm_prompts, f"vlm must be in {', '.join(allowable_vlms)}. You passed {vlm}"

        baselines_dict = dict({k: [] for k in ["dsetname", "method", "vlm"] + _METRICS})
        for method in ["vanilla", "dclip", "chils", "waffle"]:
            df = self.load_baseline(method)

            if vlm_prompts != "mean":
                df = df[df["vlm_prompts"] == vlm_prompts]

            for (vlm, dsetname), sub_df in df.groupby(["vlm", "dsetname"]):
                # let's only keep metrics and avg out remaining dimensions (just VLM, and perhaps vlm_prompt if vlm_prompt arg to this fn was 'mean')
                sub_df = sub_df[_METRICS].mean()
                for metric in _METRICS:
                    baselines_dict[metric].append(sub_df[metric])
                baselines_dict["dsetname"].append(dsetname)
                baselines_dict["method"].append(method)
                baselines_dict["vlm"].append(vlm)

        baselines_df = pd.DataFrame.from_dict(baselines_dict)
        return baselines_df

    def best_runs_given_metric(
        self,
        results_df: pd.DataFrame,
        inputs_to_show: List[str] = [
            "attributer_keys",
            "vlm_prompts",
            "predictor",
            "vlm_prompt_dim_handler",
            "lamb",
        ],
        metric_to_sort_by: str = "accuracy",
        metrics_to_show: List[str] = _METRICS,
        num_to_show: int = 20,
        dsetname: str = "mean",
    ) -> pd.DataFrame:
        if dsetname != "mean":
            results_df = results_df[results_df["dsetname"] == dsetname]

        results_df = results_df.groupby(inputs_to_show).mean("accuracy").reset_index()
        return results_df.sort_values(metric_to_sort_by, ascending=False).head(
            num_to_show
        )[inputs_to_show + metrics_to_show]

    def baselines_summarize_stats(self, important_only: bool = True) -> pd.DataFrame:
        # with important_only, we add focus on fairness metrics + overall acc
        baselines = self.baseline_numbers()
        summarized = baselines.groupby("method").mean("accuracy")
        if important_only:
            summarized = summarized[_IMPORTANT_METRICS]
        return summarized

    def k_vs_lamb(
        self,
        log_dir: str = "sep21_k_and_lamb_super_fine",
        percentile: int = 5
    ):
        df = self.collect_jsons_for_sweep(log_dir)
        # ideally we should change second condition to be isin(_DSETNAMES), but this is the same
        df2 = df[(df.sims_or_vecs == 'sims') & (df.dsetname != 'dollarstreet__income_level_thresh_0.8')]
        df['k'] = df2['predictor'].apply(lambda x: int(x.split('_')[-2]))
        df2 = df2.groupby(['k', 'lamb']).mean('accuracy').reset_index()
        f, axs = plt.subplots(1,2, figsize=(10,4))
        for param, ax in zip(['k', 'lamb'], axs):
            ax.set_facecolor((0.93,0.93,0.93))
            ax.grid(color='white', zorder=0)
            grouped = df2.groupby([param]).mean('accuracy').reset_index()
            grouped = grouped.rename(columns={'lamb':'$\lambda$', 'k':'$k$'})
            if param == 'k':
                sns.scatterplot(grouped, y="accuracy", x=f"avg worst {percentile}th percentile class accs", hue="$k$", hue_norm=LogNorm(), ax=ax, legend="auto", zorder=6, s=80)
            else:
                sns.scatterplot(grouped, y="accuracy", x=f"avg worst {percentile}th percentile class accs", hue="$\lambda$", ax=ax, legend="auto", zorder=6, s=80)
            ax.set_xlabel(f'Accuracy on Worst {percentile}% of Classes', fontsize=14)
            ax.set_ylabel('Overall Accuracy', fontsize=14)
        f.tight_layout(); f.savefig(f'plots/tradeoff/k_and_lamb_{percentile}_percentile.jpg', dpi=300)

    def sims_vs_vecs(
        self,
        log_dir: str = "sep21_k_and_lamb_super_fine",
        percentile: int = 5
    ):
        df = self.collect_jsons_for_sweep(log_dir)
        df = df[df.dsetname != 'dollarstreet__income_level_thresh_0.8']
        df['sims_or_vecs'] = df.predictor.apply(lambda x: 'vecs' if 'vecs' in x else 'sims')
        df['k'] = df['predictor'].apply(lambda x: int(x.split('_')[-2]))
        f, ax = plt.subplots(1,1, figsize=(6,4))
        df = df.rename(columns={'lamb':'$\lambda$', 'sims_or_vecs':'Sims or Vecs'})
        grouped = df.groupby(['$\lambda$', 'Sims or Vecs']).mean('accuracy')
        sns.scatterplot(data = grouped, y="accuracy", x=f"avg worst {percentile}th percentile class accs", style="Sims or Vecs", hue='$\lambda$', ax=ax, zorder=6)
        ax.set_facecolor((0.93,0.93,0.93))
        ax.grid(color='white', zorder=0)
        ax.set_xlabel(f'Accuracy on Worst {percentile}% of Classes', fontsize=14)
        ax.set_ylabel('Overall Accuracy', fontsize=14)
        ax.legend(bbox_to_anchor=(1.02,0.85))
        f.tight_layout(); f.savefig('plots/tradeoff/sims_or_vecs.jpg', dpi=300)


    def our_best_method(self) -> pd.DataFrame:
        df = self.collect_jsons_for_sweep("sep14_our_bests_2")
        # df = pd.read_csv('mazda_analysis/experiment_dfs/aug28_add_in_attrs.csv')
        # best_attr_keys =  "['auto_global', 'country', 'income_level', 'llm_co_occurring_objects', " + \
        #                     "'llm_dclip', 'llm_kinds', 'llm_states', 'region', 'vanilla']"
        best_attr_keys = "['auto_global', 'income_level', 'llm_co_occurring_objects', 'llm_dclip', 'llm_kinds', 'llm_states', 'region', 'vanilla']"
        # ours = df[(df.predictor == 'new_average_top_16_sims') & (df.attributer_keys == best_attr_keys) & (df.lamb == 0)]
        ours = df[
            (df.predictor == "average_top_16_sims")
            & (df.attributer_keys == best_attr_keys)
            & (df.lamb == 0)
        ]
        ours["method"] = ["ours"] * len(ours)
        return ours

    def baselines_and_ours(self, our_k:int = 16, our_lamb:float = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dirs = glob('/checkpoint/mazda/mmmd_results/experiments/sep20_all_*')
        all_df = []
        for d in dirs:
            df = self.collect_jsons_for_sweep(d.split('/')[-1])
            df['method'] = [d.split('/')[-1].split('_')[-2]]*len(df)
            all_df.append(df)
        df = pd.concat(all_df)
        baselines = df[df.method != 'ours']
        ours = df[(df.method == 'ours') & (df.predictor == f'average_top_{our_k}_sims') & (df.lamb == our_lamb)]
        return baselines, ours


    def beautify_dsetname(self, dsetname: str) -> str:
        return (
            dsetname.split("_full")[0]
            .split("__")[0]
            .replace("_", " ")
            .title()
            .replace("Mit", "MIT")
            .replace("0.8", "(Coarse)")
            .replace("0.9", "(Fine)")
            .replace("Thresh ", "")
            .replace('Dollarstreet (Fine)', 'Dollarstreet')
        )

    def beautify_methodname(self, methodname: str) -> str:
        renamer = dict(
            {
                "vanilla": "Vanilla",
                "dclip": "DCLIP",
                "chils": "CHiLS",
                "ours": "Ours",
                "waffle": "Waffle",
            }
        )
        return renamer[methodname]

    def acc_by_method_table(self, vlm: str = 'both'):
        # summary = self.baselines_summarize_stats()
        # ours = self.our_best_method()
        baselines, ours = self.baselines_and_ours()
        ours_with_base = pd.concat([baselines, ours])
        if vlm != 'both':
            assert vlm in ['clip_ViT-B/16', 'blip-2'], f"Unrecognized vlm: {vlm}"
            ours_with_base = ours_with_base[ours_with_base.vlm == vlm]

        ours_with_base = ours_with_base[ours_with_base.dsetname.isin(_ALL_DSETNAMES)]

        metrics_to_show = [
            "accuracy", "average worst subpop accuracy", "avg worst 20th percentile class accs", "avg worst 20th percentile subpop accs", 
            "avg worst 10th percentile class accs", "avg worst 10th percentile subpop accs", 
        ]

        summary = ours_with_base.groupby(['method']).mean('accuracy')[metrics_to_show]
        summary = summary.reindex(["vanilla", "dclip", "waffle", "chils", "ours"])
        summary = summary.rename(index=self.beautify_methodname)

        # Idk why i need to do this groupby 'lamb', but do not fear, there is only one lamb value
        # ours = ours.groupby("method").mean("accuracy")[_IMPORTANT_METRICS]
        # summary = pd.concat([summary, ours])
        table_str = summary.style.highlight_max(axis=0, props="textbf:--rwrap;").format(precision=2).to_latex()#float_format="{:.2f}".format)
        # save_str = f'for_paper/acc_by_method/{vlm}.txt'
        with open('for_paper/acc_by_method.txt', 'w') as f:
            f.write(table_str)
        print(table_str)

    def tables_method_by_dset_per_metric(
        self, save_root: str = "for_paper/tables/by_method_and_dset/"
    ):
        os.makedirs(save_root, exist_ok=True)
        baselines, ours = self.baselines_and_ours()
        ours_with_base = pd.concat([baselines, ours])
        grouped = ours_with_base.groupby(["method", "dsetname"]).mean("accuracy")
        metrics_to_show = [
            "accuracy",
            "average worst subpop accuracy",
            "avg worst 20th percentile class accs",
            "avg worst 20th percentile subpop accs",
            "avg worst 10th percentile class accs",
            "avg worst 10th percentile subpop accs",
            # "avg worst 5th percentile class accs",
            # "avg worst 5th percentile subpop accs",
            # "worst class accuracy"
        ]
        for metric in metrics_to_show:#_IMPORTANT_METRICS:
            df = (
                grouped[metric]
                .reset_index()
                .pivot(index="method", columns="dsetname")[metric]
            )
            df = df.reindex(
                columns=[
                    # "method",
                    "dollarstreet__income_level_thresh_0.9",
                    "geode__region",
                    "mit_states_0.8",
                    "mit_states_0.9",
                    "entity13",
                    "entity30",
                    "nonliving26",
                    "living17",
                    # "dollarstreet__income_level_thresh_0.8",
                ]
            )
            df = df.reindex(["vanilla", "dclip", "waffle", "chils", "ours"])
            df = df.rename(
                columns=self.beautify_dsetname, index=self.beautify_methodname
            )
            print(df)

            # df = df.style.highlight_max(axis=0, props="textbf:--rwrap;")

            table_str = df.to_latex(
                index=True,#False,
                float_format="%.2f",
                caption=metric
            )

            table_str = df.style.highlight_max(axis=0, props="textbf:--rwrap;").format(precision=2).to_latex()
            print(table_str)
            with open(f'{save_root}/{metric.replace(" ", "_")}.txt', 'w') as f:
                f.write(table_str)

    def by_group(self, attr_keys):
        groups = []
        attr_keys = literal_eval(attr_keys)
        for key in attr_keys:
            if key == "auto_global":
                groups.append("Global")
            elif key == "income_level":
                groups.append("Geographic")
            elif key == "llm_kinds":
                groups.append("LLM (core)")
            elif key == "llm_backgrounds":
                groups.append("LLM (spurious)")
        return groups

    def add_in_attrs(
        self,
        log_dir: str = 'sep21_add_in_attrs',
        metrics_to_show: List[str] = ["accuracy", "avg worst 10th percentile class accs"],
        save_fname: str = 'adding_in_attrs.jpg'
    ):
        df = self.collect_jsons_for_sweep(log_dir)
        # TODO: update _DSETNAMES so this becomes isin(_DSETNAMES)
        df = df[df.dsetname.isin(['dollarstreet__income_level_thresh_0.9', 'geode__region', 'living17', 'nonliving26', 'entity13', 'entity30', 'mit_states_0.8', 'mit_states_0.9'])]
        df['num_attributers'] = df.attributer_keys.apply(lambda x:len(literal_eval(x)))
        grouped = df.groupby(['predictor', 'lamb', 'num_attributers']).mean('accuracy').reset_index()
        grouped['predictor_with_lamb'] = [f"{p}_{l}" for p,l in zip(list(grouped.predictor), list(grouped.lamb))]

        all_attributer_keys = [literal_eval(x) for x in df.attributer_keys]
        attributers_by_len = dict()
        for attributers in all_attributer_keys:
            num_attributers = len(attributers)
            if num_attributers in attributers_by_len:
                assert attributers == attributers_by_len[num_attributers], "Some funny business. In this analysis, only one attributer should be added at a time."
            else:
                attributers_by_len[num_attributers] = attributers

        nums = sorted(list(attributers_by_len))
        attributers_in_order = [[attributers_by_len[1][0]]]
        for i in range(min(nums)+1, max(nums)+1):
            ith_attributer = [x for x in attributers_by_len[i] if x not in attributers_in_order]
            attributers_in_order.append(ith_attributer)

        ith_attribute = [[x for x in attributers_by_len[i+1] if x not in attributers_by_len[i]][0] for i in range(1, len(attributers_by_len))]
        ith_attribute = ['+ ' + x.replace('_', ' ').title().replace('Llm ', '').replace('Dclip', 'Descriptors').replace('Co Occurring', 'Co-occurring\n') for x in ith_attribute]
        ith_attribute = ['Classname Only'] + ith_attribute

        filtered = grouped[grouped.predictor_with_lamb.isin(['average_sims_1.0', 'chils_0.0', 'average_top_16_sims_0.0'])]
        filtered['Predictor'] = filtered.predictor_with_lamb.apply(lambda x: ' '.join(x.split('_')[:-1]).title())

        # let's add in the first point (classname only; sweep only does this w predictor=average sims)
        row = filtered[filtered.num_attributers == 1]
        filtered2 = filtered
        for predictor in ['Chils', 'Average Top 16 Sims']:
            row2 = row
            row2['Predictor'] = predictor
            filtered2 = pd.concat((filtered2, row2))

        filtered2['Consolidation Scheme'] = filtered2.Predictor.map({'Average Top 16 Sims': 'Ours', 'Average Sims': 'Average', 'Chils':'CHiLS'})
        
        n_rows = int(np.ceil(len(metrics_to_show) / 2))
        f, axs = plt.subplots(n_rows, 2, figsize=(9, 3.5*n_rows))
        if n_rows == 1:
            axs = [axs]
        for i, metric in enumerate(metrics_to_show):
            ax = axs[i // 2][i % 2]
            # sns.lineplot(filtered2, x="num_attributers", y=metric, hue="Predictor", ax=ax, zorder=6, hue_order=['Average Top 16 Sims', 'Average Sims', 'Chils'])
            sns.lineplot(filtered2, x="num_attributers", y=metric, hue="Consolidation Scheme", ax=ax, zorder=6, hue_order=['Ours', 'Average', 'CHiLS'])
            ax.set_facecolor((0.93,0.93,0.93))
            ax.grid(color='white', zorder=0)
            ax.set_xticks(range(1, len(ith_attribute)+1))
            ax.set_xticklabels(ith_attribute, rotation=40, fontsize=9)
            metric = metric.title()
            if 'Avg Worst' in metric:
                words = metric.split(' ')
                percentile = int(words[2][:-2])
                acc_type = words[-2]
                acc_type += ('es' if acc_type == 'Class' else 's')
                metric = f'Worst {percentile}% of {acc_type}'
            ax.set_ylabel(metric, fontsize=16)
            ax.set_xlabel('Adding in Attribute Types Sequentially', fontsize=16)
        f.tight_layout(); f.savefig(f'plots/{save_fname}.jpg', dpi=300, bbox_inches='tight')  

    def add_in_attrs_appendix(self):
        self.add_in_attrs(
            metrics_to_show=['avg worst 20th percentile class accs', 'avg worst 20th percentile subpop accs'],
            save_fname = "adding_in_attrs_appendix"
        )

    def adding_in_attributes_try_many_orders(
        self,
        # df_csv_path: str = 'mazda_analysis/experiment_dfs/aug31_new_orders_0.csv'
        log_dir: str = 'aug29_add_in_attrs_new_order',
        # predictors_with_lamb_to_show : List[str] = ['average_sims_1.0', 'chils_0.0', 'max_of_max_0.0', 'new_average_top_8_sims_0.0'],
        # metrics: List[str] = ['accuracy', 'worst class accuracy']
    ):
        df = self.collect_jsons_for_sweep(log_dir)

        df["attribute types"] = df.attributer_keys.apply(lambda x: self.by_group(x))
        attributers_by_len = dict({len(v): v for v in df["attribute types"]})
        attributers_by_len = dict(
            sorted(attributers_by_len.items(), key=lambda x: x[0])
        )
        attributers_in_order = []
        for i, g in attributers_by_len.items():
            ith_g = [x for x in g if x not in attributers_in_order]
            attributers_in_order.append(ith_g)

        ith_attribute = ["Classname\nOnly", "+ " + attributers_by_len[1][0]]
        ith_attribute += [
            "+ "
            + [x for x in attributers_by_len[i + 1] if x not in attributers_by_len[i]][
                0
            ]
            for i in range(1, len(attributers_by_len))
        ]

        df["num_groups"] = df["attribute types"].apply(lambda x: len(x))
        grouped = (
            df.groupby(["predictor", "lamb", "num_groups"])
            .mean("accuracy")
            .reset_index()
        )
        grouped["predictor_with_lamb"] = [
            f"{p}_{l}" for p, l in zip(list(grouped.predictor), list(grouped.lamb))
        ]

        metrics_to_show = ["accuracy", "worst class accuracy", "poor", "Africa"]
        n_rows = int(np.ceil(len(metrics_to_show) // 2))
        f, axs = plt.subplots(n_rows, len(metrics_to_show) // 2, figsize=(15, 10))
        for i, metric in enumerate(metrics_to_show):
            ax = axs[i // n_rows, i % n_rows]
            sns.lineplot(
                data=grouped,
                x="num_groups",
                y=metric,
                hue="predictor_with_lamb",
                ax=ax,
                zorder=6,
            )
            ax.set_facecolor((0.93, 0.93, 0.93))
            ax.grid(color="white", zorder=0)
            ax.set_xticks(range(1, len(ith_attribute) + 1))
            ax.set_xticklabels(ith_attribute, rotation=30, fontsize=14)
            ax.set_ylabel(metric.title(), fontsize=20)
            ax.set_xlabel("Adding in Attribute Types Sequentially", fontsize=20)

        save_path = "plots/attr_orders/" + df_csv_path.split("_")[-1].replace(
            ".csv", ".jpg"
        )
        f.tight_layout()
        f.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


class BaseTable:
    REGIONS = ["Asia", "Europe", "Americas", "Africa", "SouthEastAsia", "WestAsia"]
    INCOMES = ["poor", "lower middle class", "upper middle class", "rich"]
    METHOD_ORDER = ["vanilla", "dclip", "waffle", "chils", "ours"]
    VLM_ORDER = ["clip_ViT-B/16", "blip2"]

    def __init__(self, baselines: pd.DataFrame, ours: pd.DataFrame = None):
        """Makes a table containing baseline performance"""
        self.df = baselines
        if ours is not None:
            self.df = pd.concat([baselines, ours], ignore_index=True)
        self.df["worst region"] = self.df[self.REGIONS].min(axis=1, numeric_only=True)
        self.df["worst income"] = self.df[self.INCOMES].min(axis=1, numeric_only=True)
        self.analyzer = Analyze()


class SummaryTable(BaseTable):
    def __init__(
        self,
        baselines: pd.DataFrame,
        ours: pd.DataFrame = None,
        vlm: str = "clip_ViT-B/16",
    ):
        """Makes a table containing baseline performance"""
        super().__init__(baselines, ours=ours)
        self.vlm = vlm
        self.table = self.make_table()

    def make_table(self) -> pd.DataFrame:
        table = self.df
        # set method to categorical with custom order
        table["method"] = pd.Categorical(table["method"], self.METHOD_ORDER)
        table["vlm"] = pd.Categorical(table["vlm"], self.VLM_ORDER)
        table = table[table["vlm"] == self.vlm]
        table = table.sort_values(by=["vlm", "dsetname", "method"])[
            [
                "method",
                "accuracy",
                "worst region",
                "worst income",
                "average worst subpop accuracy",
                "avg worst 20th percentile class accs",
                "avg worst 20th percentile subpop accs",
                "avg worst 10th percentile class accs",
                "avg worst 10th percentile subpop accs"
            ]
        ]
        table = self.average_over_datasets(table)
        return table

    def average_over_datasets(self, table):
        return table.groupby("method").mean()

    def to_latex(self) -> str:
        table_latex = self.table
        table_latex = table_latex.rename(
            columns={
                "avg worst 20th percentile class accs": "Worst 20\% of classes",
                "average worst subpop accuracy": "Worst Subpopulation",
                "worst income": "Worst Income",
                "worst region": "Worst Region",
                "accuracy": "Accuracy",
            }
        )
        return table_latex.to_latex(
            index=True,
            float_format="%.2f",
            formatters={"dsetname": lambda x: x.replace("_", " ")},
            caption="Zero-shot classification across hierarchical, states, and geographic datasets.",
        )


class GeographicShiftTable(BaseTable):
    def __init__(self, 
        baselines: pd.DataFrame, 
        ours: pd.DataFrame = None,
        vlm: str = "clip_ViT-B/16"
        ):
        """Makes a table containing baseline performance"""
        super().__init__(baselines, ours=ours)
        self.vlm = vlm
        self.table = self.make_table()

    def make_table(self) -> pd.DataFrame:
        table = self.df[
            self.df["dsetname"].isin(["dollarstreet__income_level_thresh_0.9", "geode__region"])
        ]
        # set method to categorical with custom order
        table["method"] = pd.Categorical(table["method"], self.METHOD_ORDER)
        # table["vlm"] = pd.Categorical(table["vlm"], self.VLM_ORDER)
        table = table[table["vlm"] == self.vlm]
        table = table.sort_values(by=["vlm", "dsetname", "method"])[
            [
                "vlm",
                "dsetname",
                "method",
                "accuracy",
                # "worst region",
                # "worst income",
                # "average worst subpop accuracy",
                "avg worst 20th percentile class accs",
                "avg worst 20th percentile subpop accs",
                "avg worst 10th percentile class accs",
                "avg worst 10th percentile subpop accs",
                # "avg worst 5th percentile class accs",
                # "avg worst 5th percentile subpop accs"
            ]
        ]
        table = self.average_over_trials(table)
        return table
    
    def average_over_trials(self, table: pd.DataFrame) -> pd.DataFrame:
        return table.groupby(["dsetname", "method", "vlm"]).mean().reset_index()

    def to_latex(self) -> str:
        table_cols = ["method", "accuracy", "worst region", "worst income", "average worst subpop accuracy", "avg worst 20th percentile class accs", "avg worst 20th percentile subpop accs"]

        table_latex = self.table
        table_strs = []
        for dsetname, table_latex in self.table.groupby('dsetname'):
            table_latex = table_latex[table_cols]
            table_latex = table_latex.replace(np.nan, "-")
            table_latex = table_latex.replace("dollarstreet__income_level_thresh_0.8", "DollarStreet")
            table_latex = table_latex.replace("geode__region", "GeoDE")
            table_latex = table_latex.replace("clip_ViT-B/16", "CLIP")
            table_latex = table_latex.replace("blip2", "BLIP-2")
            table_latex = table_latex.rename(
                columns={"avg worst 20th percentile class accs": "Worst 20\% of Classes",
                         "avg worst 20th percentile subpop accs": "Worst 20\% of Subpops",
                         "average worst subpop accuracy": "Avg Worst Subpop"}
            )
            table_latex.method = table_latex.method.apply(self.analyzer.beautify_methodname)
            table_strs.append(
                table_latex.to_latex(
                    index=False,
                    float_format="%.2f",
                    formatters={"dsetname": lambda x: x.replace("_", " ")},
                    caption="Performance on geographically diverse household object classification.",
                )
            )
        return table_strs


class NonGeographicDatasetsTable(BaseTable):
    VLM_ORDER = ["clip_ViT-B/16", "blip2"]

    def __init__(
        self,
        baselines: pd.DataFrame,
        ours: pd.DataFrame = None,
        vlm: str = "clip_ViT-B/16",
    ):
        """Makes a table containing baseline performance"""
        super().__init__(baselines, ours=ours)
        self.vlm = vlm
        self.table = self.make_table()

    def make_table(self) -> pd.DataFrame:
        table = self.df[
            ~self.df["dsetname"].isin(["dollarstreet__region", "geode__region", "dollarstreet__region_thresh_0.8", "dollarstreet__region_thresh_0.9",
                                       "dollarstreet__income_level_thresh_0.8", "dollarstreet__income_level_thresh_0.9"])
        ]
        table["dataset_type"] = table["dsetname"].apply(
            lambda x: "States" if "mit_states" in x else "Hierarchical"
        )
        # set method to categorical with custom order
        table["method"] = pd.Categorical(table["method"], self.METHOD_ORDER)
        # filter vlm
        table = table[table["vlm"] == self.vlm]
        table = table.sort_values(by=["vlm", "dsetname", "method"])[
            [
                "dataset_type",
                "method",
                "accuracy",
                "average worst subpop accuracy",
                "avg worst 20th percentile class accs",
                "avg worst 20th percentile subpop accs",
                "avg worst 10th percentile class accs",
                "avg worst 10th percentile subpop accs",
                # "avg worst 5th percentile class accs",
                # "avg worst 5th percentile subpop accs"
            ]
        ]
        table.method = table.method.apply(self.analyzer.beautify_methodname)
        table = self.average_over_dataset_types(table)
        return table

    def average_over_dataset_types(self, table: pd.DataFrame) -> pd.DataFrame:
        return table.groupby(["dataset_type", "method"]).mean()

    def to_latex(self) -> str:
        table_metrics = ["accuracy", "average worst subpop accuracy", "avg worst 20th percentile class accs", "avg worst 20th percentile subpop accs"]
        table_latex = self.table
        table_latex = table_latex[table_metrics]
        table_latex = table_latex.rename(
            columns={"avg worst 20th percentile class accs": "Worst 20\% of Classes",
                     "avg worst 20th percentile subpop accs": "Worst 20\% of Subpops",
                     "average worst subpop accuracy": "Avg Worst Subpop"}
            # columns={"average worst subpop accuracy": "Worst Subpopulation"}
        )
        table_latex = table_latex.rename(columns={"accuracy": "Accuracy"})
        table_latex.index = table_latex.index.map(lambda x: (x[0], x[1].title()))
        return table_latex.to_latex(
            index=True,
            float_format="%.2f",
            column_format="llccc",
            formatters={"dsetname": lambda x: x.replace("_", " ")},
            caption="Zero-shot classification on datasets with known variation types.",
        )


class NonGeographicRowPerDatasetTable(NonGeographicDatasetsTable):
    def make_table(self) -> pd.DataFrame:
        table = self.df[
            ~self.df["dsetname"].isin(["dollarstreet__region", "geode__region"])
        ]
        table["dataset_type"] = table["dsetname"].apply(
            lambda x: "States" if "mit_states" in x else "Hierarchical"
        )
        # set method to categorical with custom order
        table["method"] = pd.Categorical(table["method"], self.METHOD_ORDER)
        table = table.sort_values(by=["vlm", "dsetname", "method"])[
            [
                "dsetname",
                "vlm",
                "method",
                "accuracy",
                "average worst subpop accuracy",
                "avg worst 20th percentile class accs",
            ]
        ]
        return table

    def to_latex(self) -> str:
        table_latex = self.table
        table_latex = table_latex.rename(
            columns={"avg worst 20th percentile class accs": "Worst 20\% of classes"}
        )
        table_latex = table_latex.rename(
            columns={"average worst subpop accuracy": "Worst Subpopulation"}
        )
        table_latex = table_latex.replace("clip_ViT-B/16", "CLIP")
        table_latex = table_latex.replace("blip2", "BLIP-2")
        table_latex["dsetname"] = (
            table_latex["dsetname"]
            .replace("mit_states_0.8", "MIT States (Coarse)")
            .replace("mit_states_0.9", "MIT States (Fine)")
        )
        table_latex = table_latex.rename(columns={"accuracy": "Accuracy"})
        # table_latex.index = table_latex.index.map(lambda x: (x[0], x[1].upper()))
        return table_latex.to_latex(
            index=False,
            float_format="%.2f",
            # column_format="llccc",
            formatters={"dsetname": lambda x: x.replace("_", " ")},
            caption="Zero-shot classification on datasets with known variation types.",
        )


if __name__ == "__main__":
    analyzer = Analyze()
    for i in range(23):
        analyzer.adding_in_attributes(
            f"mazda_analysis/experiment_dfs/aug31_new_orders_{i}.csv"
        )

    # generate table
    # analyzer = analysis.Analyze()
    # baseline_numbers = analyzer.baseline_numbers()
    # our_numbers = analyzer.our_best_method()

    # geographic table
    # table = analysis.GeographicShiftTable(baseline_numbers, our_numbers)
