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
from ast import literal_eval

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": "cmss10",
        "axes.formatter.use_mathtext": "True",
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

            # if vlm != 'both':
            #     # df = df[df.vlm == vlm]

            # df = df.groupby(['vlm', 'dsetname']).mean('accuracy').reset_index()

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

    def general_plotter(
        self,
        results_df: pd.DataFrame,
        col_to_compare: str,
        save_fname: str,
        col_to_compare_vals: List[str] = None,
        metrics: List[str] = _METRICS,
        n_subplots_per_row: int = 5,
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
        f, axs = plt.subplots(
            n_rows, n_subplots_per_row, figsize=(7 * n_subplots_per_row, 4 * n_rows)
        )
        avg_f, avg_axs = plt.subplots(
            n_rows, n_subplots_per_row, figsize=(5 * n_subplots_per_row, 4 * n_rows)
        )
        if n_rows == 1:
            axs = [axs]
            avg_axs = [avg_axs]

        # we'll have a color for each of the things we want to compare
        if col_to_compare_vals is None:
            col_to_compare_vals = list(set(results_df[col_to_compare]))
        else:
            results_df = results_df[
                results_df[col_to_compare].isin(col_to_compare_vals)
            ]
        cmap = mpl.colormaps["plasma"]
        colors = [
            cmap((i + 1) / len(col_to_compare_vals))
            for i in range(len(col_to_compare_vals))
        ]

        results_df["dsetname"] = results_df["dsetname"].apply(
            lambda x: x.split("__")[0].title()
        )
        avg_df = results_df.groupby(col_to_compare).mean("dsetname").reset_index()

        for i, metric in enumerate(metrics):
            ax, avg_ax = [
                axs_set[i // n_subplots_per_row][i % n_subplots_per_row]
                for axs_set in [axs, avg_axs]
            ]
            sns.barplot(
                data=results_df,
                x="dsetname",
                y=metric,
                hue=col_to_compare,
                ax=ax,
                palette=colors,
                hue_order=col_to_compare_vals,
            )

            sns.barplot(
                data=avg_df,
                x=col_to_compare,
                y=metric,
                ax=avg_ax,
                palette=colors,
                order=col_to_compare_vals,
            )

            for curr_ax in [ax, avg_ax]:
                for container in curr_ax.containers:
                    curr_ax.bar_label(container, fmt="%.2f", fontsize=8, rotation=90)
                curr_ax.tick_params(axis="x", rotation=90)
                curr_ax.set_xlabel(None)
                curr_ax.spines[["right", "top"]].set_visible(False)
            ax.legend().remove()  # loc='lower right')

        f.tight_layout()
        f.savefig(f"plots/{save_fname}.jpg", dpi=300, bbox_inches="tight")
        avg_f.tight_layout()
        avg_f.savefig(f"plots/{save_fname}_avg.jpg", dpi=300, bbox_inches="tight")

    def baselines_summarize_stats(self, important_only: bool = True) -> pd.DataFrame:
        # with important_only, we add focus on fairness metrics + overall acc
        baselines = self.baseline_numbers()
        summarized = baselines.groupby("method").mean("accuracy")
        if important_only:
            summarized = summarized[_IMPORTANT_METRICS]
        return summarized

    def k_vs_lamb(
        self,
        log_dir: str = "aug29_k_and_lamb_super_fine",
        nrows: int = 2,
        save_fname: str = "k_vs_lamb",
    ):
        df = self.collect_jsons_for_sweep(log_dir)
        df["k"] = df["predictor"].apply(lambda x: int(x.split("_")[-2]))
        df2 = df.groupby(["k", "lamb"]).mean("accuracy").reset_index()
        pivoted = df2.pivot(index="k", columns="lamb", values="accuracy")
        sns.heatmap(pivoted, annot=True, fmt=".4f")

        ncols = int(np.ceil(len(_IMPORTANT_METRICS / nrows)))
        f, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        for i, metric in enumerate(_IMPORTANT_METRICS):
            pivoted = df2.pivot(index="k", columns="lamb", values=metric)
            sns.heatmap(pivoted, annot=True, fmt=".4f", ax=axs[i % nrows, i // nrows])
            axs[i % nrows, i // nrows].set_title(metric.title())
        f.tight_layout()
        f.savefig(f"plots/{save_fname}.jpg", dpi=300, bbox_inches="tight")

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

    def beautify_dsetname(self, dsetname: str) -> str:
        return (
            dsetname.split("_full")[0]
            .replace("_", " ")
            .title()
            .replace("Mit", "MIT")
            .replace("0.8", "(Coarse)")
            .replace("0.9", "(Fine)")
            .replace("Thresh ", "")
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

    def acc_by_method_table(self):
        summary = self.baselines_summarize_stats()
        ours = self.our_best_method()
        # Idk why i need to do this groupby 'lamb', but do not fear, there is only one lamb value
        ours = ours.groupby("method").mean("accuracy")[_IMPORTANT_METRICS]
        summary = pd.concat([summary, ours])
        table_str = summary.to_latex(float_format="{:.2f}".format)
        # with open('for_paper/acc_by_method.txt', 'w') as f:
        #     f.write(table_str)
        print(table_str)

    def tables_method_by_dset_per_metric(
        self, save_root: str = "for_paper/tables/by_method_and_dset/"
    ):
        os.makedirs(save_root, exist_ok=True)
        baselines = self.baseline_numbers()
        ours = self.our_best_method()
        # ours['method'] = ['ours'] * len(ours)
        ours_with_base = pd.concat([baselines, ours])
        grouped = ours_with_base.groupby(["method", "dsetname"]).mean("accuracy")
        for metric in _IMPORTANT_METRICS:
            df = (
                grouped[metric]
                .reset_index()
                .pivot(index="method", columns="dsetname")[metric]
            )
            df = df.reindex(
                columns=[
                    "living17",
                    "entity13",
                    "entity30",
                    "nonliving26",
                    "dollarstreet__region",
                    "geode__region",
                    "mit_states_0.8",
                    "mit_states_0.9",
                ]
            )
            df = df.reindex(["vanilla", "dclip", "waffle", "chils", "ours"])
            df = df.rename(
                columns=self.beautify_dsetname, index=self.beautify_methodname
            )
            table_str = (
                df.style.highlight_max(axis=0, props="textbf:--rwrap;")
                .format(precision=2)
                .to_latex()
            )  # float_format="{:.2f}".format)
            print(table_str)
            # with open(f'{save_root}/{metric}.txt', 'w') as f:
            #     f.write(table_str)

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

    def adding_in_attributes(
        self,
        df_csv_path: str = "mazda_analysis/experiment_dfs/aug31_new_orders_0.csv"
        # log_dir: str = 'aug29_add_in_attrs_new_order',
        # predictors_with_lamb_to_show : List[str] = ['average_sims_1.0', 'chils_0.0', 'max_of_max_0.0', 'new_average_top_8_sims_0.0'],
        # metrics: List[str] = ['accuracy', 'worst class accuracy']
    ):
        # add_in_attrs = analyzer.collect_jsons_for_sweep('aug29_add_in_attrs_new_order')
        # add_in_attrs_base = analyzer.collect_jsons_for_sweep('aug29_add_in_attrs_new_order')
        # df = pd.concat(add_in_attrs, add_in_attrs_base)

        df = pd.read_csv(df_csv_path)

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


class GeographicShiftTable:
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
        self.table = self.make_table()

    def make_table(self) -> pd.DataFrame:
        table = self.df[
            self.df["dsetname"].isin(["dollarstreet__region", "geode__region"])
        ]
        # set method to categorical with custom order
        table["method"] = pd.Categorical(table["method"], self.METHOD_ORDER)
        table["vlm"] = pd.Categorical(table["vlm"], self.VLM_ORDER)
        table = table.sort_values(by=["vlm", "dsetname", "method"])[
            [
                "vlm",
                "dsetname",
                "method",
                "accuracy",
                "worst region",
                "worst income",
                "avg worst 20th percentile class accs",
            ]
        ]
        return table

    def to_latex(self) -> str:
        table_latex = self.table
        table_latex = table_latex.replace(np.nan, "-")
        table_latex = table_latex.replace("dollarstreet__region", "DollarStreet")
        table_latex = table_latex.replace("geode__region", "GeoDE")
        table_latex = table_latex.replace("clip_ViT-B/16", "CLIP")
        table_latex = table_latex.replace("blip2", "BLIP-2")
        table_latex = table_latex.rename(
            columns={"avg worst 20th percentile class accs": "worst 20\% of classes"}
        )
        return table_latex.to_latex(
            index=False,
            float_format="%.2f",
            formatters={"dsetname": lambda x: x.replace("_", " ")},
            caption="Performance on geographically diverse household object classification.",
        )


class NonGeographicDatasetsTable:
    METHOD_ORDER = ["vanilla", "dclip", "waffle", "chils", "ours"]
    VLM_ORDER = ["clip_ViT-B/16", "blip2"]

    def __init__(
        self,
        baselines: pd.DataFrame,
        ours: pd.DataFrame = None,
        vlm: str = "clip_ViT-B/16",
    ):
        """Makes a table containing baseline performance"""
        self.df = baselines
        self.vlm = vlm
        if ours is not None:
            self.df = pd.concat([baselines, ours], ignore_index=True)
        self.table = self.make_table()

    def make_table(self) -> pd.DataFrame:
        table = self.df[
            ~self.df["dsetname"].isin(["dollarstreet__region", "geode__region"])
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
            ]
        ]
        table = self.average_over_dataset_types(table)
        return table

    def average_over_dataset_types(self, table: pd.DataFrame) -> pd.DataFrame:
        return table.groupby(["dataset_type", "method"]).mean()

    def to_latex(self) -> str:
        table_latex = self.table
        table_latex = table_latex.rename(
            columns={"avg worst 20th percentile class accs": "Worst 20\% of classes"}
        )
        table_latex = table_latex.rename(
            columns={"average worst subpop accuracy": "Worst Subpopulation"}
        )
        table_latex = table_latex.rename(columns={"accuracy": "Accuracy"})
        table_latex.index = table_latex.index.map(lambda x: (x[0], x[1].upper()))
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
