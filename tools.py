import pandas as pd
from typing import List, Optional
from glob import glob
import os
from constants import _CACHED_DATA_ROOT
import json
from tqdm import tqdm
import submitit
from models.vlm import CLIP
from datasets import Breeds, DollarstreetDataset, MITStates
from sklearn.metrics import average_precision_score
from models.attributer import init_attributer, infer_attrs
from models.predictor import cos_sim, init_vlm_prompt_dim_handler
import numpy as np
from my_utils import cache_data, load_cached_data
from metrics import *
from main import *
from analysis import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr
import clip

def collect_results(log_dir: str) -> pd.DataFrame:
    log_dir_path = os.path.join(_CACHED_DATA_ROOT, 'experiments', log_dir, 'results', '*.json')    
    results_paths = glob(log_dir_path)

    with open(results_paths[0], 'r') as f:
        eg_results = json.load(f)
    keys = list(eg_results.keys())
    all_results = dict({k:[] for k in keys})

    for results_path in tqdm(results_paths):
        with open(results_path, 'r') as f:
            results = json.load(f)
        for key in keys: 
            all_results[key].append(results[key])
    
    df = pd.DataFrame(zip(*[all_results[key] for key in keys]), columns=keys)
    return df

def summarize(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None, 
    metrics: List[str] = ['accuracy', 'average worst subpop accuracy', 'worst class accuracy'],
    mode: str = 'mean'
):
    if cols is None:
        cols = [c for c in df.columns if c not in (metrics+['std dev worst subpop accuracy', 'Unnamed: 0', 'pred_classnames', 'identifiers'])]
    for c in cols:
        for curr, sub_df in df.groupby(c):
            if mode == 'mean':
                vals = [sub_df[metric].mean() for metric in metrics]
            elif mode == 'max':
                vals = [sub_df[metric].max() for metric in metrics]
            msg = f'{c:<25}: {curr:<60}, ' 
            msg += ', '.join([f"{metric.title()}: {val:.2f}" for metric, val in zip(metrics, vals)])
            print(msg)
            # print(f'{c:<25}: {curr:<60}, Acc: {sub_df["accuracy"].mean():.2f},'+
            #       f'AvgWSA: {sub_df["average worst subpop accuracy"].mean():.2f}')
        print()

def find_failed_runs(log_dir: str):
    jobs_dirs = glob(os.path.join(_CACHED_DATA_ROOT, 'experiments', log_dir, 'jobs', '*'))
    jobs_dirs = [f for f in jobs_dirs if '_' in f.split('/')[-1]]
    failed_runs, errors = [], []
    for job_dir in jobs_dirs:
        err_file = os.path.join(job_dir, job_dir.split('/')[-1]+'_0_log.err')
        with open(err_file, 'r') as f:
            lines = f.readlines()
        # if lines[-1] != 'INFO:submitit:Job completed successfully\n':
        if lines[-2][-24:] == 'Exited with exit code 1\n':#'INFO:submitit:Job completed successfully\n':
            print(''.join(lines))
            print()
            failed_runs.append(err_file)
            errors.append(lines[-4])
    return failed_runs, errors

def get_accs_by_class(output_dict):
    pred_classnames, ids, dset = [output_dict[x] for x in ['pred_classnames', 'identifiers', 'dset']]
    is_correct_by_id = mark_as_correct(pred_classnames, ids, dset)
    acc_by_class, _ = acc_by_class_and_subpop(is_correct_by_id, dset)
    return acc_by_class

def gains_by_class(ours_output_dict, baseline_output_dict):
    ours_accs_by_class, base_accs_by_class = [get_accs_by_class(x) for x in [ours_output_dict, baseline_output_dict]]

    gains_by_class = dict({c:ours_accs_by_class[c]-base_accs_by_class[c] for c in ours_accs_by_class})
    sorted_gains_by_class = dict(sorted(gains_by_class.items(), lambda x:-1*x[1]))

    f, ax = plt.subplots(1,1, figsize=(10,4))
    xticks, xticklabels = [], []
    for c in sorted_gains_by_class:
        ax.bar(2*i-0.4, base_accs_by_class[c], color='crimson')
        ax.bar(2*i+0.4, base_accs_by_class[c], color='lightgreen')
        xticks.append(2*i); xticklabels.append(c)
        i+=1
        if i > 10:
            break
    
    ax.set_xticks(xticks); ax.set_xticklabels(xticklabels)
    f.tight_layout(); f.savefig(f'plots/{save_fname}.jpg', dpi=300)


def diversity_accuracy_correlation(vlm):

    results = dict()
    cache_path = os.path.join(_CACHED_DATA_ROOT, 'diversity_acc_by_dset.pkl')
    for dset in tqdm([Breeds('living17'), Breeds('entity13'), Breeds('entity30'), Breeds('nonliving26'), 
                 MITStates(max_allowable_sim_of_classnames=0.8), MITStates(max_allowable_sim_of_classnames=0.9), 
                 DollarstreetDataset(), GeodeDataset()]):

        image_embeddings, identifiers = vlm.embed_all_images(dset)
        texts_by_subpop_by_class = infer_attrs(dset.classnames, [init_attributer('vanilla', dset, None)], ['USE OPENAI IMAGENET TEMPLATES'])
        text_embeddings_by_subpop_by_cls = vlm.embed_subpopulation_descriptions(texts_by_subpop_by_class)
        vlm_prompt_dim_handler = init_vlm_prompt_dim_handler('average_and_norm_then_stack')
        predictor = init_predictor('average_sims', lamb=0)
        # predictor = init_predictor('average_top_8_sims', lamb=0)
        pred_classnames, confidences = predictor.predict(image_embeddings, text_embeddings_by_subpop_by_cls, dset.classnames, vlm_prompt_dim_handler)
        is_correct_by_id = mark_as_correct(pred_classnames, identifiers, dset)
        acc_by_class, acc_by_subpop = acc_by_class_and_subpop(is_correct_by_id, dset)
        id_to_ind = dict({identifier:i for i, identifier in enumerate(identifiers)})
        accs, diversity_measures, cs_kept = [], [], []
        for c in dset.classnames:
            if c not in acc_by_class:
                continue
            accs.append(acc_by_class[c])
            cls_inds = [id_to_ind[identifier] for identifier in dset.ids_for_class(c)]
            cls_vecs = image_embeddings[cls_inds]
            diversity_measures.append((cls_vecs - cls_vecs.mean(0)).norm(dim=1).mean().item())
            cs_kept.append(c)
        
        r = pearsonr(accs, diversity_measures)[0]
        print(f'Pearson coefficient of correlation between Class Accuracy and Avg Dist to Mean: {r}')

        results[dset.get_dsetname()] = dict({'accs': accs, 'diversities': diversity_measures, 'classnames':cs_kept, 'r': r})
        cache_data(cache_path, results)
        # return accs, diversity_measures, r

def plot_diversity_scattters(dsetnames_to_plot='default', save_path='default'):
    analyzer = Analyze()
    results_dict = load_cached_data(_CACHED_DATA_ROOT+'diversity_acc_by_dset.pkl')
    accs, dsetnames, diversities = [],[],[]
    for dsetname in results_dict:
        curr_accs, curr_diversities = [results_dict[dsetname][x] for x in ['accs', 'diversities']]
        accs.extend(curr_accs)
        diversities.extend(curr_diversities)
        dsetnames.extend([dsetname]*len(curr_accs))
    df = pd.DataFrame(zip(dsetnames, accs, diversities), columns = ['Dataset', 'Class Accuracy', 'Class Diversity'])

    if dsetnames_to_plot == 'default':
        dsetnames_to_plot = list(results_dict.keys())
    ncols = len(dsetnames_to_plot) // 2
    f, axs = plt.subplots(2,ncols, figsize=(2*ncols,4), sharey=True, sharex=True)
    for i, dsetname in enumerate(dsetnames_to_plot):
        ax = axs[i // 4, i%4]
        sns.regplot(data=df[df.Dataset == dsetname], x="Class Diversity", y="Class Accuracy", ax=ax)
        ax.set_title(analyzer.beautify_dsetname(dsetname) + '\n$\mathbf{r='+f"{results_dict[dsetname]['r']:.2f}" + '}$', fontsize=11)

    if save_path == 'default':
        save_path = 'plots/acc_vs_diversity_scatters.jpg' 

    f.tight_layout()
    f.savefig(save_path, dpi=300, bbox_inches='tight')

    r_df = pd.DataFrame([(analyzer.beautify_dsetname(dsetname), v['r']) for dsetname, v in results_dict.items()], columns=['Dataset', "Pearson's $r$"])
    r_df = r_df.set_index('Dataset')
    table_str = r_df.to_latex(float_format="{:.2f}".format)
    table_save_path = save_path.replace('plots/', 'for_paper/')
    with open(table_save_path, 'w') as f:
        f.write(table_str)


def check_over_all_datasets():
    all_accs, all_divs, all_rs = [], [], []
    for dset in [Breeds('living17'), Breeds('entity13'), Breeds('entity30'), Breeds('nonliving26'), 
                 MITStates(max_allowable_sim_of_classnames=0.8), MITStates(max_allowable_sim_of_classnames=0.9), 
                 GeodeDataset(), DollarstreetDataset()]:
        accs, divs, r = diversity_accuracy_correlation(dset, vlm)
        all_accs.extend(accs)
        all_divs.extend(divs)
        all_rs.append(r)

# from main import *; dset = Breeds(); llm = Vicuna();attributers = [init_attributer(k, dset, llm) for k in ['vanilla', 'llm_kinds_chils']]
# texts_by_subpop_by_class = infer_attrs(dset.classnames, attributers, ['a photo of a {}']); vlm= CLIP('ViT-B/16')
# text_embeddings_by_subpop_by_cls = vlm.embed_subpopulation_descriptions(texts_by_subpop_by_class)
# vlm_prompt_dim_handler = init_vlm_prompt_dim_handler('stack_all')
# text_embeddings_by_cls, subpop_captions_by_cls = vlm_prompt_dim_handler.convert_to_embeddings_by_cls(text_embeddings_by_subpop_by_cls)


def ap_for_classname_vs_subpop(dset, vlm):
    cache_dir = os.path.join(_CACHED_DATA_ROOT, 'aps_by_class_and_subpop', vlm.get_modelname())
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, dset.get_dsetname()+'.pkl')

    aps_by_class_and_subpop = dict()
    image_embeddings, identifiers = vlm.embed_all_images(dset)

    attributers = [init_attributer(key, dset, None) for key in ['vanilla', 'groundtruth']]
    texts_by_subpop_by_class = infer_attrs(dset.classnames, attributers, ['USE OPENAI IMAGENET TEMPLATES'])
    text_embeddings_by_subpop_by_cls = vlm.embed_subpopulation_descriptions(texts_by_subpop_by_class)

    vlm_prompt_dim_handler = init_vlm_prompt_dim_handler('average_and_norm_then_stack')
    text_embeddings_by_cls, subpop_captions_by_cls = vlm_prompt_dim_handler.convert_to_embeddings_by_cls(text_embeddings_by_subpop_by_cls)

    for classname in dset.classnames:
        class_ids = dset.ids_for_class(classname)
        cls_embeddings = text_embeddings_by_cls[classname]
        sims = cos_sim(image_embeddings, cls_embeddings)
        classname_vec_ind = subpop_captions_by_cls[classname].index(classname)

        ys = [(identifier in class_ids) for identifier in identifiers]
        ap = average_precision_score(ys, sims[:, classname_vec_ind].detach().cpu().numpy())
        aps_by_class_and_subpop[(classname, classname)] = (ap, ap)
        # print(f'AP for detecting entire class using CLSNAME embedding: {ap:.3f}')

        for attr in dset.attrs_by_class[classname]:
            subpop_caption = dset.caption_gt_subpop(classname, attr)
            subpop_vec_ind = subpop_captions_by_cls[classname].index(subpop_caption)
            subpop_ids = dset.ids_for_subpop(classname, attr)
            idx = np.array([i for i, identifier in enumerate(identifiers) if ((identifier in subpop_ids) or (identifier not in class_ids))])
            ys = [(identifier in subpop_ids) for identifier in identifiers if ((identifier in subpop_ids) or (identifier not in class_ids))]
            ap_subpop_vec = average_precision_score(ys, sims[idx, subpop_vec_ind].detach().cpu().numpy())
            print(f'{subpop_caption}, {len(subpop_ids)} images')
            print(f'AP for SUBPOP  embedding: {ap_subpop_vec:.3f}')
            ap_clsname_vec = average_precision_score(ys, sims[idx, classname_vec_ind].detach().cpu().numpy())
            print(f'AP for CLSNAME embedding: {ap_clsname_vec:.3f}')
            print()

            aps_by_class_and_subpop[(classname, subpop_caption)] = (ap_subpop_vec, ap_clsname_vec)

            cache_data(cache_path, aps_by_class_and_subpop)


def visualize_utilized_subpops(dset, our_preds, base_preds, image_embeddings, identifiers, text_embeddings_by_cls, subpop_captions_by_cls):
    sims_by_class = dict({
        classname: cos_sim(image_embeddings, vecs) 
            for classname, vecs in text_embeddings_by_cls.items()
    })

    """
    What do we want to do?

    Per image, we want to visualize: 
    - the image itself
    - the predicted classname
    - the subpops with highest similarity for the predicted class (w/ the sim?)
    """

    idx = np.arange(len(identifiers))
    np.random.shuffle(idx)
    f, axs = plt.subplots(10, 10, figsize=(45, 45))
    for i in range(min(100, len(identifiers))):
        ax = axs[i // 10, i % 10]
    # f, axs = plt.subplots(2, 2, figsize=(9, 9))
    # for i in range(min(4, len(identifiers))):
    #     ax = axs[i // 2, i % 2]
        identifier, pred_cls, base_pred = [x[idx[i]] for x in [identifiers, our_preds, base_preds]]
        top_sims, top_inds = sims_by_class[pred_cls][idx[i]].topk(k=8)
        subpops_with_sims = [f"{subpop_captions_by_cls[pred_cls][ind]} ({sim:.3f})" for ind, sim in zip(top_inds, top_sims)]
        title = '\n'.join([identifier, f'vanilla pred: {base_pred}']+subpops_with_sims)
        ax.set_title(title, color='red' if pred_cls not in dset.valid_classnames_for_id(identifier) else 'black', fontsize=6)
        ax.set_axis_off()
        try:
            img = dset[identifier][0]
            ax.imshow(img.numpy().swapaxes(0,1).swapaxes(1,2))
        except:
            pass
    f.tight_layout(); f.savefig(f'plots/examples/{dset.get_dsetname()}.jpg', dpi=300)

def heuristic_subpop_to_attr(subpop: str, classname: str):
    subpop = subpop.lower()
    if ' which is ' in subpop: # 
        return subpop.split(' which is ')[-1]
    elif ' which has ' in subpop:
        return subpop.split(' which has ')[-1]
    # elif ' from a ' in subpop:
    #     return subpop[subpop.index('from a'):]
    elif 'from the country' in subpop:
        return 'from ' + subpop.split(' from the country ')[-1]
    elif 'from the region' in subpop:
        return 'from ' + subpop.split(' from the region ')[-1]
    elif ' in the background' in subpop:
        return subpop[subpop.index('with a'):].split(' in the background')[0] 
    else:
        return subpop.replace(classname+' '+classname, classname)#.strip()
    
def save_one_eg_inference(img, our_pred, base_pred, subpops, save_path):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": "cmss10",
            "axes.formatter.use_mathtext": "True",
            "mathtext.fontset": "stixsans"
        }
    )

    f, ax = plt.subplots(1,1, figsize=(3,3.75))
    ax.imshow(img.numpy().swapaxes(0,1).swapaxes(1,2))
    _ = [ax.spines[spine].set_visible(False) for spine in ['right', 'left', 'top', 'bottom']]
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'Standard prediction: {base_pred.title()} $X$', color='red', fontsize=13)
    # ax.set_xlabel(f'Standard prediction: {base_pred.title()} $X$', color='red', fontsize=13)
    xlabel = f'Ours: {our_pred.title()} $\checkmark$, namely...'

    attrs = [heuristic_subpop_to_attr(s, our_pred) for s in subpops]
    for i, attr in enumerate(attrs):
        # if i % 2 == 0:
        #     xlabel += '\n'
        # if i < 4:
            # xlabel += ', '
        xlabel += '\n$\mathbf{'+attr.replace(' ','\;').replace('flowers', 'flower')+'}$'
    ax.set_xlabel(xlabel, color='green', fontsize=15)
    # ax.set_title(xlabel, color='green', fontsize=13)
    f.tight_layout(); f.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_many_eg_inferences(dset, our_preds, base_preds, image_embeddings, identifiers, text_embeddings_by_cls, subpop_captions_by_cls, k_to_show=4):
    """
    ths is going to show 
    """
    
    sims_by_class = dict({
        classname: cos_sim(image_embeddings, vecs) 
            for classname, vecs in text_embeddings_by_cls.items()
    })

    save_dir = f'plots/eg_inferences7/{dset.get_dsetname()}/'
    os.makedirs(save_dir, exist_ok=True)

    jobs = []
    # executor = submitit.AutoExecutor(folder=_CACHED_DATA_ROOT+'eg_inferences')
    # executor.update_parameters(timeout_min=180, slurm_partition="learnlab,devlab", mem_gb=80, gpus_per_node=1, tasks_per_node=1, slurm_constraint='volta32gb,ib4')

    # with executor.batch():
    for j, (identifier, our_pred, base_pred) in enumerate(zip(identifiers, our_preds, base_preds)):
        f, ax = plt.subplots(1,1, figsize=(3,3.75))
        img = dset[identifier][0]
        top_sims, top_inds = sims_by_class[our_pred][j].topk(k=k_to_show)        
        subpops = [subpop_captions_by_cls[our_pred][ind] for ind in top_inds]

        save_path = save_dir+f'{our_pred.replace(" ","_")}__{j}.jpg'
        save_one_eg_inference(img, our_pred, base_pred, subpops, save_path)
            # jobs.append(executor.submit(save_one_eg_inference, (img, our_pred, base_pred, subpops, save_path)))


        # ax.imshow(img.numpy().swapaxes(0,1).swapaxes(1,2))
        # _ = [ax.spines[spine].set_visible(False) for spine in ['right', 'left', 'top', 'bottom']]
        # ax.set_xticks([]); ax.set_yticks([])
        # ax.set_title(f'Standard prediction: {base_pred.title()} $X$', color='red', fontsize=13)
        # xlabel = f'Ours: {our_pred.title()} $\checkmark$, namely...'

        # attrs = [heuristic_subpop_to_attr(s, our_pred) for s in subpops]
        # for i, subpop in enumerate(attrs):
        #     if i % 2 == 0:
        #         xlabel += '\n'
        #     xlabel += '$\mathit{'+subpop.replace(' ','\;')+'}$'
        #     if i < 3:
        #         xlabel += ', '
        # ax.set_xlabel(xlabel, color='green', fontsize=13)
        # f.tight_layout(); f.savefig(save_dir+f'{our_pred.replace(" ","_")}__{j}.jpg', dpi=300, bbox_inches='tight')


def compare_accs(our_preds, base_preds, identifiers, dset, save_path=None):
    base_acc_by_class, base_acc_by_subpop = acc_by_class_and_subpop(mark_as_correct(base_preds, identifiers, dset), dset)
    our_acc_by_class, our_acc_by_subpop = acc_by_class_and_subpop(mark_as_correct(our_preds, identifiers, dset), dset)

    subpop_gains, class_gains = dict(), dict()
    for c in base_acc_by_subpop:
        class_gains[c] = our_acc_by_class[c] - base_acc_by_class[c]
        for s in base_acc_by_subpop[c]:
            subpop_gains[(c,s)] = our_acc_by_subpop[c][s] - base_acc_by_subpop[c][s]

    class_gains, subpop_gains = [dict(sorted(gains.items(), key=lambda x: -1*x[1])) for gains in [class_gains, subpop_gains]]
    for c in list(class_gains.keys())[:100]:
        print(f'Class: {c:<50}, Our acc: {our_acc_by_class[c]:.2f}, Gain: {class_gains[c]:.2f}')

    for c,s in list(subpop_gains.keys())[:100]:
        subpop_cap = dset.caption_gt_subpop(c, s)
        print(f'Subpop: {subpop_cap:<50}, Our acc: {our_acc_by_subpop[c][s]:.2f}, Gain: {subpop_gains[(c,s)]:.2f}')
    
    if save_path:
        d = dict({
            'ours': dict({'by_class': our_acc_by_class, 'by_subpop': our_acc_by_subpop}),
            'base': dict({'by_class': base_acc_by_class, 'by_subpop': base_acc_by_subpop}),
            'identifiers': identifiers, 'our_preds': our_preds, 'base_preds': base_preds,
            'gains': dict({'by_class': class_gains, 'by_subpop': subpop_gains})
        })
        cache_data(save_path, d)

def compare_ours_to_base(dset, vlm, classes_to_focus_on=None):
    image_embeddings, identifiers = vlm.embed_all_images(dset)
    if classes_to_focus_on:
        ids_to_keep = []
        for classname in classes_to_focus_on:
            ids_to_keep.extend(dset.ids_for_class(classname))
        ids_to_keep = list(set(ids_to_keep))
        inds_to_keep = np.array([i for i, identifier in enumerate(identifiers) if identifier in ids_to_keep])
        image_embeddings = image_embeddings[inds_to_keep]
        identifiers = [identifiers[i] for i in inds_to_keep]

    attr_keys = ['auto_global', 'country', 'income_level', 'llm_co_occurring_objects', 'llm_backgrounds', 'llm_dclip', 'llm_kinds', 'llm_states', 'region', 'vanilla']
    # attr_keys = ['auto_global', 'country', 'income_level', 'llm_co_occurring_objects', 'llm_dclip', 'llm_kinds', 'llm_states', 'region', 'vanilla']
    llm = Vicuna()
    attributers = [init_attributer(key, dset, llm) for key in attr_keys]
    texts_by_subpop_by_class = infer_attrs(dset.classnames, attributers, ['USE OPENAI IMAGENET TEMPLATES'])
    # texts_by_subpop_by_class = infer_attrs(dset.classnames, attributers, ['a photo of a {}'])#'USE OPENAI IMAGENET TEMPLATES'])
    vlm_prompt_dim_handler = init_vlm_prompt_dim_handler('average_and_norm_then_stack')
    text_embeddings_by_subpop_by_cls = vlm.embed_subpopulation_descriptions(texts_by_subpop_by_class)
    text_embeddings_by_cls, subpop_captions_by_cls = vlm_prompt_dim_handler.convert_to_embeddings_by_cls(text_embeddings_by_subpop_by_cls)
    predictor  = init_predictor('average_top_16_sims', lamb=0)
    our_preds, _ = predictor.predict(image_embeddings, text_embeddings_by_subpop_by_cls,
                                    dset.classnames, vlm_prompt_dim_handler)

    vanilla_embeddings_by_subpop_by_cls = dict({c:dict({c:v[c]}) for c,v in text_embeddings_by_subpop_by_cls.items()})
    base_preds, _ = predictor.predict(image_embeddings, vanilla_embeddings_by_subpop_by_cls,
                                    dset.classnames, vlm_prompt_dim_handler)

    save_dir = os.path.join(_CACHED_DATA_ROOT, 'accs_by_class_and_subpop', vlm.get_modelname())
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, dset.get_dsetname()+'.pkl')
    # compare_accs(our_preds, base_preds, identifiers, dset, save_path=save_path)

    diff_inds = [i for i, (p1, p2) in enumerate(zip(base_preds, our_preds)) if ((p1 != p2) and (p2 in dset.valid_classnames_for_id(identifiers[i])))]
    identifiers, our_preds, base_preds = [[x for i,x in enumerate(l) if i in diff_inds] for l in [identifiers, our_preds, base_preds]]
    image_embeddings = image_embeddings[np.array(diff_inds)]
    save_many_eg_inferences(dset, our_preds, base_preds, image_embeddings, identifiers, text_embeddings_by_cls, subpop_captions_by_cls)

    # visualize_utilized_subpops(dset, our_preds, base_preds, image_embeddings, identifiers, text_embeddings_by_cls, subpop_captions_by_cls)

def visualize_bias(dset, d, classname, attr, ids_we_fixed):
    # My favs are 3 for balloon, 5 for arctic fox, and 1 for streetview
    # 2,3 for keyboard and penguin. 

    # f, axs = plt.subplots(1,2,figsize=(7,3.75))
    f, axs = plt.subplots(1,2,figsize=(6,4))
    dsetname = dset.get_dsetname()

    if 'mit_states' in dsetname:
        best_attr = 'typical' 
    elif 'dollarstreet' in dsetname:
        best_attr = 'rich'
    else:
        attrs_for_class = list(d['base']['by_subpop'][classname].keys())
        best_attr = attrs_for_class[np.argmax(list(d['base']['by_subpop'][classname].values()))]

    hard_ids = [idtfr for idtfr in dset.ids_for_subpop(classname, attr) if idtfr in ids_we_fixed]

    try:
        easy_img_in_class = dset[dset.ids_for_subpop(classname, best_attr)[1]][0]
        hard_img_in_class = dset[hard_ids[min(4, len(hard_ids)-1)]][0]
    except:
        print(f'Currently dont have images for the class {classname} in {dsetname} saved')
        return

    base_easy_acc, our_easy_acc = [d[method]['by_class'][classname] for method in ['base', 'ours']]
    base_hard_acc, our_hard_acc = [d[method]['by_subpop'][classname][attr] for method in ['base', 'ours']]

    axs[0].imshow(easy_img_in_class.numpy().swapaxes(0,1).swapaxes(1,2))
    # axs[0].set_title(f'Base Accuracy: {base_easy_acc:.2f}%', fontsize=16)
    # axs[0].set_xlabel(f'Our Accuracy: {our_easy_acc:.2f}%', fontsize=16, fontweight='bold')
    axs[0].set_title(f'Base Acc.: {base_easy_acc:.2f}%', fontsize=19)
    axs[0].set_xlabel(f'Our Acc.: {our_easy_acc:.2f}%', fontsize=19, fontweight='bold')

    gain = f'{(our_hard_acc - base_hard_acc):.1f}'
    axs[1].imshow(hard_img_in_class.numpy().swapaxes(0,1).swapaxes(1,2))
    # axs[1].set_title(f'Base Accuracy: {base_hard_acc:.2f}%', color='red', fontsize=16, fontweight='bold')
    # axs[1].set_xlabel(f'Our Accuracy: {our_hard_acc:.2f}%'+ '($\mathbf{+'+gain+'}$)', color='forestgreen', fontsize=16, fontweight='bold')
    axs[1].set_title(f'Base Acc.: {base_hard_acc:.2f}%', color='red', fontsize=19, fontweight='bold')
    axs[1].set_xlabel(f'Our Acc.: {our_hard_acc:.2f}%'+ '($\mathbf{+'+gain+'}$)', color='forestgreen', fontsize=19, fontweight='bold')

    for ax in axs:
        _ = [ax.spines[x].set_visible(False) for x in ['top', 'bottom', 'left', 'right']]
        ax.set_xticks([]); ax.set_yticks([])

    attr = attr.replace(' '+classname, '')

    title = f"{classname.title()} vs \n"
    if 'dollarstreet' in dsetname:
        title += f"{classname.title()} " +"$\mathbf{from\;"
        if dset.attr_col == 'income_level':
            title += f"a\;{attr}\;country"+"}$"
            title = title.replace('poor', 'low\;income').replace('lower middle class', 'lower\;middle\;income')
        else:
            title += attr+"}$"
    else:
        title += "$\mathbf{"+attr.title()+"}$"+f" {classname.title()}"
    f.suptitle(title, fontsize=23, y=1)
    save_path = f'plots/bias_examples2/{dsetname}/{attr.replace(" ","_")}__{classname.replace("/" or " ", "_")}.jpg'
    os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
    f.tight_layout(); f.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(save_path)

def construct_ap_and_acc_gain_df(dset):
    dsetname = dset.get_dsetname()
    accs_and_preds = load_cached_data(f'{_CACHED_DATA_ROOT}/accs_by_class_and_subpop/clip__ViT-B_16/{dsetname}.pkl')
    aps = load_cached_data(f'{_CACHED_DATA_ROOT}/aps_by_class_and_subpop/clip__ViT-B_16/{dsetname}.pkl')

    identifiers, our_preds, base_preds = [accs_and_preds[key] for key in ['identifiers', 'our_preds', 'base_preds']]
    ids_we_fixed = [idtfr for idtfr, p1, p2 in zip(identifiers, base_preds, our_preds) if ((p1 != p2) and (p2 in dset.valid_classnames_for_id(idtfr)))]

    clsnames, subpops, attrs, classname_aps, subpop_aps, ap_gains, acc_gains, base_accs = [],[],[],[],[],[],[],[]
    for (classname, attr), acc_gain in accs_and_preds['gains']['by_subpop'].items():
        clsnames.append(classname)
        attrs.append(attr)
        subpop = dset.caption_gt_subpop(classname, attr)
        subpops.append(subpop)
        subpop_ap, classname_ap = aps[(classname, subpop)]
        classname_aps.append(classname_ap); subpop_aps.append(subpop_ap)
        ap_gains.append(subpop_ap-classname_ap)
        acc_gains.append(acc_gain)
        base_accs.append(accs_and_preds['base']['by_subpop'][classname][attr])

    df = pd.DataFrame(zip(clsnames, subpops, attrs, ap_gains, acc_gains, classname_aps, subpop_aps, base_accs), 
            columns=['classname', 'subpop', 'attr', 'ap_gain', 'acc_gain', 'classname_ap', 'subpop_ap', 'base_acc'])
    return df, ids_we_fixed, accs_and_preds

def save_many_bias_examples(dset, min_ap=0.2, min_gain=8):
    df, ids_we_fixed, accs_and_preds = construct_ap_and_acc_gain_df(dset)
    sub_df = df[(df.ap_gain > min_ap) & (df.acc_gain > min_gain)]
    for i, row in sub_df.iterrows():
        classname, attr = [row[x] for x in ['classname', 'attr']]
        visualize_bias(dset, accs_and_preds, classname, attr, ids_we_fixed)
    # if 'balloon' in dset.classnames:
    #     visualize_bias(dset, accs_and_preds, 'balloon', 'deflated', ids_we_fixed) # use id 3


def visualize_ap_gain(dset, classname, attr, classname_ap, subpop_ap, acc_gain, ids_we_fixed):
    f, ax = plt.subplots(1,1, figsize=(3,4))
    dsetname = dset.get_dsetname()
    eg_ids = [idtfr for idtfr in dset.ids_for_subpop(classname, attr) if idtfr in ids_we_fixed]
    try:
        eg_img = dset[eg_ids[2]][0]
    except:
        print(f'Not enough improved images for {attr} {classname} or images not found.')
        return
    
    ax.imshow(eg_img.numpy().swapaxes(0,1).swapaxes(1,2))
    _ = [ax.spines[x].set_visible(False) for x in ['top', 'bottom', 'left', 'right']]
    ax.set_xticks([]); ax.set_yticks([])

    if 'dollarstreet' in dsetname or 'geode' in dsetname:
        title = classname.title()
        if dset.attr_col == 'income_level':
            title += "$\mathbf{\;from\;a}$\n$\mathbf{"+f"{attr}\;country"+"}$"
            title = title.replace('poor', 'low\;income').replace('lower middle class', 'lower\;middle\;income')
            # title += "$\mathbf{\;from\;a}$\n$\mathbf{"+f"{attr}\;country"+"}$"
        else:
            title += "\n$\mathbf{\;from\;"+attr+"}$"
    else:
        title = "$\mathbf{"+attr.replace(' '+classname, '').title().replace(' ','\;')+"}$\n"+f"{classname.title()}"
    ax.set_title(title, fontsize=18)
    # xlabel = "{x:<15} AP: $\mathbf{}"

    ax.text(10, 240, "Classname alone", fontsize=15)
    ax.text(150, 240, f"AP: {classname_ap*100:5.1f}", fontsize=15, color='red')
    ax.text(10, 259, "with $\mathbf{Attribute}$", fontsize=15)
    ax.text(150, 259, f"AP: {subpop_ap*100:5.1f}", fontsize=15, color='green')
    ax.text(15, 278, "$\\rightarrow$ Our Acc. Gain: $\mathbf{+"+f"{acc_gain:.1f}"+"}$", fontsize=15, color='green')


    # ax.text(1, 240, "{x:<16} AP: {ap:4.1f}".format(x="Classname alone", ap=classname_ap*100), fontsize=14)    
    # ax.text(1, 259, "{x:<18} AP: {ap:4.1f}".format(x="with Attribute", ap=subpop_ap*100), fontsize=16, color='green')
    # ax.text(12, 278, "$\\rightarrow$ Our Acc. Gain: $\mathbf{+"+f"{acc_gain:.1f}"+"}$", fontsize=16, color='green')

    # xlabel = '\n'.join([f"{x:<12} AP: {ap*100:.1f}" for x,ap in zip(["Classname", "+ Attribute"], [classname_ap, subpop_ap])])
    # xlabel += '\n' + f'Our Acc. Gain: +{acc_gain:.2f}'
    # neg = subpop_ap < classname_ap
    # ax.set_xlabel(xlabel, color="red" if neg else "black", fontsize=14)
    save_dir = f'plots/ap_gain_examples2/{dsetname}'
    os.makedirs(save_dir, exist_ok=True)
    # save_path = f'{save_dir}/{"bad__" if neg else ""}{attr.replace("/" or " ","_")}__{classname.replace("/" or " ", "_")}.jpg'
    save_path = f'{save_dir}/{attr.replace("/" or " ","_")}__{classname.replace("/" or " ", "_")}.jpg'
    os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
    f.tight_layout(); f.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def view_many_ap_gain_egs(dset, min_ap=0.2, min_gain=5):
    df, ids_we_fixed, _ = construct_ap_and_acc_gain_df(dset)
    sub_df = df[(df.ap_gain > min_ap) & (df.acc_gain > min_gain)]
    for i, row in sub_df.iterrows():
        classname, attr, classname_ap, subpop_ap, acc_gain = [row[x] for x in ['classname', 'attr', 'classname_ap', 'subpop_ap', 'acc_gain']]
        visualize_ap_gain(dset, classname, attr, classname_ap, subpop_ap, acc_gain, ids_we_fixed)


def compare_to_linear_probe(subsample_factor=None):
    val_dset = Breeds()
    train_dset = Breeds(inet_split='train')
    vlm = CLIP('ViT-B/16')
    c_to_ind = dict({c:i for i,c in enumerate(val_dset.classnames)})
    xs, ys, ids, arctic_fox_inds, fox_inds = dict(), dict(), dict(), dict(), dict()
    for split, dset in zip(['train', 'val'], [train_dset, val_dset]):
        img_embeddings, ids[split] = vlm.embed_all_images(dset)

        if split == 'train' and subsample_factor:
            subset = np.arange(0,img_embeddings.shape[0], subsample_factor)
            img_embeddings = img_embeddings[subset]
            ids[split] = [identifier for i, identifier in enumerate(ids[split]) if i in subset]

        xs[split] = img_embeddings.detach().cpu().numpy()
        ys[split] = np.array([c_to_ind[dset.data_df.loc[identifier]['valid_classnames'][0]] for identifier in ids[split]])
        arctic_fox_ids = dset.ids_for_subpop('fox', 'Arctic fox')
        arctic_fox_inds[split] = np.array([i for i,identifier in enumerate(ids[split]) if identifier in arctic_fox_ids])
        fox_ids = dset.ids_for_class('fox')
        fox_inds[split] = np.array([i for i,identifier in enumerate(ids[split]) if identifier in fox_ids])

    non_arctic_fox_inds = [i for i in range(xs['train'].shape[0]) if i not in arctic_fox_inds['train']]

    results = []
    # for frac_arctic_fox in [0.01, 0.02, 0.035, 0.05, 0.1, 0.15, 0.2, 0.25]:
    for frac_arctic_fox in [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25]:
        num_train_arctic_foxes = int(frac_arctic_fox * len(fox_inds['train']))
        # print(num_train_arctic_foxes, non_arctic_fox_inds)
        new_train_inds = np.array(non_arctic_fox_inds + list(arctic_fox_inds['train'][:num_train_arctic_foxes]))
        # new_train_inds = np.concatenate(non_arctic_fox_inds, arctic_fox_inds['train'][:num_train_arctic_foxes])
        train_x, train_y = [full[new_train_inds] for full in [xs['train'], ys['train']]]

        clf = LogisticRegression(random_state=0, max_iter=1000).fit(train_x, train_y)
        acc = clf.score(xs['val'], ys['val'])
        fox_acc = clf.score(xs['val'][fox_inds['val']], ys['val'][fox_inds['val']])
        arctic_fox_acc = clf.score(xs['val'][arctic_fox_inds['val']], ys['val'][arctic_fox_inds['val']])

        print(f"Num arctic foxes: {num_train_arctic_foxes} ({int(frac_arctic_fox*100)}%). Acc: {100*acc:.2f}%, Fox Acc: {fox_acc*100:.2f}%, Arctic Fox Acc: {arctic_fox_acc*100:.2f}%")
        results.append([frac_arctic_fox, num_train_arctic_foxes, acc, fox_acc, arctic_fox_acc])
    
    results = np.array(results)
    columns = ['frac_arctic_fox', 'num_train_arctic_foxes', 'acc', 'fox_acc', 'arctic_fox_acc']
    df = pd.DataFrame(dict({c:results[:,i] for i,c in enumerate(columns)}))
    df.to_csv(_CACHED_DATA_ROOT+'arctic_fox_case_study_few_shot.csv')

def plot_comparison_to_linear_probe():
    _ZERO_SHOT_BIAS = 32.5 ### DANGER DANGER HARD CODED NUMBER!!!
    df = pd.read_csv(_CACHED_DATA_ROOT+'arctic_fox_case_study_2.csv')
    df['bias'] = 100*(df['fox_acc'] - df['arctic_fox_acc'])
    df['frac_arctic_fox'] = df['frac_arctic_fox'] * 100
    f, ax = plt.subplots(1,1, figsize=(3.5,3.5))
    ax.set_facecolor((0.93,0.93,0.93))
    ax.grid(color='white', zorder=0)
    ax.hlines(y=32.5, xmin=0, xmax=df['frac_arctic_fox'].max(), zorder=5, label='Zero-shot Bias', color='red', ls='--')
    ax.legend()
    # ax.plot(df["frac_arctic_fox"], df["bias"], )
    sns.lineplot(data=df, x="frac_arctic_fox", y="bias", markers=True, zorder=6, marker="*")
    # ax.set_xlabel("Ratio of Arctic Fox to Fox Images\nin Linear Probe Training Set", fontsize=14)
    ax.set_xlabel("% of Foxes that are Arctic Foxes\nin Linear Probe Training Set", fontsize=14)
    ax.set_ylabel("Bias (Fox Acc. - Arctic Fox Acc.)", fontsize=14)
    ax.set_ylim([0,42])
    f.tight_layout(); f.savefig('plots/compare_linear_probe.jpg', dpi=300, bbox_inches='tight')


def plot_ap_gain_hists():
    from analysis import Analyze
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": "cmss10",
    #         "axes.formatter.use_mathtext": "True",
    #         "mathtext.fontset": "stixsans"
        }
    )

    analyzer = Analyze()
    all_dfs = []
    for dset in [Breeds('entity13'), Breeds('entity30'), Breeds('nonliving26'), Breeds('living17'), 
                MITStates(max_allowable_sim_of_classnames=0.9), MITStates(max_allowable_sim_of_classnames=0.8)]:#, 
                # DollarstreetDataset('income_level', max_allowable_sim_of_classnames=0.9), GeodeDataset()]:
        df, ids_we_fixed, accs_and_preds = construct_ap_and_acc_gain_df(dset)
        df['dsetname'] = [analyzer.beautify_dsetname(dset.get_dsetname())]*len(df)
        all_dfs.append(df)

    df = pd.concat(all_dfs)
    f, ax = plt.subplots(1,1, figsize=(5,4))
    ax = sns.stripplot(df, x='ap_gain', y='dsetname', hue='dsetname', legend=True, jitter=False, s=10, marker="D", linewidth=1, alpha=.05)
    # ax = sns.swarmplot(df, x='ap_gain', y='dsetname', hue='dsetname', legend=True, size=2,#stat='density', bins=20,
    #     hue_order=dsets_in_order)#, palette=colors)#sns.color_palette("Greens", as_cmap=False))
    # sns.histplot(df, x='ap_gain', hue='dsetname', ax=ax, legend=True, bins=20)
    sns.move_legend(ax, "upper left", title='Dataset', frameon=False, fontsize=15, bbox_to_anchor=(-0.07, 1), handletextpad=0.01)
    # ax.set_xlabel('AP of Classname & Attribute \n - AP of Classname Alone', fontsize=14)
    ax.set_xlabel('AP Gain from Adding Attribute', fontsize=16)
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.set_xlim([-1,1])
    ax.set_xticks([-1,-0.5,0,0.5,1])
    f.savefig('plots/ap_gains2.jpg', dpi=300, bbox_inches='tight')


def obtain_zero_shot_head(clip_model, dset, prompt_template='a photo of a {}'):
    text_prompts = torch.cat([clip.tokenize(prompt_template.format(clsname)) for clsname in dset.classnames]).cuda()
    clip_model = clip_model.cuda()
    text_ftrs = clip_model.encode_text(text_prompts)
    text_ftrs /= text_ftrs.norm(dim=-1, keepdim=True)
    return text_ftrs

def record_problem_classes(dsetname='mit_states', thresh=0.85):
    '''
    Some classes are arguably overlapping. We'll remove ones that have CLIP text similarity above 0.9.
    Specifically, for any pair of classes w/ CLIP similarity > 0.9, we remove the class with higher average
    CLIP similarity to all classes, as these are likely more broad and potentially overlapping w. others. 
    '''
    ### Compute similarity of text embeddings for each class name
    if dsetname == 'mit_states':
        dset = MITStates(max_allowable_sim_of_classnames=1)
    elif dsetname == 'dollarstreet':
        dset = DollarstreetDataset()
    n_classes = len(dset.classnames)
    clip_model, _ = clip.load('ViT-B/16')
    head = obtain_zero_shot_head(clip_model, dset)
    sims = (head @ head.T)
    sims = sims - torch.eye(sims.shape[0]).cuda() * torch.diag(sims)
    sorted_sims = torch.argsort(-1*sims.flatten())

    # For each pair w/ sim > thresh, add broader class to problematic class list
    problem_classes = []
    for i in tqdm(sorted_sims):
        if sims[i // n_classes, i % n_classes] <= thresh:
            break
        # if one of the pair is already added, we don't need to add a new one
        if (i//n_classes not in problem_classes) and (i%n_classes not in problem_classes):
            c1, c2 = [dset.classnames[j] for j in [i//n_classes, i%n_classes]]
            # now we need to remove one of two colliding classes; we'll remove the one most similar to everyone else
            if sims[i//n_classes].mean() > sims[i%n_classes].mean():
                problem_classes.append(i//n_classes)
                kept_c = c2
            else:
                problem_classes.append(i%n_classes)
                kept_c = c1

            print(f'Collision between {c1} and {c2}. Kept {kept_c}')
    # we'll need to filter by name, so let's save by name
    print(f'{len(problem_classes)} problematic classes found out of {n_classes}.')
    problem_class_names = [dset.classnames[i] for i in problem_classes]
    cache_data(f'/checkpoint/mazda/data/meta_files/{dsetname}_problem_classes_thresh_{thresh}.pkl', problem_class_names)

def diversity_accuracy_correlation(vlm):

    results = dict()
    cache_path = os.path.join(_CACHED_DATA_ROOT, 'diversity_acc_by_dset.pkl')
    for dset in tqdm([Breeds('living17'), Breeds('entity13'), Breeds('entity30'), Breeds('nonliving26'), 
                 MITStates(max_allowable_sim_of_classnames=0.8), MITStates(max_allowable_sim_of_classnames=0.9), 
                 DollarstreetDataset(), GeodeDataset()]):

        image_embeddings, identifiers = vlm.embed_all_images(dset)
        texts_by_subpop_by_class = infer_attrs(dset.classnames, [init_attributer('vanilla', dset, None)], ['USE OPENAI IMAGENET TEMPLATES'])
        text_embeddings_by_subpop_by_cls = vlm.embed_subpopulation_descriptions(texts_by_subpop_by_class)
        vlm_prompt_dim_handler = init_vlm_prompt_dim_handler('average_and_norm_then_stack')
        predictor = init_predictor('average_sims', lamb=0)
        # predictor = init_predictor('average_top_8_sims', lamb=0)
        pred_classnames, confidences = predictor.predict(image_embeddings, text_embeddings_by_subpop_by_cls, dset.classnames, vlm_prompt_dim_handler)
        is_correct_by_id = mark_as_correct(pred_classnames, identifiers, dset)
        acc_by_class, acc_by_subpop = acc_by_class_and_subpop(is_correct_by_id, dset)
        id_to_ind = dict({identifier:i for i, identifier in enumerate(identifiers)})
        accs, diversity_measures, cs_kept = [], [], []
        for c in dset.classnames:
            if c not in acc_by_class:
                continue
            accs.append(acc_by_class[c])
            cls_inds = [id_to_ind[identifier] for identifier in dset.ids_for_class(c)]
            cls_vecs = image_embeddings[cls_inds]
            diversity_measures.append((cls_vecs - cls_vecs.mean(0)).norm(dim=1).mean().item())
            cs_kept.append(c)
        
        r = pearsonr(accs, diversity_measures)[0]
        print(f'Pearson coefficient of correlation between Class Accuracy and Avg Dist to Mean: {r}')

        results[dset.get_dsetname()] = dict({'accs': accs, 'diversities': diversity_measures, 'classnames':cs_kept, 'r': r})
        cache_data(cache_path, results)
        # return accs, diversity_measures, r


if __name__ == '__main__':
    executor = submitit.AutoExecutor(folder=_CACHED_DATA_ROOT+'/aps_by_class_and_subpop_experiments/')
    executor.update_parameters(timeout_min=180, slurm_partition="learnlab,devlab", mem_gb=80, gpus_per_node=1, tasks_per_node=1, slurm_constraint='volta32gb,ib4')

    # executor.update_parameters(timeout_min=180, slurm_partition="cml-dpart", slurm_qos="cml-default", slurm_account="cml-sfeizi", mem_gb=32, gpus_per_node=1, tasks_per_node=1)
    jobs = []
    # vlm = CLIP('ViT-B/16')
    # diversity_accuracy_correlation(vlm)

    # dsets = []
    # dsets += [Breeds(x) for x in ['living17']]#'nonliving26', 'living17', 'entity13', 'entity30']]
    # # dsets += [Breeds(x) for x in ['nonliving26', 'living17', 'entity13', 'entity30']]
    # dsets += [MITStates(max_allowable_sim_of_classnames=thresh) for thresh in [0.8, 0.9]]
    # dsets += [DollarstreetDataset(attr_col='income_level', max_allowable_sim_of_classnames=0.9)]#, GeodeDataset()]
    # dsets = [DollarstreetDataset(attr_col='income_level', max_allowable_sim_of_classnames=0.9), GeodeDataset()]
    # with executor.batch():
    # for dset in dsets:
        # save_many_bias_examples(dset)
        # view_many_ap_gain_egs(dset)
            # jobs.append(executor.submit(ap_for_classname_vs_subpop, dset, vlm))
            # jobs.append(executor.submit(compare_ours_to_base, dset, vlm))
            # jobs.append(executor.submit(view_many_ap_gain_egs, dset))
            # ap_for_classname_vs_subpop(dset, vlm)
            # compare_ours_to_base(dset, vlm)


    #     dset = Breeds('living17')
    # # compare_ours_to_base(dset, vlm, ["fox", "ape", "wolf"])
    #     job = executor.submit(compare_ours_to_base, dset, vlm, ["fox", "ape", "wolf"])

    # dset = MITStates(max_allowable_sim_of_classnames=0.9)
    # compare_ours_to_base(dset, vlm, ["tulip", "pear", "balloon", "tomato", "clock"])
        # jobs.append(executor.submit(compare_ours_to_base, dset, vlm, ["tulip", "pear", "balloon", "tomato", "clock", "canyon", "shower", "velvet"]))

    #     dset = MITStates(max_allowable_sim_of_classnames=0.8)
    # # # compare_ours_to_base(dset, vlm, ["tulip", "pear", "balloon", "tomato", "clock"])
    #     jobs.append(executor.submit(compare_ours_to_base, dset, vlm, ["tulip", "pear", "balloon", "tomato", "clock", "canyon", "shower", "velvet"]))

    #     dset = DollarstreetDataset(attr_col='income_level', max_allowable_sim_of_classnames=0.9)
    # # # compare_ours_to_base(dset, vlm, ["street view", "stove/hob", "roof"])
    #     jobs.append(executor.submit(compare_ours_to_base, dset, vlm, ["street view", "stove/hob", "roof"]))

    # for dset in [Breeds(), MITStates(max_allowable_sim_of_classnames=0.8), DollarstreetDataset('income_level')]:
    # for dset in [Breeds(x) for x in ['entity13', 'entity30', 'nonliving26']]:
    # for dset in [MITStates(max_allowable_sim_of_classnames=0.8)]:#, , DollarstreetDataset('income_level')]:
    # for dset in [DollarstreetDataset('income_level')]:
        # ap_for_classname_vs_subpop(dset, vlm)
        # compare_ours_to_base(dset, vlm)#, classes_to_focus_on=['starting stove', 'bedroom',  'living room', 'street view', 'kitchen sink', 'radio', 'computer', 'tools', 'toilet', 'bathroom/toilet'])
        # save_many_bias_examples(dset)
        # view_many_ap_gain_egs(dset)
            # job = executor.submit(ap_for_classname_vs_subpop, dset, vlm)
            # jobs.append(job)
    
    plot_ap_gain_hists()

    # dset = DollarstreetDataset('income_level')#Breeds()#MITStates(max_allowable_sim_of_classnames=0.8)
    # save_many_bias_examples(dset)
    # visualize_bias()
    # visualize_bias(dset, 'balloon', 'deflated')

    # dset = Breeds()

    # compare_to_linear_probe()
    # plot_comparison_to_linear_probe()

    # plot_diversity_scattters()
    # for thresh in [0.7, 0.8, 0.85, 0.9]: 
    # for thresh in [0.5]:
        # record_problem_classes('dollarstreet', thresh=thresh)