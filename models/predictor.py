from abc import ABC, abstractmethod
from typing import Dict, Tuple, List
from torch import Tensor
import torch


def cos_sim(tens1: Tensor, tens2: Tensor) -> Tensor:
    """
    Given tens1 of shape N x d and tens2 of shape M x d, we return a 
    Tensor of shape N x M where the (i,j)^th cell is the similarity of
    the i^th row in tens1 to the j^th col of tens2. Usually, tens1 
    contains image embeddings, and tens2 has text embeddings
    """
    tens1 = tens1 / tens1.norm(dim=-1, keepdim=True)
    tens2 = tens2 / tens2.norm(dim=-1, keepdim=True)
    return tens1 @ tens2.T


'''
Predictor code was previously written to expect text_embeddings_by_cls as Dict[str, Tensor].
We now have a 3D structure of (classname, Dict[subpop, Tensor of embedding of subpops templated in vlm prompts]])

I'll provide a couple functions to get us back to Dict[str, Tensor], for ease of compatibility with old code.
'''

class VLMPromptDimHandler(ABC):
    @abstractmethod
    def convert_embeddings_for_one_class(
        embeddings_by_subpop: Dict[str, Tensor]
    ) -> Tensor:
        raise NotImplementedError

    def convert_to_embeddings_by_cls(
        self, 
        embeddings_by_subpop_by_cls: Dict[str, Dict[str, Tensor]]
    ) -> Tuple[Dict[str, Tensor], Dict[str, List[str]]]:
        # text_embeddings_by_cls = dict({
        #     classname: self.convert_embeddings_for_one_class(embeddings_by_subpop)
        #         for classname, embeddings_by_subpop in embeddings_by_subpop_by_cls.items()
        # })
        text_embeddings_by_cls, subpop_captions_by_cls = dict(), dict()
        for classname, embeddings_by_subpop in embeddings_by_subpop_by_cls.items():
            embeddings_for_cls, subpop_captions = self.convert_embeddings_for_one_class(embeddings_by_subpop)
            text_embeddings_by_cls[classname] = embeddings_for_cls.cuda()
            subpop_captions_by_cls[classname] = subpop_captions

        return text_embeddings_by_cls, subpop_captions_by_cls

class AverageAndNormThenStack(VLMPromptDimHandler):
    ### In CLIP, after averaging over template prompts, they normalize back to hypersphere. We do that here.
    def convert_embeddings_for_one_class(
        self,
        embeddings_by_subpop: Dict[str, Tensor]
    ) -> Tensor:
        embeddings_for_cls = []
        subpop_captions = []
        for subpop, embeddings_for_subpop in embeddings_by_subpop.items():
            # here we average over the VLM templates, getting one embedding per subpop
            subpop_avg = embeddings_for_subpop.mean(0)
            # now we normalize the avg vec so that we're back on the hypersphere
            subpop_avg /= subpop_avg.norm()
            embeddings_for_cls.append(subpop_avg)
            subpop_captions.append(subpop)
        return torch.stack(embeddings_for_cls, axis=0), subpop_captions

class AverageAndStack(VLMPromptDimHandler):
    ### Same as AverageAndNormThenStack, except without the normalization
    def convert_embeddings_for_one_class(
        self,
        embeddings_by_subpop: Dict[str, Tensor]
    ) -> Tensor:
        # tens here is the embeddings_tensor for 1 subpop that has been templated over many VLM prompts
        embeddings_for_cls, subpop_captions = [], []
        for subpop, embeddings_for_subpop in embeddings_by_subpop.items():
            embeddings_for_cls.append(embeddings_for_subpop.mean(0))
            subpop_captions.append(subpop)
        return torch.stack(embeddings_for_cls, axis=0), subpop_captions

class StackAll(VLMPromptDimHandler):
    ### Here we stack everything, and let our predictor's handle the rest
    def convert_embeddings_for_one_class(
        self,
        embeddings_by_subpop: Dict[str, Tensor]
    ) -> Tensor:
        embeddings_for_cls, subpop_captions = [], []
        for subpop, embeddings_for_subpop in embeddings_by_subpop.items():
            embeddings_for_cls.append(embeddings_for_subpop)
            subpop_captions.append(subpop)
        return torch.vstack(embeddings_for_cls), subpop_captions

### Now we try reweighting attributes based on sim to their own class and/or other classes

class AttrReweighter:
    def compute_weights(
        self,
        mode: str,
        text_embeddings_by_cls: Dict[str, Tensor],
        text_embeddings_by_subpop_by_cls: Dict[str, Dict[str, Tensor]],
        classnames: List[str],
    ) -> Dict[str, Tensor]:
        classname_only_embeddings_dict = dict({
            classname: text_embeddings_by_subpop_by_cls[classname][classname].mean(0, keepdim=True)
                for classname in classnames
        })
        classname_only_vecs = torch.vstack([classname_only_embeddings_dict[c] for c in classnames])
        weights_by_cls = dict()
        for cls_ind, classname in enumerate(classnames):
            # Similarity of each subpop embedding for the class to the classname only embeddings
            sims = cos_sim(text_embeddings_by_cls[classname], classname_only_vecs)
            # Separately compute (i) sim of each subpop to its own class (ii) mean of sims to other classes
            sims_to_class = sims[:, cls_ind]
            mean_sims_to_other_classes = torch.hstack([sims[:, :cls_ind], sims[:, cls_ind+1:]]).mean(1)
            if mode == 'sim_to_class':
                weights_by_cls[classname] = sims_to_class
            elif mode == 'sim_to_other_class':
                # here we want to downweight attrs that are very similar to other classes
                weights_by_cls[classname] = 1 / mean_sims_to_other_classes
            elif mode == 'ratio':
                weights_by_cls[classname] = sims_to_class / mean_sims_to_other_classes
            else:
                raise ValueError(f"Mode {mode} not recognized. Must be one of 'sim_to_class', 'sim_to_other_class', or 'ratio'")
        return weights_by_cls



### Our predictors, where we consolidate to final class logits
class Predictor(ABC):
    @abstractmethod
    def compute_logits(
        self,
        image_embeddings: Tensor,
        text_embeddings_by_cls: Dict[str, Tensor],
        weights_by_cls: Dict[str, Tensor],
        classnames: List[str] 
    ) -> Tensor:
        raise NotImplementedError

    def predict(
        self, 
        image_embeddings: Tensor,
        text_embeddings_by_subpop_by_cls: Dict[str, Dict[str, Tensor]],
        classnames: List[str],
        vlm_dim_handling: VLMPromptDimHandler,
        weights_by_cls: Dict[str, Tensor] = None,
        batch_size:int = 512,
    ) -> Tuple[List[str], Tensor]:
        # given image embeddings and dict of text embeddings by subpop for each class, 
        # return predicted classnames and confidences
        text_embeddings_by_cls, _ = vlm_dim_handling.convert_to_embeddings_by_cls(text_embeddings_by_subpop_by_cls)

        pred_classnames, confidences = [], []
        for i in range(0, image_embeddings.shape[0], batch_size):
            batch_img_embeddings = image_embeddings[i:i+batch_size].cuda()
            logits = self.compute_logits(batch_img_embeddings, text_embeddings_by_cls, classnames, weights_by_cls)
            probs = torch.nn.functional.softmax(logits, dim=1)
            batch_confidences, preds = probs.max(1)
            batch_pred_classnames = [classnames[i.item()] for i in preds]

            pred_classnames.extend(batch_pred_classnames)
            confidences.append(batch_confidences)
        
        return pred_classnames, confidences


class CHiLS(Predictor):
    def __init__(self, k: int = 1, mode : str = 'sims'):
        self.k = k
        self.mode = mode
        # We use an auxiliary predictor to 1) get superclass (i.e. classname only) sims 2) get top subpop sims
        # Note for superclass sim, choice of k does not matter
        self.aux_predictor = AverageTopK(k = self.k, mode=self.mode)

    # CHiLS has this superclass reweighting step, so we'll implement its predict function entirely.
    def predict(
        self, 
        image_embeddings: Tensor,
        text_embeddings_by_subpop_by_cls: Dict[str, Dict[str, Tensor]],
        classnames: List[str],
        vlm_dim_handling: VLMPromptDimHandler,
        weights_by_cls: Dict[str, Tensor] = None
    ) -> Tuple[List[str], Tensor]:
        # First let's take out the classname only subpops
        # Note that the classname by itself needs to be among the subpops (i.e. must also use vanilla attributer)
        assert sum([(classname not in text_embeddings_by_subpop_by_cls[classname]) for classname in classnames]) == 0, \
            "Missing classname alone as subpop for each class. You must include 'vanilla' as attributer when using CHiLS." 
        classname_only_embeddings_dict = dict({
            classname: text_embeddings_by_subpop_by_cls[classname][classname].mean(0, keepdim=True).cuda()
                for classname in classnames
        })
        superclass_sims = self.aux_predictor.compute_logits(image_embeddings, classname_only_embeddings_dict, classnames, weights_by_cls)
        text_embeddings_by_cls, _ = vlm_dim_handling.convert_to_embeddings_by_cls(text_embeddings_by_subpop_by_cls)
        # Notice we use superclass *probabilities* (i.e. apply softmax)
        superclass_probs = torch.nn.functional.softmax(superclass_sims, dim=1)
        # Now we compute probabilities for each subpopulations; but we only care about the max one per class
        end_of_class = [0]
        all_vecs = []
        for i, c in enumerate(classnames):
            vecs_for_cls = text_embeddings_by_cls[c]
            all_vecs.append(vecs_for_cls)
            end_of_class.append(end_of_class[-1] + vecs_for_cls.shape[0])
        all_vecs = torch.vstack(all_vecs)
        subclass_sims = cos_sim(image_embeddings, all_vecs)
        subclass_probs = torch.nn.functional.softmax(subclass_sims, dim=1)

        max_prob_per_class = torch.hstack([
            subclass_probs[:, end_of_class[i]:end_of_class[i+1]].max(1, keepdim=True).values
                for i, _ in enumerate(classnames)
        ])

        # Reweight with superclass (i.e. classname only) probability
        reweighted_sims_by_class = torch.stack([
            max_prob_per_class[:, i] * superclass_probs[:, i] 
                for i, classname in enumerate(classnames)
        ], axis=1)
        confidences, preds = reweighted_sims_by_class.max(1)

        pred_classnames = [classnames[i.item()] for i in preds]
        return pred_classnames, confidences
    
    def compute_logits(        
        self,
        image_embeddings: Tensor,
        text_embeddings_by_cls: Dict[str, Tensor],
        classnames: List[str] 
    ) -> Tensor:
        raise Exception("For CHiLS, everything should go directly through the predict function, since we reweight by superclass probabilities." +
                        "In other words, predictor.compute_logits should never be called when predictor is CHiLS. There's a bug; find it.")

class AverageTopK(Predictor):
    def __init__(self, k: int=8, mode: str = 'sims', lamb: float = 0, top: bool=True):
        self.k = k
        assert mode in ['sims', 'vecs'], f"Mode {mode} not recognized. Must be 'sims' or 'vecs'"
        self.mode = mode
        self.lamb = lamb
        # in this case, we look at the BOTTOM k (crazy right!? but curious! let's see what happens)
        self.top = top

    def compute_logits(
        self,
        image_embeddings: Tensor,
        text_embeddings_by_cls: Dict[str, Tensor], 
        classnames: List[str],
        weights_by_cls: Dict[str, Tensor]=None
    ) -> Tensor:

        logits = []
        for classname in classnames:
            embeddings_for_class = text_embeddings_by_cls[classname]
            sims = cos_sim(image_embeddings, embeddings_for_class)
            if weights_by_cls is not None:
                sims *= weights_by_cls[classname]

            top_k_sims, top_k_inds = sims.topk(k=min(self.k, sims.shape[1]), dim=1, largest=self.top)
            if self.mode == 'sims':    # take average of the top k sims
                avg_top_k = top_k_sims.mean(1)
                avg_all = sims.mean(1)
            elif self.mode == 'vecs':  # or average the top k vecs and take sim to that avg
                # for each sample, we want to take the most similar vectors, and average them
                # the score for each sample is the cos_sim of its embedding to the avg of the k text embeddings 
                # within the class that they are most similar to
                top_k_vecs_per_sample = embeddings_for_class[top_k_inds]   # has shape (Num_samples, k, d)
                avg_top_k_vecs_per_sample = top_k_vecs_per_sample.mean(1)  # has shape (Num_samples, d)
                # the diag is bc we only want the sim of img vec n with the avg vecs for that sample, which is in position n
                avg_top_k = torch.diag(cos_sim(image_embeddings, avg_top_k_vecs_per_sample))
                avg_vec_all = embeddings_for_class.mean(dim=0, keepdim=True)
                avg_all = cos_sim(image_embeddings, avg_vec_all).squeeze()

            logit = self.lamb * avg_all + (1 - self.lamb) * avg_top_k
            logits.append(logit)
        
        return torch.stack(logits, dim=1)


class KNN(Predictor):
    def __init__(self, k: int, mode: str):
        self.k = k
        assert mode in ['voting', 'sims', 'avgsims'], f"Mode {mode} not recognized for kNN predictor. Must be one of 'voting', 'sims', 'avgsims'."
        self.mode = mode

    def compute_logits(
        self, 
        image_embeddings: Tensor,
        text_embeddings_by_cls: Dict[str, Tensor], 
        classnames: List[str],
        weights_by_cls: Dict[str, Tensor]
    ) -> Tensor:
        
        all_embeddings = []
        vec_ind_to_cls_ind = []
        for i, classname in enumerate(classnames):
            cls_vecs = text_embeddings_by_cls[classname]
            all_embeddings.append(cls_vecs)
            vec_ind_to_cls_ind.extend([i]*cls_vecs.shape[0])

        all_embeddings = torch.vstack(all_embeddings)                   # shape (C*M, d), where C is # classes
        vec_ind_to_cls_ind = torch.tensor(vec_ind_to_cls_ind).cuda()    # shape (C*M, )
        sims = cos_sim(image_embeddings, all_embeddings)                # shape (N, C*M)
        num_classes = len(classnames)
        top_k_sims, top_k_inds = sims.topk(k=min(self.k * num_classes, sims.shape[1]), dim=1)
        # map top_k inds to class inds: creates (N, k) vec of ints in [0, C-1]
        selected_cls_inds_per_sample = vec_ind_to_cls_ind[top_k_inds]
        logits = []
        for i, classname in enumerate(classnames):
            # binary mask indicating if a val in top_k_sims corresponds to current class 
            vecs_selected_for_cls = (selected_cls_inds_per_sample == i)
            if self.mode == 'voting':
                # every one in the topk gets equal weight
                logits.append(vecs_selected_for_cls.sum(1))
            elif self.mode == 'sims':
                logits.append((top_k_sims * vecs_selected_for_cls).sum(1))
            # elif self.mode == 'avgsims'        
        return torch.stack(logits, dim=1).float()

class AdaptiveKNN(Predictor):
    def __init__(self, z_thresh: float, mode: str, lamb: float):
        self.z_thresh = z_thresh
        assert mode in ['voting', 'sims', 'avgsims'], f"Mode {mode} not recognized for kNN predictor. Must be one of 'voting', 'sims', 'avgsims'."
        self.mode = mode
        self.lamb = lamb

    def compute_logits(
        self, 
        image_embeddings: Tensor,
        text_embeddings_by_cls: Dict[str, Tensor], 
        classnames: List[str],
        weights_by_cls = None
    ) -> Tensor:
        
        sims = []
        avg_all_logits = []
        vec_ind_to_cls_ind = []
        for i, classname in enumerate(classnames):
            cls_vecs = text_embeddings_by_cls[classname]
            # all_embeddings.append(cls_vecs)
            sims_for_class = cos_sim(image_embeddings, cls_vecs)
            sims.append(sims_for_class)
            avg_all_logits.append(sims_for_class.mean(1, keepdim=True))
            vec_ind_to_cls_ind.extend([i]*cls_vecs.shape[0])

        map_to_cls_ind = torch.tensor([vec_ind_to_cls_ind]*image_embeddings.shape[0]).cuda()
        # stack and normalize sims (per sample); also normalize avg_sims_logits
        sims, avg_all_logits = [torch.hstack(x) for x in [sims, avg_all_logits]]
        mean, std = sims.mean(1, keepdim=True), sims.std(1, keepdim=True)
        # sims, avg_all_logits = [(x-mean)/std for x in [sims, avg_all_logits]]
        sims = (sims-mean)/std
        # remove sims that don't meet z_score threshold from consideration
        # let's be careful though in case z_thresh is too high. In this case we default to max of max
        thresh_tens = torch.min(torch.tensor(self.z_thresh), sims.max(1, keepdim=True).values)
        mask = sims < thresh_tens
        map_to_cls_ind[mask] = -1
        del mask
        del thresh_tens
        # let's unnormalize ... shouldn't really matter but wtv
        sims = mean + std * sims
        del std
        del mean
        # now compute logits
        logits = []
        for cls_ind, classname in enumerate(classnames):
            in_cls_meeting_thresh = (map_to_cls_ind == cls_ind)
            if self.mode == 'voting':
                logits.append(in_cls_meeting_thresh.sum(1))
            elif self.mode == 'sims':
                logits.append((sims * in_cls_meeting_thresh).sum(1))
            elif self.mode == 'avgsims':
                num_in_class_meeting_thresh = in_cls_meeting_thresh.sum(1)
                num_in_class_meeting_thresh[num_in_class_meeting_thresh==0] = 1 # to avoid nans
                logits.append((sims * in_cls_meeting_thresh).sum(1) / num_in_class_meeting_thresh)

        knn_logits = torch.stack(logits, dim=1).float()
        return self.lamb * avg_all_logits + (1 - self.lamb) * knn_logits

def init_predictor(key: str, lamb: float) -> Predictor:
    if key == 'average_vecs':
        predictor = AverageTopK(mode='vecs', lamb=1)
    elif key == 'average_sims':
        predictor = AverageTopK(mode='sims', lamb=1)
    elif key == 'max_of_max':
        predictor = AverageTopK(k=1, lamb=0)
    elif 'adaptive_knn' in key:
        _, _, mode, z_thresh = key.split('_')
        predictor = AdaptiveKNN(z_thresh=float(z_thresh), mode=mode, lamb=lamb)
    elif 'knn' in key:
        _, mode, k = key.split('_')
        predictor = KNN(k=int(k), mode=mode)
    elif 'chils' in key:
        kwargs = key.split('_')
        assert len(kwargs) in [1,2,3], "Wrong key for chils. Must be one of chils, chils_k, or chils_k_{sims or vecs}"
        # Notice default (key == 'chils') is just k=1 and mode is 'sims' (tho it makes no difference w. k =1)
        k = int(kwargs[1]) if len(kwargs) > 1 else 1
        mode = kwargs[2] if len(kwargs) == 3 else 'sims'
        predictor = CHiLS(k=k, mode=mode)
    elif key.split('_')[0] == 'average':
        # I'm expecting something like average_{top or bottom}_{k}_{vecs or sims}.
        # Regex would be better for elif but its ok; we have input protection here + in AverageTopK constructor
        _, top_or_bottom, k, mode = key.split('_')
        assert top_or_bottom in ['top', 'bottom'], f"Invalid option for top or bottom: {top_or_bottom}"
        top = (top_or_bottom == 'top')
        k = int(k)
        predictor = AverageTopK(k=k, mode=mode, lamb=lamb, top=top)
    else:
        raise ValueError(f'Predictor with key {key} not recognized. Is it implemented? Should be in ./models/predictor.py')
    return predictor


def init_vlm_prompt_dim_handler(key: str) -> VLMPromptDimHandler:
    if key == 'average_and_norm_then_stack':
        handler = AverageAndNormThenStack()
    elif key == 'average_and_stack':
        handler = AverageAndStack()
    elif key == 'stack_all':
        handler = StackAll()
    else:
        raise ValueError(f'VLMPromptDimHandler with key {key} not recognized.')
    return handler