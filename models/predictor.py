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
    ) -> Dict[str, Tensor]:
        text_embeddings_by_cls = dict({
            classname: self.convert_embeddings_for_one_class(embeddings_by_subpop)
                for classname, embeddings_by_subpop in embeddings_by_subpop_by_cls.items()
        })
        return text_embeddings_by_cls

class AverageAndNormThenStack(VLMPromptDimHandler):
    ### In CLIP, after averaging over template prompts, they normalize back to hypersphere. We do that here.
    def convert_embeddings_for_one_class(
        self,
        embeddings_by_subpop: Dict[str, Tensor]
    ) -> Tensor:
        embeddings_for_cls = []
        for subpop, embeddings_for_subpop in embeddings_by_subpop.items():
            # here we average over the VLM templates, getting one embedding per subpop
            subpop_avg = embeddings_for_subpop.mean(0)
            # now we normalize the avg vec so that we're back on the hypersphere
            subpop_avg /= subpop_avg.norm()
            embeddings_for_cls.append(subpop_avg)
        return torch.stack(embeddings_for_cls, axis=0)

class AverageAndStack(VLMPromptDimHandler):
    ### Same as AverageAndNormThenStack, except without the normalization
    def convert_embeddings_for_one_class(
        self,
        embeddings_by_subpop: Dict[str, Tensor]
    ) -> Tensor:
        # tens here is the embeddings_tensor for 1 subpop that has been templated over many VLM prompts
        embeddings_for_cls = [embeddings_for_subpop.mean(0) 
            for _, embeddings_for_subpop in embeddings_by_subpop.items()]
        return torch.stack(embeddings_for_cls, axis=0)

class StackAll(VLMPromptDimHandler):
    ### Here we stack everything, and let our predictor's handle the rest
    def convert_embeddings_for_one_class(
        self,
        embeddings_by_subpop: Dict[str, Tensor]
    ) -> Tensor:
        embeddings_for_cls = []
        for subpop, embeddings_for_subpop in embeddings_by_subpop.items():
            embeddings_for_cls.append(embeddings_for_subpop)
            # embeddings_for_cls = torch.vstack((embeddings_for_cls, embeddings_for_subpop))
        return torch.vstack(embeddings_for_cls)

### Our predictors, where we consolidate to final class logits
class Predictor(ABC):

    @abstractmethod
    def compute_logits(
        self,
        image_embeddings: Tensor,
        text_embeddings_by_cls: Dict[str, Tensor] 
    ) -> Tensor:
        raise NotImplementedError

    def predict(
        self, 
        image_embeddings: Tensor,
        # text_embeddings_by_cls: Dict[str, Tensor],
        text_embeddings_by_subpop_by_cls: Dict[str, Dict[str, Tensor]],
        classnames: List[str],
        vlm_dim_handling: VLMPromptDimHandler
    ) -> Tuple[List[str], Tensor]:
        # given image embeddings and dict of text embeddings for each class, 
        # return predicted classnames and confidences
        # logits = self.compute_logits(image_embeddings, text_embeddings_by_cls, classnames)
        text_embeddings_by_cls = vlm_dim_handling.convert_to_embeddings_by_cls(text_embeddings_by_subpop_by_cls)
        logits = self.compute_logits(image_embeddings, text_embeddings_by_cls, classnames)
        probs = torch.nn.functional.softmax(logits, dim=1)
        confidences, preds = probs.max(1)

        pred_classnames = [classnames[i.item()] for i in preds]
        return pred_classnames, confidences


class AverageVecs(Predictor):
    def compute_logits(
        self,
        image_embeddings: Tensor,
        text_embeddings_by_cls: Dict[str, Tensor],
        classnames: List[str]
    ) -> Tensor:

        logits = []
        for classname in classnames:
            embeddings_for_class = text_embeddings_by_cls[classname]
            avg_class_embedding = embeddings_for_class.mean(dim=0, keepdim=True)
            sims_to_avg_vec = cos_sim(image_embeddings, avg_class_embedding)
            logits.append(sims_to_avg_vec)
        
        return torch.stack(logits, dim=1)

    
class AverageSims(Predictor):
    def compute_logits(
        self,
        image_embeddings: Tensor,
        text_embeddings_by_cls: Dict[str, Tensor], 
        classnames: List[str]
    ) -> Tensor:

        logits = []
        for classname in classnames:
            embeddings_for_class = text_embeddings_by_cls[classname]
            sims = cos_sim(image_embeddings, embeddings_for_class)
            avg_sims = sims.mean(dim=1)
            logits.append(avg_sims)
        
        return torch.stack(logits, dim=1)

class MaxOfMax(Predictor):
    def compute_logits(
        self,
        image_embeddings: Tensor,
        text_embeddings_by_cls: Dict[str, Tensor], 
        classnames: List[str]
    ) -> Tensor: 

        logits = []
        for classname in classnames:
            embeddings_for_class = text_embeddings_by_cls[classname]
            sims = cos_sim(image_embeddings, embeddings_for_class)
            max_sims = sims.max(1).values
            logits.append(max_sims)
        
        return torch.stack(logits, dim=1)


class AverageTopKSims(Predictor):
    def __init__(self, k: int):
        self.k = k

    def compute_logits(
        self,
        image_embeddings: Tensor,
        text_embeddings_by_cls: Dict[str, Tensor], 
        classnames: List[str]
    ) -> Tensor:

        logits = []
        for classname in classnames:
            embeddings_for_class = text_embeddings_by_cls[classname]
            sims = cos_sim(image_embeddings, embeddings_for_class)
            top_k_sims = sims.topk(k=min(self.k, sims.shape[1]), dim=1).values
            avg_top_k_sims = top_k_sims.mean(1)
            logits.append(avg_top_k_sims)
        
        return torch.stack(logits, dim=1)


class LinearInterpolationAverageSimsTopK(Predictor):
    def __init__(self, k: int, lamb: float = 0.5):
        self.k = k
        self.lamb = lamb 
        
    def compute_logits(
        self,
        image_embeddings: Tensor,
        text_embeddings_by_cls: Dict[str, Tensor], 
        classnames: List[str]
    ) -> Tensor:

        logits = []
        for _, embeddings_for_class in text_embeddings_by_cls.items():
            sims = cos_sim(image_embeddings, embeddings_for_class)
            
            top_k_sims = sims.topk(k=min(self.k, embeddings_for_class.shape[0]), dim=1).values
            avg_top_k_sims = top_k_sims.mean(1)
            avg_sims = self.get_average_sims(sims, image_embeddings, embeddings_for_class)

            inter_sims = self.lamb * avg_sims + (1. - self.lamb) * avg_top_k_sims
            logits.append(inter_sims)
        
        return torch.stack(logits, dim=1)
    
    def get_average_sims(self,sims, image_embeddings, embeddings_for_class):

        return sims.mean(dim=1)

class LinearInterpolationAverageVecsTopk(LinearInterpolationAverageSimsTopK):
    def __init__(self, k: int, lamb: float = 0.5):
        self.k = k
        self.lamb = lamb 

    def get_average_sims(self,sims, image_embeddings, embeddings_for_class):

        avg_class_embedding = embeddings_for_class.mean(dim=0)
        sims_to_avg_vec = cos_sim(image_embeddings, avg_class_embedding)
        return sims_to_avg_vec

def init_predictor(key: str, lamb: float) -> Predictor:
    if key == 'max_of_max':
        predictor = MaxOfMax()
    elif key == 'average_vecs':
        predictor = AverageVecs()
    elif key == 'average_sims':
        predictor = AverageSims()
    elif 'average_top_' in key:
        k = int(key.split('_')[-1])
        predictor = AverageTopKSims(k=k)
    elif 'interpol_vecs_top_' in key:
        k = int(key.split('_')[-1])
        predictor = LinearInterpolationAverageVecsTopk(k=k, lamb=lamb)
    elif 'interpol_sims_top_' in key:
        k = int(key.split('_')[-1])
        predictor = LinearInterpolationAverageSimsTopK(k=k, lamb=lamb)
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