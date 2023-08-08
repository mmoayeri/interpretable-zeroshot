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

class Predictor(ABC):

    def __init__(self):
        pass

    # def get_predictor_name(self) -> str:
    #     return self.predictor_name

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
        text_embeddings_by_cls: Dict[str, Tensor],
        classnames: List[str]
    ) -> Tuple[List[str], Tensor]:
        # given image embeddings and dict of text embeddings for each class, 
        # return predicted classnames and confidences
        logits = self.compute_logits(image_embeddings, text_embeddings_by_cls, classnames)
        probs = torch.nn.functional.softmax(logits, dim=1)
        confidences, preds = probs.max(1)

        pred_classnames = [classnames[i.item()] for i in preds]
        return pred_classnames, confidences


class AverageVecs(Predictor):
    # def __init__(self, predicto)

    def compute_logits(
        self,
        image_embeddings: Tensor,
        text_embeddings_by_cls: Dict[str, Tensor],
        classnames: List[str]
    ) -> Tensor:

        logits = []
        for classname in classnames:
            embeddings_for_class = text_embeddings_by_cls[classname]
            avg_class_embedding = embeddings_for_class.mean(dim=0)
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
            
            top_k_sims = sims.topk(k=self.k, dim=1).values
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