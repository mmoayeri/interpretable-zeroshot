from abc import ABC, abstractmethod
from typing import Dict, Tuple
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
        text_embeddings_by_cls: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        # given image embeddings and dict of text embeddings for each class, 
        # return predictions and confidences
        # note that the predictions
        logits = self.compute_logits(image_embeddings, text_embeddings_by_cls)
        preds = logits.argmax(1)
        probs = torch.nn.functional.softmax(logits, dim=1)
        confidences, preds = probs.max(1)
        return preds, confidences 

        # classnames = list(text_embeddings_by_cls.keys())
        # pred_classnames = [classnames[i.item()] for i in preds]
        # return pred_classnames, confidences 


class AverageVecs(Predictor):
    # def __init__(self, predicto)

    def compute_logits(
        self,
        image_embeddings: Tensor,
        text_embeddings_by_cls: Dict[str, Tensor]
    ) -> Tensor:

        logits = []
        for i, (classname, embeddings_for_class) in enumerate(text_embeddings_by_cls.items()):
            print(classname)
            avg_class_embedding = embeddings_for_class.mean(dim=0)
            sims_to_avg_vec = cos_sim(image_embeddings, avg_class_embedding)
            logits.append(sims_to_avg_vec)
        
        return torch.stack(logits, dim=1)

    
class AverageSims(Predictor):
    def compute_logits(
        self,
        image_embeddings: Tensor,
        text_embeddings_by_cls: Dict[str, Tensor] 
    ) -> Tensor:

        logits = []
        for classname, embeddings_for_class in text_embeddings_by_cls.items():
            sims = cos_sim(image_embeddings, embeddings_for_class)
            avg_sims = sims.mean(dim=1)
            logits.append(avg_sims)
        
        return torch.stack(logits, dim=1)

class MaxOfMax(Predictor):
    def compute_logits(
        self,
        image_embeddings: Tensor,
        text_embeddings_by_cls: Dict[str, Tensor]
    ) -> Tensor: 

        logits = []
        for classname, embeddings_for_class in text_embeddings_by_cls.items():
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
        text_embeddings_by_cls: Dict[str, Tensor]
    ) -> Tensor:

        logits = []
        for classname, embeddings_for_class in text_embeddings_by_cls.items():
            sims = cos_sim(image_embeddings, embeddings_for_class)
            top_k_sims = sims.topk(k=self.k, dim=1).values
            avg_top_k_sims = top_k_sims.mean(1)
            logits.append(avg_top_k_sims)
        
        return torch.stack(logits, dim=1)


class LinearInterpolationAverageTopKSims(Predictor):
    def __init__(self, k: int, lamb: float = 0.5):
        self.k = k
        self.lamb = lamb 

    def compute_logits(
        self,
        image_embeddings: Tensor,
        text_embeddings_by_cls: Dict[str, Tensor]
    ) -> Tensor:

        logits = []
        for _, embeddings_for_class in text_embeddings_by_cls.items():
            sims = cos_sim(image_embeddings, embeddings_for_class)
            top_k_sims = sims.topk(k=self.k, dim=1).values
            avg_top_k_sims = top_k_sims.mean(1)
            avg_sims = sims.mean(dim=1)
            inter_sims = self.lamb * avg_sims + (1. - self.lamb) * avg_top_k_sims
            logits.append(inter_sims)
        
        return torch.stack(logits, dim=1)