from abc import ABC, abstractmethod
from torch import Tensor
from typing import Dict, List
from my_utils import cache_data, load_cached_data
from constants import _CACHED_DATA_ROOT, _IMAGENET_OPENAI_TEMPLATES
import clip
import os
import torch
from tqdm import tqdm
import numpy as np

class VLM(ABC):
    def __init__(self, model_key: str, batch_size: int):
        raise NotImplementedError

    @abstractmethod
    def encode_image_batch(self, imgs: Tensor) -> Tensor:
        # should return tensor of size Nxd where d is VLM space dim and N is imgs.shape[0]
        raise NotImplementedError

    @abstractmethod
    def encode_texts(self, texts: List[str], vlm_prompt_templates: List[str]) -> Tensor:
        # should return tensor of size Nxd where d is VLM space dim and N is len(texts)
        raise NotImplementedError

    @abstractmethod
    def get_modelname(self) -> str:
        # should return a string name for the VLM, used in caching
        # Convention we'll use '{gen vlm type}__{specific instance}' (e.g. 'clip__ViT-B_16')
        # also we replace any slashes w/ '_' to avoid issues in caching
        raise NotImplementedError

    @abstractmethod
    def get_batchsize(self) -> int:
        # return an appropriate batchsize for encoding images
        raise NotImplementedError

    @abstractmethod
    def get_image_transform(self):
        # Should return a torchvision.transforms object that takes PIL images and returns 
        # tensors as desired by vlm.image_encoder
        raise NotImplementedError

    def embed_all_images(self, dset) -> Tensor:
        # cache_path = os.path.join(_CACHED_DATA_ROOT, 
        #DEBUG 
        cache_path = os.path.join("/checkpoint/dianeb/mmmd_results", 
                                    'image_embeddings', 
                                    self.get_modelname(), 
                                    f'{dset.get_dsetname()}.pkl')
        if os.path.exists(cache_path):
            dat = load_cached_data(cache_path)
            image_embeddings, identifier_idx = [dat[x] for x in ['image_embeddings', 'identifier_idx']]
        else:
            dset.transform = self.get_image_transform()
            loader = torch.utils.data.DataLoader(dset, batch_size=self.batch_size)
            image_embeddings, identifier_idx = [], []
            for dat in tqdm(loader):
                with torch.no_grad():
                    imgs = dat[0]
                    identifier_idx.extend(dat[1])
                    batch_embeddings = self.encode_image_batch(imgs)#.flatten(1)
                    image_embeddings.extend(batch_embeddings)#.detach().cpu().numpy())
            image_embeddings = torch.vstack(image_embeddings)
            identifier_idx = np.array(identifier_idx)

            data_to_cache = dict({'image_embeddings': image_embeddings, 'identifier_idx':identifier_idx})
            cache_data(cache_path, data_to_cache)
        
        return image_embeddings, identifier_idx

    def embed_subpopulation_descriptions(
        self, 
        subpops_by_class: Dict[str, List[str]], 
        vlm_prompt_templates: List[str]
    ) -> Dict[str, Tensor]:
        # each class will have a Tensor of embeddings corresponding to the different subpopulation descriptions
        # provided for the class. in the vanilla case, there will only be one description per class in subpops_by_class
        # (i.e. just the classname), and so embeddings_by_cls.values() will consist of tensors of shape (1, d), where d is VLM dim 

        if vlm_prompt_templates == ['USE OPENAI IMAGENET TEMPLATES']:
            vlm_prompt_templates = _IMAGENET_OPENAI_TEMPLATES

        embeddings_by_cls = dict({
            classname: self.encode_texts(subpop_descriptions, vlm_prompt_templates)
                for classname, subpop_descriptions in subpops_by_class.items()
        })

        return embeddings_by_cls


class CLIP(VLM):
    def __init__(self, model_key: str, batch_size: int = 64):
        self.model, self.transform = clip.load(model_key)
        self.model = self.model.cuda()
        self.model_key = model_key
        self.batch_size = batch_size

    def encode_image_batch(self, imgs: Tensor) -> Tensor:
        with torch.no_grad():
            image_embeddings = self.model.encode_image(imgs.cuda())
            image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
        return image_embeddings

    def encode_texts(self, texts: List[str], vlm_prompt_templates: List[str]) -> Tensor:
        with torch.no_grad():
            text_embeddings = []
            for text in texts:
                templated_text = [vlm_prompt.format(text) for vlm_prompt in vlm_prompt_templates]
                tokens = clip.tokenize(templated_text).cuda()     # tokenize
                embedded = self.model.encode_text(tokens)         # embed with text encoder
                embedded /= embedded.norm(dim=-1, keepdim=True)   # normalize to hypersphere
                embedded = embedded.mean(dim=0)                   # average over vlm_prompts
                embedded /= embedded.norm()                       # normalize again to hypersphere
                text_embeddings.append(embedded)
            text_embeddings = torch.stack(text_embeddings, dim=0).cuda()
        return text_embeddings

    def get_modelname(self) -> str:
        return 'clip__'+self.model_key.replace('/','_')

    def get_batchsize(self) -> int:
        return self.batch_size

    def get_image_transform(self):
        return self.transform