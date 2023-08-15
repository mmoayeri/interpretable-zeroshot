from abc import ABC, abstractmethod
from torch import Tensor
from typing import Dict, List, Tuple
from my_utils import cache_data, load_cached_data
from constants import _CACHED_DATA_ROOT, _IMAGENET_OPENAI_TEMPLATES, _CONDENSED_OPENAI_TEMPLATES
import clip
import os
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess


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

    def embed_all_images(self, dset) -> Tuple[Tensor, List[str]]:
        cache_path = os.path.join(
            _CACHED_DATA_ROOT,
            "image_embeddings",
            self.get_modelname(),
            f"{dset.get_dsetname()}.pkl",
        )
        if os.path.exists(cache_path):
            dat = load_cached_data(cache_path)
            image_embeddings, identifiers = [
                dat[x] for x in ["image_embeddings", "identifiers"]
            ]
        else:
            dset.transform = self.get_image_transform()
            loader = torch.utils.data.DataLoader(dset, batch_size=self.batch_size)
            image_embeddings, identifiers = [], []
            for dat in tqdm(loader):
                with torch.no_grad():
                    imgs = dat[0]
                    identifiers.extend(dat[1])
                    batch_embeddings = self.encode_image_batch(imgs)  # .flatten(1)
                    image_embeddings.extend(
                        batch_embeddings
                    )  # .detach().cpu().numpy())
            image_embeddings = torch.vstack(image_embeddings)

            data_to_cache = dict(
                {"image_embeddings": image_embeddings, "identifiers": identifiers}
            )
            cache_data(cache_path, data_to_cache)

        return image_embeddings, identifiers

    def embed_subpopulation_descriptions(
        self, subpops_by_class: Dict[str, List[str]], vlm_prompt_templates: List[str]
    ) -> Dict[str, Tensor]:
        # each class will have a Tensor of embeddings corresponding to the different subpopulation descriptions
        # provided for the class. in the vanilla case, there will only be one description per class in subpops_by_class
        # (i.e. just the classname), and so embeddings_by_cls.values() will consist of tensors of shape (1, d), where d is VLM dim

        if vlm_prompt_templates == ["USE OPENAI IMAGENET TEMPLATES"]:
            vlm_prompt_templates = _IMAGENET_OPENAI_TEMPLATES
        elif vlm_prompt_templates == ['USE CONDENSED OPENAI TEMPLATES']:
            vlm_prompt_templates = _CONDENSED_OPENAI_TEMPLATES

        embeddings_by_cls = dict(
            {
                classname: self.encode_texts(subpop_descriptions, vlm_prompt_templates)
                for classname, subpop_descriptions in subpops_by_class.items()
            }
        )

        return embeddings_by_cls


class CLIP(VLM):
    def __init__(self, model_key: str, batch_size: int = 64):
        self.model, self.transform = clip.load(model_key)
        self.model = self.model.cuda().eval()
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
                templated_text = [
                    vlm_prompt.format(text) for vlm_prompt in vlm_prompt_templates
                ]
                tokens = clip.tokenize(templated_text).cuda()  # tokenize
                embedded = self.model.encode_text(tokens)  # embed with text encoder
                embedded /= embedded.norm(
                    dim=-1, keepdim=True
                )  # normalize to hypersphere
                embedded = embedded.mean(dim=0)  # average over vlm_prompts
                embedded /= embedded.norm()  # normalize again to hypersphere
                text_embeddings.append(embedded)
            text_embeddings = torch.stack(text_embeddings, dim=0)
        return text_embeddings

    def get_modelname(self) -> str:
        return "clip__" + self.model_key.replace("/", "_")

    def get_batchsize(self) -> int:
        return self.batch_size

    def get_image_transform(self):
        return self.transform


class BLIP2(VLM):
    """Implements a VLM  for BLIP2 model
    based  on https://github.com/salesforce/LAVIS/blob/3446bac20c5646d35ae383ebe6d13cec4f8b00cb/examples/blip2_feature_extraction.ipynb
    """

    def __init__(self, frozen_text_encoder: str = "default", device: str = "cuda"):
        self.frozen_text_encoder = frozen_text_encoder
        if frozen_text_encoder != "default":
            raise ValueError(f"{frozen_text_encoder} not supported")
        self.device = device

        (
            self.model,
            self.vis_processors,
            self.txt_processors,
        ) = load_model_and_preprocess(
            name="blip2_feature_extractor",
            model_type="pretrain",
            is_eval=True,
            device=device,
        )

    def encode_image_batch(self, imgs: Tensor) -> Tensor:
        with torch.no_grad():
            image_embeddings = self.model.extract_features(
                {"image": imgs}, mode="image"
            ).image_embeds
        return image_embeddings

    def build_text_inputs(
        self, texts: List[str], vlm_prompt_templates: List[str]
    ) -> List[str]:
        text_inputs = []
        for vlm_prompt in vlm_prompt_templates:
            text_inputs.extend([vlm_prompt.format(text) for text in texts])
        return text_inputs

    def encode_texts(self, texts: List[str], vlm_prompt_templates: List[str]) -> Tensor:
        text_inputs = self.build_text_inputs(texts, vlm_prompt_templates)
        with torch.no_grad():
            text_embeddings = None
        return text_embeddings


class InstructBLIP(VLM):
    """Implements a VLM class for InstructBLIP, the latest iteration on BLIP2.

    Requires installing LAVIS locally and downloading VICUNA weights
        see https://github.com/salesforce/LAVIS/tree/main/projects/instructblip

    Implements an extra
        - `generate_image_conditioned_text` method

    Args:
        frozen_text_encoder: vicuna-7b or vicuna-13b
        device: cuda or cpu (for running locally when GPU memory is insufficient)
    """

    def __init__(self, frozen_text_encoder: str = "vicuna-7b", device: str = "cuda"):
        self.frozen_text_encoder = frozen_text_encoder
        if frozen_text_encoder != "vicuna-7b":
            raise ValueError(f"{frozen_text_encoder} not supported")
        self.device = device

        (
            self.model,
            self.vis_processors,
            self.text_processors,
        ) = load_model_and_preprocess(
            name="blip2_vicuna_instruct",
            model_type=frozen_text_encoder,
            is_eval=True,
            device=device,
        )

    def generate_image_conditioned_text(
        self,
        image: Tensor,
        prompt="Can you tell me about this image in detail?",
    ) -> str:
        """Returns generated text based on image.

        Args:
            image: tensor of shape (batch size, 3, 224, 224)
                image is expected to be processed using vis_processors
        """
        generated_text = self.model.generate({"image": image, "prompt": prompt})
        return generated_text

    def encode_image_batch(self, imgs: Tensor) -> Tensor:
        raise NotImplemented

    def encode_texts(self, texts: List[str], vlm_prompt_templates: List[str]) -> Tensor:
        raise NotImplemented

    def get_modelname(self) -> str:
        return "clip__" + self.model_key.replace("/", "_")

    def get_batchsize(self) -> int:
        return self.batch_size

    def get_image_transform(self):
        return self.transform
