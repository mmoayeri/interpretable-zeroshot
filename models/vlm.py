from abc import ABC, abstractmethod
from torch import Tensor
from typing import Dict, List, Tuple
from my_utils import cache_data, load_cached_data
from constants import (
    _CACHED_DATA_ROOT,
)
import clip
import os
import torch
from tqdm import tqdm
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

    def build_text_inputs(
        self, texts: List[str], vlm_prompt_templates: List[str]
    ) -> List[str]:
        """Combines texts and vlm_prompt_templates into list of string inputs"""
        text_inputs = []
        for vlm_prompt in vlm_prompt_templates:
            text_inputs.extend([vlm_prompt.format(text) for text in texts])
        return text_inputs

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
        self, texts_by_subpop_by_class: Dict[str, Dict[str, List[str]]]
    ) -> Dict[str, Dict[str, Tensor]]:
        embeddings_by_subpop_by_cls = dict(
            {
                classname: dict(
                    {
                        subpop_caption: self.encode_texts(
                            subpop_caption_in_vlm_templates
                        )
                        for subpop_caption, subpop_caption_in_vlm_templates in subpop_dict.items()
                    }
                )
                for classname, subpop_dict in texts_by_subpop_by_class.items()
            }
        )
        return embeddings_by_subpop_by_cls


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

    def encode_texts(self, texts: List[str]) -> Tensor:
        with torch.no_grad():
            text_embeddings = []
            tokens = clip.tokenize(texts).cuda()
            text_embeddings = self.model.encode_text(tokens)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings

    def get_modelname(self) -> str:
        return "clip__" + self.model_key.replace("/", "_")

    def get_batchsize(self) -> int:
        return self.batch_size

    def get_image_transform(self):
        return self.transform


class BLIP2(VLM):
    """
    Implements a VLM  for BLIP2 model
    based  on https://github.com/salesforce/LAVIS/blob/3446bac20c5646d35ae383ebe6d13cec4f8b00cb/examples/blip2_feature_extraction.ipynb
    """

    def __init__(
        self, frozen_text_encoder: str = "bert-base-uncased", device: str = "cuda"
    ):
        self.frozen_text_encoder = frozen_text_encoder
        if frozen_text_encoder != "bert-base-uncased":
            raise ValueError(f"{frozen_text_encoder} not supported")
        self.device = device

        (
            self.model,
            self.vis_processors,
            self.text_processors,
        ) = load_model_and_preprocess(
            name="blip2_feature_extractor",
            model_type="pretrain",
            is_eval=True,
            device=device,
        )

    def encode_image_batch(
        self, imgs: Tensor, project_embeddings: bool = False
    ) -> Tensor:
        """project_embeddings extracts embeddings post-projections which is often used in zero shot"""
        with torch.no_grad():
            if project_embeddings:
                image_embeddings = self.model.extract_features(
                    {"image": imgs}, mode="image"
                ).image_embeds_proj
            else:
                image_embeddings = self.model.extract_features(
                    {"image": imgs}, mode="image"
                ).image_embeds
        return image_embeddings

    def encode_texts(
        self,
        texts: List[str],
        vlm_prompt_templates: List[str],
        project_embeddings: bool = False,
    ) -> Tensor:
        """project_embeddings extracts embeddings post-projections which is often used in zero shot"""
        text_inputs = self.build_text_inputs(texts, vlm_prompt_templates)
        with torch.no_grad():
            processed_text_inputs = []
            for text_input in text_inputs:
                processed_text_input = self.text_processors["eval"](text_input)
                processed_text_inputs.append(processed_text_input)

            if project_embeddings:
                text_embeddings = self.model.extract_features(
                    {"text_input": processed_text_inputs}, mode="text"
                ).text_embeds_proj
            else:
                text_embeddings = self.model.extract_features(
                    {"text_input": processed_text_inputs}, mode="text"
                ).text_embeds
        return text_embeddings

    def get_image_transform(self):
        return self.vis_processors

    def get_modelname(self) -> str:
        return f"BLIP-2-{self.frozen_text_encoder}"

    def __repr__(self) -> str:
        return self.get_modelname()

    def get_batchsize(self) -> int:
        return 64


class InstructBLIP(VLM):
    """
    TODO: need to figure out feature extraction for this model.
    Implements a VLM class for InstructBLIP, the latest iteration on BLIP2.

    Requires installing LAVIS locally and downloading VICUNA weights
        see https://github.com/salesforce/LAVIS/tree/main/projects/instructblip

    Implements an extra
        - `generate_image_conditioned_text` method

    Args:
        frozen_text_encoder: vicuna7b or vicuna13b
        device: cuda or cpu (for running locally when GPU memory is insufficient)
    """

    def __init__(self, frozen_text_encoder: str = "vicuna13b", device: str = "cuda"):
        self.frozen_text_encoder = frozen_text_encoder
        if frozen_text_encoder not in {"vicuna7b", "vicuna13b"}:
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

    def encode_image_batch(self, images: Tensor) -> Tensor:
        """Based on feature_extraction from BLIP2.
        https://github.com/salesforce/LAVIS/blob/f982acc73288408bceda2d35471a8fcf55aa04ca/lavis/models/blip2_models/blip2_qformer.py#L387
        """
        image_embeds_frozen = self.model.ln_vision(self.model.visual_encoder(images))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(image_embeds_frozen.size()[:-1], dtype=torch.long).to(
            self.device
        )
        query_tokens = self.model.query_tokens.expand(
            image_embeds_frozen.shape[0], -1, -1
        )

        query_output = self.model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_embeds = query_output.last_hidden_state
        return image_embeds

    def encode_texts(self, texts: List[str], vlm_prompt_templates: List[str]) -> Tensor:
        text_inputs = self.build_text_inputs(texts, vlm_prompt_templates)
        with torch.no_grad():
            text_embeddings_list = []
            for text_input in text_inputs:
                text_embedding = self.extract_text_features(text_input)
                text_embeddings_list.append(text_embedding)
        text_embeddings = torch.stack(text_embeddings_list, dim=0)
        # remove dimension 1 -> (batch size, ?, 768)
        text_embeddings = text_embeddings.squeeze(1)
        return text_embeddings

    def extract_text_features(self, text: str) -> Tensor:
        """Returns an embedding for the given string.
        Implementation is based on feature extraction from
        https://github.com/salesforce/LAVIS/blob/f982acc73288408bceda2d35471a8fcf55aa04ca/lavis/models/blip2_models/blip2_qformer.py#L387
        """
        text_tokens = self.model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.model.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        text_output = self.model.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_embedding = text_output.last_hidden_state
        return text_embedding

    def get_modelname(self) -> str:
        return f"InstructBLIP-{self.frozen_text_encoder}"

    def __repr__(self) -> str:
        return self.get_modelname()

    def get_batchsize(self) -> int:
        return 64

    def get_image_transform(self):
        return self.transform
