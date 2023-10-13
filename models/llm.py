from abc import ABC, abstractmethod
from torch import Tensor
from constants import _CACHED_DATA_ROOT
from my_utils import cache_data, load_cached_data
from typing import Dict, List, Tuple
from fastchat.model import load_model, get_conversation_template, add_model_args
import torch
import os
from tqdm import tqdm

class LLM(ABC):

    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def get_modelname(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def answer_questions(self, questions: List[str]) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def parse_answer(self, answer: str) -> List[str]:
        # given single answer to a prompt, separate all individual attributes contained in the answer
        # E.g. asking 'list diff kinds of fox' may yield '1. Kit fox\n2. Arctic fox\n3. Red fox'
        # this function converts that answer string to ['Kit fox', 'Arctic fox', 'Red fox']
        raise NotImplementedError

class Vicuna(LLM):

    def __init__(self, model_key: str ='vicuna-13b-v1.5'):
        self.model_key = model_key
        # Loading the LLM takes a lot of space, so we don't unless we need to. 
        self.model = 'NOT YET LOADED'

    def set_up_model(self):
        self.model, self.tokenizer = load_model(f'lmsys/{self.model_key}', device='cuda', num_gpus=1)
        self.tokenizer.padding_side = 'left'
        # First, we provide a general initial prompt to the LLM.
        conv = get_conversation_template(f'lmsys/{self.model_key}')
        starter_prompt = conv.get_prompt()
        _ = self.answer_questions([starter_prompt]) 

    def get_modelname(self) -> str:
        return self.model_key

    def answer_questions(self, questions: List[str]) -> List[str]:
        if self.model == 'NOT YET LOADED':
            self.set_up_model()

        prompt_list = ['User: '+q+'\nAssistant:' for q in questions]
        input_ids = self.tokenizer(prompt_list, padding=True).input_ids
        input_tens = torch.tensor(input_ids).cuda()
        output_ids = self.model.generate(input_tens, do_sample=True, temperature=0.7, repetition_penalty=1, max_new_tokens=512)
        outputs = [self.tokenizer.decode(out_ids[len(in_ids):], skip_special_tokens=True, spaces_between_special_tokens=False) 
                    for in_ids, out_ids in zip(input_ids, output_ids)]
        return outputs

    def parse_answer(self, answer: str) -> List[str]:
        '''
        TODO: We should think more about this...

        Currently, this expects responses to be in the form of a numbered list. 
        So we expect answer to look like "1. Kit fox\n2. Arctic fox\n3. Red fox"
        '''
        # separate per line
        individual_answers = answer.split('\n')
        # remove leading numbers like '1. '
        # sometimes there is a period at the end of a response, we remove that as well
        # sometimes the llm says {axes of variance}: {instances} (e.g. Appearance: decorative); we just want the later part
        cleaned_answers = [ans.split('. ')[-1].split(':')[-1].strip().replace('.', '') for ans in individual_answers]
        return cleaned_answers

