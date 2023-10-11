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

    def infer_attrs(
        self, dset,
        llm_prompts: List[Tuple[str, str]]
    ) -> Dict[str, List[str]]:
        '''
        THIS IS DEPRECATED BUT I'M SCARED / TOO ATTACHED/SENTIMENTAL TO DELETE IT JUST YET.
        I'm sorry Mark I am not yet fully ruthless in my deleting old code.
        But officially, llm_prompts is dead. Long live llm_prompts. Attributer is in. 
        '''
        attrs_by_class = dict({classname: [] for classname in dset.classnames})
        for prompt_nickname, llm_prompt in llm_prompts:
            # Two cases that do not need the LLM: classname only (no attr info) or GT attrs
            if prompt_nickname == 'classname':
                # in this case, we add 'None' as an attribute for every class.
                # When forming a subpop description, the dataset objects will
                # know to simply return the classname when attribute is None.
                for classname in attrs_by_class:
                    attrs_by_class[classname].append(None)
            elif prompt_nickname == 'groundtruth':
                assert dset.has_gt_attrs, f"LLM prompt nickname 'groundtruth' cannot \
                be used for a dataset ({dset.get_dsetname()}) that is not attributed."
                for classname in attrs_by_class:
                    attrs_by_class[classname].extend(dset.gt_attrs_by_class(classname))
            # Otherwise, we ask the LLM for attributes.
            else:    
                # We cache LLM responses to avoid asking the LLM the same question
                # over and over, as it is expensive / takes a while.

                cache_path = os.path.join(
                    _CACHED_DATA_ROOT,
                    'subpops_from_llm',
                    prompt_nickname,
                    self.get_modelname(),
                    dset.get_dsetname()
                ) + '.pkl'

                if os.path.exists(cache_path):
                    dat = load_cached_data(cache_path)
                    assert llm_prompt == dat['llm_prompt'], "Attempting to use cached \
                    LLM responses. However, the exact LLM prompt differs from what is \
                    passed. This occurs when prompt_name is reused, but the associated \
                    full llm_prompt has changed. Either use a new prompt_name, change \
                    the name of the directory _CACHED_DATA_ROOT/subpops_from_llm/prompt_nickname, \
                    or delete that directory."
                    answers = dat['answers']
                else:
                    answers = dict({classname:self.answer_questions([llm_prompt.format(classname)])[0] 
                                        for classname in tqdm(dset.classnames)})
                    # We save the exact prompt as well, since the directory name is actually just the 
                    # prompt nickname, which is intended to be a one word summary of llm_prompt
                    save_dict = dict({'answers': answers, 'llm_prompt': llm_prompt})
                    cache_data(cache_path, save_dict)

                for classname, answer in answers.items():
                    attrs_in_answer = self.parse_answer(answer)
                    attrs_by_class[classname].extend(attrs_in_answer)

        return attrs_by_class

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

