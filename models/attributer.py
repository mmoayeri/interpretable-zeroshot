from abc import ABC, abstractmethod
from typing import Dict, List
from models.llm import LLM
import os
from constants import _CACHED_DATA_ROOT, _IMAGENET_OPENAI_TEMPLATES, _CONDENSED_OPENAI_TEMPLATES
from my_utils import load_cached_data
from datasets import ClassificationDset

class Attributer(ABC):
    '''
    BASE CLASS FOR ATTRIBUTERS
    Each attributer infers a list of attributes per classname.
    Input always contains a list of classnames.

    There other inputs specifc to certain types of attributer.
      - For LLM inferred attributes, we also pass the LLM and an LLMQuery object
        that explains what we are asking and how to form captions. 
        See LLMQuery class(es) and LLMBased(Attributer) class below. 

      - For Groundtruth attributes, we also pass a dataset (that has gt attrs)
        See GroundTruths(Azttributer) class below. 

    In practice, we use the subpop_captions_by_class function for each 
    attributer, which returns a dictionary of (classname, list of subpop descriptions)
    key-value pairs (e.g. ('fox', 'arctic fox, a kind of fox'))

    Check attribute() function below to see how attribute inference 
    is handled in end-to-end flow. 
    '''

    @abstractmethod
    def infer_attrs(self, classnames: List[str]) -> Dict[str, List[str]]:
        # Given list of classnames, return attrs_by_class dict
        raise NotImplementedError

    @abstractmethod
    def caption_subpop(self, classname: str, attr: str) -> str:
        '''
        Convert (classname, attr) pair into single string caption
        e.g. ('fox', 'arctic fox') -> 'arctic fox, a kind of fox'
        e.g. ('stove', 'Burundi') -> 'a stove from the country Burundi'
        e.g. ('stove', 'Asia') -> ' a stove from the region Asia'
        '''
        raise NotImplementedError

    def subpop_captions_by_class(self, classnames: List[str]) -> Dict[str, List[str]]:
        attrs_by_class = self.infer_attrs(classnames)
        return dict({
            classname: [self.caption_subpop(classname, attr) for attr in attrs]
                for classname, attrs in attrs_by_class.items()
        })


def infer_attrs(
    classnames: List[str], 
    attributers: List[Attributer], 
    vlm_prompt_templates: List[str]
    ) -> Dict[str, Dict[str, List[str]]]:
    '''
    PRIMARY FUNCTION used in main.py.
    Given a list of attributer objects, a list of classnames, and a list of vlm_prompts,
    returns a dictionary with key per classname, and a *dictionary* as value.
    That dictionary has subpopulation description as the key, and list of templated subpops
    (determined by the vlm_prompts) as the value.

    e.g. output dict may have a key of 'dog' with the value:
    dict({'a labroader, a kind of dog': ['a photo of a labroader, a kind of dog', 'a drawing of a labroader, a kind of dog'],
          'a pekingese, a kind of dog': ['a photo of a pekingese, a kind of dog', 'a drawing of a pekingese, a kind of dog']})
    in the above example, vlm_prompts would be ['a photo of a {}', 'a drawing of a {}']
    '''
    subpops_by_class = dict({classname: [] for classname in classnames})
    for attributer in attributers:
        curr_subpops_by_class = attributer.subpop_captions_by_class(classnames)
        for classname, subpops in curr_subpops_by_class.items():
            subpops_by_class[classname].extend(subpops)
        
    subpops_by_class = dict({classname:list(set(subpops)) for classname, subpops in subpops_by_class.items()})

    # replace some key word mome
    if vlm_prompt_templates == ['USE OPENAI IMAGENET TEMPLATES']:
        vlm_prompt_templates = _IMAGENET_OPENAI_TEMPLATES
    elif vlm_prompt_templates == ['USE CONDENSED OPENAI TEMPLATES']:
        vlm_prompt_templates = _CONDENSED_OPENAI_TEMPLATES

    # Now we build the 3D structure, allowing for averaging over vlm_prompts or not
    texts_by_subpop_by_class = dict({
        classname: dict({
            subpop_caption: [template.format(subpop_caption) for template in vlm_prompt_templates]
                for subpop_caption in subpops
        }) for classname, subpops in subpops_by_class.items()
    })
    return texts_by_subpop_by_class

class ClassnameOnly(Attributer):
    # Only returns classname
    def infer_attrs(self, classnames: List[str]) -> Dict[str, List[str]]:
        # Empty string here is never used
        return dict({classname: [''] for classname in classnames})

    def caption_subpop(self, classname: str, attr: str) -> str:
        return classname

class Regions(Attributer):
    '''
    Using UN geoscheme: https://en.wikipedia.org/wiki/United_Nations_geoscheme
    regions_by_continent = dict({
    'Africa': ['Northern Africa', 'Sub-Saharan Africa', 'Eastern Africa', 'Middle Africa', 'Southern Africa', 'Western Africa'],
    'Americas': ['Caribbean', 'Central America', 'South America', 'Northern America'],
    'Asia': ['Central Asia', 'Eastern Asia', 'South-eastern Asia', 'Southern Asia','Western Asia'],
    'Europe':['Eastern Europe (including Northern Asia)', 'Northern Europe', 'Channel Islands', 'Southern Europe', 'Western Europe'],
    'Oceania':['Australia and New Zealand','Melanesia', 'Micronesia', 'Polynesia']
    })
    '''

    ''' Using World Bank regions: https://datatopics.worldbank.org/sdgatlas/archive/2017/the-world-by-region.html '''
    def __init__(self):
        self.regions = ['East Asia and Pacific', 'Europe and Central Asia',
                        'Latin America and Caribbean', 'Middle East and North Africa',
                        'North America', 'South Asia', 'Sub-Saharan Africa']

    def infer_attrs(self, classnames: List[str]) -> Dict[str, List[str]]:
        return dict({classname: self.regions for classname in classnames})
        
    def caption_subpop(self, classname: str, attr: str) -> str:
        return f'{classname} from the region {attr}'


class IncomeLevels(Attributer):
    def __init__(self):
        self.income_levels = ['poor', 'lower middle class', 'upper middle class', 'rich']

    def infer_attrs(self, classnames: List[str]) -> Dict[str, List[str]]:
        return dict({classname: self.income_levels for classname in classnames})
        
    def caption_subpop(self, classname: str, attr: str) -> str:
        return f'{classname} from a {attr} country'

class Countries(Attributer):
    '''
    From ChatGPT, asking 'Present the five most populous countries from each continent in the 
    form of a python dictionary', yields:

    populous_countries = {
    'Africa': ['Nigeria', 'Ethiopia', 'Egypt', 'Democratic Republic of the Congo', 'South Africa'],
    'Asia': ['China', 'India', 'Indonesia', 'Pakistan', 'Bangladesh'],
    'Europe': ['Russia', 'Germany', 'United Kingdom', 'France', 'Italy'],
    'North America': ['United States', 'Mexico', 'Canada', 'Guatemala', 'Cuba'],
    'Oceania': ['Australia', 'Papua New Guinea', 'New Zealand', 'Fiji', 'Solomon Islands']
    }
    '''

    def __init__(self):
        self.countries = [
            'Nigeria', 'Ethiopia', 'Egypt', 'Democratic Republic of the Congo', 
            'South Africa', 'China', 'India', 'Indonesia', 'Pakistan', 'Bangladesh', 
            'Russia', 'Germany', 'United Kingdom', 'France', 'Italy', 'United States', 
            'Mexico', 'Canada', 'Guatemala', 'Cuba', 'Australia', 'Papua New Guinea', 
            'New Zealand', 'Fiji', 'Solomon Islands']

    def infer_attrs(self, classnames: List[str]) -> Dict[str, List[str]]:
        return dict({classname: self.countries for classname in classnames})
        
    def caption_subpop(self, classname: str, attr: str) -> str:
        return f'{classname} from the country {attr}'

class GroundTruths(Attributer):
    def __init__(self, dset: ClassificationDset):
        assert dset.has_gt_attrs, f"{dset.dsetname} does not have ground truth attributes, \
        so the GroundTruths Attributer cannot be used."
        self.dset = dset

    def infer_attrs(self, classnames: List[str]) -> Dict[str, List[str]]:
        return dict({classname: self.dset.gt_attrs_by_class(classname) for classname in classnames})
        
    def caption_subpop(self, classname: str, attr: str) -> str:
        '''
        Note that for dsets with gt attrs, we expect them to also provide
        a method for captioning subpops formed by their gt attrs, which we 
        use here. 
        '''
        return self.dset.caption_gt_subpop(classname, attr)

class LLMQuery(ABC):
    '''
    Simple config class to clean up how we handle LLM queries. 
    Namely, we'll need to establish the full question we ask the 
    LLM (e.g. 'List different kinds of {classname}') and the nickname
    that we use in caching the LLM responses to a specific question
    (to avoid having to repeat this step over and over).

    Importantly, each query also has its own caption_subpop, which
    combines a classname and attribute to a single string describing
    the subpopulation defined by (classname, attr), exactly as in 
    Attributer above. 
    '''
    def __init__(self, nickname: str, question: str) -> None:
        self.nickname = nickname
        self.question = question

    @abstractmethod
    def caption_subpop(self, classname: str, attr: str) -> str:
        raise NotImplementedError


class KindsQuery(LLMQuery):
    def __init__(self):
        super().__init__(
            nickname='kinds', 
            question='List 16 different kinds of {}. Only use up to three words per list item.'
        )

    def caption_subpop(self, classname: str, attr: str) -> str:
        return f'{attr}, a kind of {classname}'

class KindsRegionsIncomesQuery(LLMQuery):
    def __init__(self):
        super().__init__(
            nickname='kinds_regions_incomes', 
            question='List 16 different ways in which a {} may appear across diverse geographic regions and incomes. Only use up to three words per list item.'
        )

    def caption_subpop(self, classname: str, attr: str) -> str:
        return f'{attr}, a kind of {classname}'


class LLMBased(Attributer):
    def __init__(self, llm: LLM, llm_query: LLMQuery, cache_fname: str):
        # Note: cache_fname is typically just the dataset name
        self.llm = llm
        self.llm_query = llm_query
        self.cache_fname = cache_fname

    def infer_attrs(self, classnames: List[str]) -> Dict[str, List[str]]:
        attrs_by_class = dict({classname: [] for classname in classnames})
        cache_path = os.path.join(
            _CACHED_DATA_ROOT,
            'subpops_from_llm',
            self.llm_query.nickname,
            self.llm.get_modelname(),
            self.cache_fname # dset.get_dsetname()
        ) + '.pkl'

        if os.path.exists(cache_path):
            dat = load_cached_data(cache_path)
            assert self.llm_query.question == dat['llm_prompt'], "Attempting to use cached \
            LLM responses. However, the exact LLM prompt differs from what is \
            passed. This occurs when prompt_name is reused, but the associated \
            full llm_prompt has changed. Either use a new prompt_name, change \
            the name of the directory _CACHED_DATA_ROOT/subpops_from_llm/prompt_nickname, \
            or delete that directory."
            answers = dat['answers']
        else:
            answers = dict({classname:self.llm.answer_questions([self.llm_query.question.format(classname)])[0] 
                                for classname in tqdm(classnames)})
            # We save the exact prompt as well, since the directory name is actually just the 
            # prompt nickname, which is intended to be a one word summary of llm_prompt
            save_dict = dict({'answers': answers, 'llm_prompt': self.llm_query.question})
            cache_data(cache_path, save_dict)

        return dict({classname: self.llm.parse_answer(answers[classname]) 
                        for classname in classnames})

    def caption_subpop(self, classname: str, attr: str):
        return self.llm_query.caption_subpop(classname, attr)

def init_attributer(key: str, dset: ClassificationDset, llm: LLM) -> Attributer:
    """
    I'm hiding this ugly if block here to avoid having a ton of imports to main.
    Maybe I can do this for all my ugly if blocks in main? Not super important.
    
    I'm not happy that I need to have arguments for dset and llm when we only
    sometimes use them. Maybe better if we have like an init file that has imports
    from everything and has a bunch of these init_attributer/dataset/predictor type fns. 
    Again not very important.
    """
    # Baseline attributers
    if key == 'vanilla':
        attributer = ClassnameOnly()
    elif key == 'ground_truth':
        attributer = GroundTruths(dset)
    # 'Global info' (i.e. class-agnostic) level attributers
    elif key == 'region':
        attributer = Regions()
    elif key == 'country':
        attributer = Countries()
    elif key == 'income_level':
        attributer = IncomeLevels()
    # LLM based attributers
    elif 'llm_' in key:
        query_key = attributer.split('llm_')[-1]
        if query_key == 'kinds':
            query = KindsQuery()
        elif query_key == 'kinds_regions_incomes':
            query = KindsRegionsIncomesQuery()
        else:
            raise ValueError(f'Query key {query_key} for LLM based attributer (with key {key}) not recognized.')
        attributer = LLMBased(llm=llm, llm_query=query, cache_fname=dset.get_dsetname())
    else:
        raise ValueError(f'Attributer key {key} not recognized.')
    return attributer