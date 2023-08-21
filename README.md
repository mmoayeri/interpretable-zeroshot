# pretty-mmmd
XTREME MAKEOVER: repo edition. 

New cleaner version of mmmd repo, which allows for running the whole pipeline end to end. 

*WORK IN PROGRESS*

Tests:
`python -m pytest tests/`

# How to run pipeline end2end

main.py contains all the steps to the pipeline, as well as a function to run the entire thing.
Here are some details on the arguments needed to be passed to main() in main.py.

## Example set of arguments

Check out test_run_full_pipeline() in main.py. An example set of args is as follows:

```
args_as_dict = dict({
	'dsetname': 'dollarstreet__country.name',
	'vlm': 'blip2',
	'llm': 'vicuna-13b-v1.5',
	'attributer_keys': ['vanilla', 'income_level'],
	'vlm_prompt_dim_handler': 'stack_all', 
	'vlm_prompts': ['USE OPENAI IMAGENET TEMPLATES'],
	'predictor': 'interpol_sims_top_4',
	'lamb': 0.5
})
config = Config(args_as_dict)
output_dict = main(config)
```

This uses the Dollarstreet dataset with country.name as the groundtruth attribute (used in computing worst subpop acc). 
We also use both vanilla (i.e. classname only) and income_level (class-agnostic income descriptions) to generate attributes per class.
We use the openai templates to create texts from subpop captions, and we stack all of them instead of averaging (due to vlm_prompt_dim_handler).
Finally, we predict by interpolating w. lambda=0.5 (so just average) between the average of the top 4 sims per class and the average sim to everything in the class. 

## Running CHiLS

To do CHiLS, a few args need to be have the following specified vals:
- attributer_keys: ['vanilla', 'groundtruth'] or ['vanilla', 'llm_kinds_chils']
- vlm_prompt_dim_handler: 'average_and_norm_then_stack'
- predictor: 'chils'

It should throw an error if 'vanilla' is not in attributer_keys or if vlm_prompt_dim_handler is not 'average_and_norm_then_stack' while predictor is CHiLS.

## Arguments and allowable inputs
The below list shows the arguments needed w. their type in parens, followed by allowable choices.

<details><summary>Click to get full details on args and allowable inputs for each</summary>
- dsetname (str): 
	- Breeds dsets: `living17, entity30, entity13, nonliving26`
	- Dollarstreet: `dollarstreet__{region / country.name / income_group}`; second part says what to define as gt attr. Not super important.
	- GeoDE: `geode__{region / country.name}`; second part like in dollarstreet
	- MITStates: `mit_states__{thresh value}`; for now, just use 0.8 or 0.9 for thresh value. This determines how we filter classes.
- vlm (str): `clip_{mtype, e.g. ViT-B/16}` or `blip2`
- llm (str): `vicuna-13b-v1.5`
- attributer_keys (List[str]); Take a look at init_attributer in models/attributer.py to see all the choices. Some notes below
	- 'groundtruth' uses the gt_attrs_per_class per dataset
	- 'vanilla' is classname only
	- 'llm_{query}' uses LLM outputs following llm_query 'query' (e.g. llm_kinds --> answers to 'List diff kinds of {classname}'.)
	- 'income_level' returns a fixed set for all classes
- vlm_prompt_dim_handler (str): `average_and_norm_then_stack, stack_all, average_and_stack`; I'd only use the first two.
- vlm_prompts (List[str]): this is flexbile, but I'd really only use one of the three following:
	- `['a photo of a {}]`
	- `['USE OPENAI IMAGENET TEMPLATES']`
	- `['USE CONDENSED OPENAI TEMPLATES']`
- predictor (str): check out init_predictor in models/predictor.py for all options. Some notes below:
	- `average_top_k` will use whatever is after last '_' as k; same for `interpol_top_sims_k`
- lamb (float): only used for interpol_top_sims_k; I've never tried playing with this. 0.5 is probably fine
</details>

# BLIP


## BLIP2

Install `pip install salesforce-lavis`

For InstructBLIP special installation is required (see below).

## InstructBLIP

Installation:

`pip install -e /private/home/marksibrahim/Projects/Mazda/LAVIS/`

This will install a local version LAVIS with InstructBLIP with Vicuna7b. 

In case this does not work, you can do the following three lines:

```
git clone https://github.com/salesforce/LAVIS.git
cd LAVIS
pip install -e . 
```

Then, you'll want to copy Mark's config files for blip2_instruct_vicuna{7 or 13}b.yaml.
To do so, from the LAVIS directory, you can run 
`cp /private/home/marksibrahim/Projects/Mazda/LAVIS/lavis/configs/models/blip2/blip2_instruct_vicuna* lavis/configs/models/blip2/`

Note: currently, zero-shot classification w/ InstructBLIP is not great, so don't worry about this too much.

<details>
  <summary>Installation Details</summary>
	- vicuna-7b-v1.5: `/checkpoint/marksibrahim/models/mazda/vicuna-7b-v1.5/` (downloaded from Hugging Face using repo API)
	- install LAVIS locall and update path above following https://github.com/salesforce/LAVIS/tree/main/projects/instructblip
  
</details>

