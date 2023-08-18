# pretty-mmmd
XTREME MAKEOVER: repo edition. 

New cleaner version of mmmd repo, which allows for running the whole pipeline end to end. 

*WORK IN PROGRESS*

Tests:
`python -m pytest tests/`

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

