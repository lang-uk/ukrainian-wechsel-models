# ukrainian-nlp

Code for the [WECHSEL](https://aclanthology.org/2022.naacl-main.293/) models transferred to Ukrainian:

- https://huggingface.co/benjamin/roberta-base-wechsel-ukrainian
- https://huggingface.co/benjamin/roberta-large-wechsel-ukrainian
- https://huggingface.co/benjamin/gpt2-wechsel-ukrainian
- https://huggingface.co/benjamin/gpt2-large-wechsel-ukrainian


## Dictionaries
 - `extra_dicts/ukrainian_wiktionary.txt` — updated dict, parsed from wiktionary as of 20.05.2023. Used for the `configs/experimental/gpt2/gpt2-small.oscar.nofilter.wechsel.mediumdict.json` config
 - `extra_dicts/ukrainian_stardict.txt` — English - Ukrainian dictionary for Android • NerdCats, converted with pyglossary and aux script `aux/convert_after_pygloss.py`. Used for the `configs/experimental/gpt2/gpt2-small.oscar.nofilter.wechsel.largedict.json` config


## Credits
The part of the work in this study is done on the hardware of the Ukrainian cluster of excellence of the Ukrainian Catholic University
The bigger models are trained with the support from Google TRC program
