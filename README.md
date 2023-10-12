# Japanese-oie

This repo contains the scripts for the paper "Improving Cross-Lingual Transfer for Open Information Extraction with Linguistic Feature Projection" accepted at MRL 2023.

## Requirements

```
pip install -r requirements.txt
```

The command might be an overkill, but should work for setting up the environment.
Apart from that, please make sure to install these [spacy](https://spacy.io/) packages:

```
python -m spacy download en_core_web_sm
python -m spacy download ja_core_news_sm
python -m spacy download de_core_news_sm
```

## Training data

same as [here](https://github.com/zhanjunlang/Span_OIE/tree/master).

## Pre-processing

As pre-processing, we need to first translate sentences in the training data into the target language using a machine translator, then perform word alignment.

### Machine Translation
- Pretrained model: [NLLB](https://github.com/facebookresearch/fairseq/tree/nllb/examples/nllb)
    - Please set-up the environment following instructions in the above repo.
- Script: 
    - `prepro_scripts/translate.sh`
- Usage:
    ```
    bash prepro_scripts/translate.sh  /path/to/nllb_checkpoint [src_lang] [tgt_lang] /path/to/input/file /path/to/fairseq/
    ```
- Input: Sentences in the source language.
    - Example: 
    ```
    this word , adjectival magavan meaning `` possessing maga - '' , was once the premise that avestan maga - and median magu - were co-eval .
    - - a hunter who uses bows and arrows instead of guns .
    ```
- Output: Sentences translated into the target language.
    - Example:
    ```
    Dieses Wort , adjektiv Magavan , das " Magas besitzen " bedeutet , war einst die Voraussetzung , dass Avestan Maga - und Median Magu - gleichzeitig waren .
    Ein Jäger , der Bogen und Pfeile anstelle von Waffen benutzt .
    ```
- Notes:
    - Need to first extract all sentences from the training data (in JSON format) to create a plain-text input file.
    - The script will return three files in current directory, with the suffix `_translated.hyp`, `_translated.true`, and `_translated.true.detok` respectively.
    - Use the file with suffix `_translated.true.detok` as the input for the steps hereafter.


### Word Alignment
- Pretrained model: [awesome-align](https://github.com/neulab/awesome-align)
    - Please set-up the environment following instructions in the above repo.
- Script:
    - `prepro_scripts/word_alignment.sh`
- Usage:
    ```
    bash prepro_scripts/word_alignment.sh /path/to/input/file /path/to/pretrained/model /path/to/output/file
    ```
- Input: Pre-tokenized parallel sentences pair of source-target languages, splitted by `|||`.
    - Example:
    ```
    - - a hunter who uses bows and arrows instead of guns . ||| - Ein Jäger , der Bogen und Pfeile anstelle von Waffen benutzt .
    ```
- Output: Token-level alignment between sentences in the source language and the target language.
    - Example:
    ```
    0-1 4-4 9-10 1-2 10-11 3-4 0-0 2-3 6-7 8-9 12-12 5-6 11-5 7-8
    ```
- Notes:
    - Each row of the input file should be formatted as `{source language sentence} ||| {target language sentence}`.
    - Both the source and the target sentences should be *tokenized* and *whitespace-splitted*.



## Lingustic Feature Projection (LFP)


- Scripts:
    - `lfp_policies.py`: main script for performing LFP
    - `lfp_utis.py`: some util functions used in `lfp_policies.py`
- Usage:
    ``` 
    python lfp_policies.py --lang [de|ar|ja] \
    --name [identifier_of_generated_data] \
    [--ro] [--cs] [--cm] \   
    --output_dir /path/to/save/outputs \
    --base_data /path/to/training_data \
    --src_sents /path/to/source_sents \
    --tgt_sents /path/to/target_sents \
    --align /path/to/alignments 
    ```
- Inputs:
    - Training data for OpenIE;
    - Plain text file of sentences in the training data;
    - Plain text file of sentences in the training data, translated into the target language;
    - Alignment between tokens in source and target sentences;
- Output: Training data generated based on given LFP strategies in `JSON` format.
- Notes:
    - `--lang` and `--name` are mandatory.
    - Please make sure the tokenizer for LFP is **exactly the same one** as for word alignment (i.e., the same pre-trained model provided by SpaCy), otherwise an indexing error will be raised.
    - If `--align` is not given, the script will generate data for alignment and exit. The generated data can be directly fed into the awesome aligner. 

We have implemented three LFP strategies here: word reordering (`--ro`), code-switching (`--cs`), and inserting case markers (`--cm`, for Japanese only). 

## Training OpenIE Models

The output of `lfp_policies.py` can be directly used to train OpenIE systems. To reproduce results in the paper, please train [MILIE](https://aclanthology.org/2022.acl-long.478/) on the data using hyper-parameters introduced in the paper.

## Evaluating OpenIE Models

We include Arabic and Japanese BenchIE data in `/benchie_extended/` for evaluating OpenIE systems in these languages.

Please see the repo of [BenchIE](https://github.com/gkiril/benchie) for more details about the evaluation.