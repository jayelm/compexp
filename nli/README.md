# Compositional Explanations of Neurons - NLI

This folder contains probing experiments for a neural model for natural language inference (NLI).

## Instructions

- Download pretrained SNLI model [here](http://downloads.cs.stanford.edu/nlp/data/muj/bowman_snli/6.pth) and store in directory `./models/bowman_snli/6.pth`. This is epoch 6 of an SNLI-trained model. Or train your own via `snli_train.py`.
- Probe model with settings located in `settings.py` with `code/analyze.py`
- This creates relevant HTML files in the experiment directory, which is
    printed out (`settings.RESULT`; e.g. `exp/snli_1.0_dev-6-sentence-5`) with
    the default settings.


## Settings

The settings are mostly documented, and many of them are similar to the vision
case, with some notable specifics:

- `ALPHA`: since we probe post-ReLU activations, `None` indicates that we simply threshold when neurons are positive; otherwise this is a percentile threshold.
- `N_SENTENCE_FEATS`: how many of the most common lemmas to keep across probing?
- `METRIC`: you can optimize for an explanation quality metric besides balanced IoU
- `MIN_ACTS`: minimum number of positive activations required (as defined by `ALPHA`) to analyze a neuron


## Additional scripts

- To construct all of the pieces need to probe SNLI data, we do the following,
    which you can modify for a different dataset:
    1. Running `data/tokenize_snli.py` on the raw development text to get
       whitespace-tokenized data
    2. Adding additional features (e.g. POS, entities, deps) via the
       `annotate.py` script as applied to the `.tok`
    3. Extracting word vectors (used for neighbors operator) via
       `data/analysis/extract_wordvecs.py`.

## Dependencies

Should be the same as those in the vision folder, along with:

- `spacy==2.2.4`
- `benepar` (optional; the [Berkeley neural parser](https://github.com/nikitakit/self-attentive-parser); used only to run `annotate.py` to annotate spacy corpus for dependency tags, though they aren't used in the paper analysis)
