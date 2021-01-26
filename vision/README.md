# Compositional Explanations of Neurons - Vision

This folder contains convolutional neural network probing experiments for image classification.

## Probing Instructions

1. Download Broden dataset with `./script/dlbroden.sh` (You can comment out downloading 227 if you're not going to probe alexnet)
2. Run `./script/dlzoo_example.sh` to download (as an initial example) `resnet18` trained on `places365`.
3. Configure settings in `settings.py`. All settings are documented, but there are some obvious settings to change which I describe below.
4. Run `python probe.py` to generate results in `result/` (the output folder name is printed when you run).

The `result/` directory contains the following files:
- `tally_{layername}.csv`: - each row is a unit, with label, category,
    non-compositional labels and categories, length, iou scores, and optional
    summary statistics if computed (`{emb,wn}_summary_sim`; see below)
- `preds.csv`: csv file - each row is an image in broden, with predicted class,
    target class, and numeric variables indicating whether the `n`th neuron
    was active (exceeding `settings.SCORE_THRESHOLD`)
- `html/{layername}.html`: visualization of units for this layer.
- `html/final.html`: final classes with units that most contribute to the
    prediction. Produced only if `settings.CONTRIBUTIONS` is `True`.

## Settings

Here are some settings that matter. The default settings (length 3 formula, 8
processors) takes about 14 hours to run. Main ways to reduce computation time
are to (1) reduce formula length (`MAX_FORMULA_LENGTH`), (2) increase
processors (`PARALLEL`), (3) probe a subset of units (`UNIT_RANGE`), and (4)
restrict beam search (`BEAM_SEARCH_LIMIT`)

1. `MODEL`: the specific model to probe (currently supports resnet18, alexnet, resnet50, densenet161)
2. `PARALLEL`: how many processes to use (longer formula lengths take a while, and multiprocessing will make things faster)
3. `MAX_FORMULA_LENGTH`: the maximum formula length (default is 3)
4. `BEAM_SIZE`: size of the beam during beam search
5. `UNIT_RANGE`: only probe a subset of units (e.g. specify range(10) to probe only the first 10 units)
6. `FEATURE_NAMES`: (model specific) the layer to probe. Look at the conditional logic at the end of the file, and change the feature names for the model you want. They refer to model submodules, and can either be strings or nested tuples of accessors (i.e. `['layer2', '0', 'conv1']` will retrive `model.layer2[0].conv1`). If you want to probe subsequent layers for `resnet18` for example, note that `layer4` is actually composed of several convolutional layers (see commented out lines of `FEATURE_NAMES` for an example)
7. `CONTRIBUTIONS`: if set, treats the subsequent layers of `FEATURE_NAMES` as feeding into each other, and computes contributions between intermediate layers (see caveat about subsequent layers in feature names). Also assumes that the last probed layer feeds into final predictions, and generates a final class prediction summary with neurons that most contribute to a specific class.
8. Formula summaries: some experimental support for summarizing and measuring the semantic "consistency" of explanations that didn't make it into the final paper. Summaries include `EMBEDDING_SUMMARY`: find the word embedding closest to the primitive concepts in an explanation and `WN_SUMMARY`: find the word closest to the primitive concepts as measured by WordNet graph distance. Consistency measures include `WN_SIMILARITY`: average normalized WordNet similarity between primitives in an explanation (higher is better) and `SEMANTIC_CONSISTENCY`: average embedding similarity. WordNet requires `nltk`, embedding measures require `spacy`.
9. `BEAM_SEARCH_LIMIT`: For longer formula lengths/beam sizes, this procedure can take a while - you can limit the new candidates to be considered at each step of beam search by setting this to a smaller number e.g. 50 or 100. (Otherwise all 1k+ candidates are considered to be added to each of the existing formulas at each step of beam search). This makes search faster and still results in pretty good explanations.
10. `FORMULA_COMPLEXITY_PENALTY`: you can add a multiplicative penalty on IoU score for longer formulas, if you want a trade-off between formula complexity and expressivity. E.g. setting this penalty to 0.90 means that IoU scores for formula `F` are calculated as `iou(F) * (0.90 ** (len(F) - 1))`.
11. `DATASET`: this is the dataset the model is *trained on* - the default `places365` requires models downloaded from MIT via `script/dl*.sh` scripts. Set to `imagenet` to use imagenet-pretrained torchvision models (if available); set to `None` to probe a random baseline.

## Other scripts

- `eval.py` lets you evaluate model predictions (based on `settings.py`) on
    certain images (e.g.\ the ones in the adversarial examples folder).
- `eval_size_position.py` runs the size/position of adversarial examples
  subimages experiment in the supplementary material.

## Dependencies

This repository has been tested on python 3.7 with the following packages (though it should likely work with newer versions):

- `pyeda`
- `torch==1.4.0`
- `torchvision==0.4.2`
- `pandas==0.25.3`
- `tqdm==4.30.0`
- `imageio==2.6.1`
- `scipy==1.4.1`
- `matplotlib==3.1.3`
- `Pillow==7.0.0`
- `seaborn==0.9.0`
- `scikit-image==0.16.2`
- `pyparsing==2.4.6`
- `pyeda==0.28.0`
- `nltk==3.3` (optional)
- `spacy==2.2.4` (optional)
- `pycocotools==2.0` CUSTOM FORK

where the last package is my custom fork of the [COCO
API](https://github.com/jayelm/cocoapi) that supports mask negation to compute
`NOT(x)` concepts. (If the original API eventually adds `NOT` compatibility
(e.g. https://github.com/cocodataset/cocoapi/pull/387),
raise an issue and I can change the code to remove this dependency).

You can install this version by cloning [`jayelm/cocoapi`](https://github.com/jayelm/cocoapi) and running `make` in
the `PythonAPI` directory.

With `conda`, you can install all required packages from the supplied
`environment.yml`, *except* for pytorch (whose install is heavily
machine-dependent) and the custom fork of COCO.
