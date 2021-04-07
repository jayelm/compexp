"""
Settings
"""

import os

CUDA = True
ALPHA = None  # Use "None" to use ReLU threshold (i.e., > 0)
BEAM_SIZE = 10
MAX_FORMULA_LENGTH = 5
COMPLEXITY_PENALTY = 1.00
TOPN = 5
DEBUG = False

# Choices: iou, precision, recall
METRIC = "iou"

EMBEDDING_NEIGHBORHOOD_SIZE = 5

NEURONS = None
PARALLEL = 4

SHUFFLE = False
SAVE_EVERY = 4

# How many "maximally activating" open features to use, PER CATEGORY
MAX_OPEN_FEATS = 5
# Minimum number of activations to analyze a neuron
MIN_ACTS = 500

MODEL = "models/bowman_snli/6.pth"
MODEL_TYPE = "bowman"  # choices: bowman, minimal
RANDOM_WEIGHTS = False  # Initialize weights randomly (equivalent to an untrained model)
N_SENTENCE_FEATS = 2000  # how many of the most common sentence lemmas to keep

DATA = "data/analysis/snli_1.0_dev.feats"

assert DATA.endswith(".feats")
VECPATH = DATA.replace(".feats", ".vec")

# Overridables
if "MTDISSECT_MODEL" in os.environ:
    MODEL = os.environ["MTDISSECT_MODEL"]
if "MTDISSECT_MAX_FORMULA_LENGTH" in os.environ:
    MAX_FORMULA_LENGTH = int(os.environ["MTDISSECT_MAX_FORMULA_LENGTH"])
if "MTDISSECT_MAX_OPEN_FEATS" in os.environ:
    MAX_OPEN_FEATS = int(os.environ["MTDISSECT_MAX_OPEN_FEATS"])
if "MTDISSECT_METRIC" in os.environ:
    METRIC = os.environ["MTDISSECT_METRIC"]

mbase = os.path.splitext(os.path.basename(MODEL))[0]
dbase = os.path.splitext(os.path.basename(DATA))[0]
RESULT = f"exp/{dbase}-{mbase}-sentence-{MAX_FORMULA_LENGTH}{'-shuffled' if SHUFFLE else ''}{'-debug' if DEBUG else ''}{f'-{METRIC}' if METRIC != 'iou' else ''}{f'-random-weights' if RANDOM_WEIGHTS else ''}"

print(RESULT)
