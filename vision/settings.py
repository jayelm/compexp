"""
Configurable settings for probing
"""

GPU = True  # running on GPU is highly suggested
INDEX_FILE = "index_ade20k.csv"  # Which index file to use? If _sm, use test mode
CLEAN = False  # set to "True" if you want to clean the temporary large files after generating result
MODEL = "resnet18"  # model arch: resnet18, alexnet, resnet50, densenet161
DATASET = "places365"  # model trained on: places365 or imagenet. If None,use untrained resnet (random baseline)
MODEL_CHECKPOINT = None  # model training checkpoint. None if not used
MODEL_DATA_PERCENT = None  # model data percent (e.g. for places365). None if not used

PROBE_DATASET = "broden"  # currently only broden dataset is supported
QUANTILE = 0.005  # the threshold used for activation
TOTAL_QUANTILE = 0.01  # the threshold used for whole-image activation
SEG_THRESHOLD = 0.04  # the threshold used for visualization
SCORE_THRESHOLD = 0.04  # the threshold used for IoU score (in HTML file)
CONTRIBUTIONS = True  # If True, assume successive layers feed into each other; will use weights of layer i+1 to identify neurons contributing to layer i
TOPN = 5  # to show top N image with highest activation for each unit
PARALLEL = (
    8  # how many process is used for tallying
)
CATEGORIES = [
    "object",
    "part",
    "scene",
    "texture",
    "color",
]  # concept categories that are chosen to detect: "object", "part", "scene", "material", "texture", "color"
UNIT_RANGE = None  # Give range if you want to use only some units

# Formula summaries
EMBEDDING_SUMMARY = False  # Embedding-based summary
WN_SUMMARY = False  # Wordnet-based summary
WN_SIMILARITY = False  # Wordnet-based similarity
SEMANTIC_CONSISTENCY = False  # Evaluate semantic consistency of formulas

# Beam search params
FORMULA_COMPLEXITY_PENALTY = 1.00  # How much to downweight formulas by their length
BEAM_SEARCH_LIMIT = None  # (artificially) limit beam to this many candidates. If beam search takes a while, setting this to e.g. 50 can get you reasonably good explanations in much less time.
BEAM_SIZE = 5  # Size of the beam when doing formula search
MAX_FORMULA_LENGTH = 3  # Maximum compositional formula length

# Final layer vis params
TREE_MAXDEPTH = 4  # Index tree depth
TREE_MAXCHILDREN = 3  # Index tree max children
TREE_UNITS = range(1, 365, 10)  # How many units to build tree for

# ==== Ignore below - just some model/experiment directory naming logic ====
INDEX_SUFFIX = INDEX_FILE.split("index")[1].split(".csv")
if PROBE_DATASET != "broden" or not INDEX_SUFFIX:
    INDEX_SUFFIX = ""
else:
    INDEX_SUFFIX = INDEX_SUFFIX[0]

TEST_MODE = INDEX_FILE == "index_sm.csv"

mbase = MODEL
if MODEL_DATA_PERCENT is not None:
    mbase = f"{mbase}_{MODEL_DATA_PERCENT}pct"
if MODEL_CHECKPOINT is not None:
    mbase = f"{mbase}_{MODEL_CHECKPOINT}ckpt"

OUTPUT_FOLDER = f"result/{mbase}_{DATASET}_{PROBE_DATASET}{INDEX_SUFFIX}_neuron_{MAX_FORMULA_LENGTH}{'_test' if TEST_MODE else ''}{f'_checkpoint_{MODEL_CHECKPOINT}' if DATASET == 'ade20k' else ''}"

print(OUTPUT_FOLDER)
# ==== Ignore above - just some model/experiment directory naming logic ====


# ==== sub settings ====
# Most of these you can ignore - main one to change is FEATURE_NAMES which
# specifies the layers to be probed. Layers are specified as (possibly nested)
# accessors onto the model (e.g. ['layer4', '1', 'conv2'] will probe
# model.layer4[1].conv2).

if PROBE_DATASET == "broden":
    if MODEL != "alexnet":
        DATA_DIRECTORY = "dataset/broden1_224"
        IMG_SIZE = 224
    else:
        DATA_DIRECTORY = "dataset/broden1_227"
        IMG_SIZE = 227
else:
    raise NotImplementedError(f"Unknown dataset {PROBE_DATASET}")

if DATASET == "places365":
    NUM_CLASSES = 365
elif DATASET == "imagenet":
    NUM_CLASSES = 1000
elif DATASET == "ade20k":
    NUM_CLASSES = 365

if MODEL not in {"resnet18", "resnet50", "densenet161", "renset101", "alexnet", "vgg16"}:
    raise NotImplementedError(f"model = {MODEL}")

if MODEL == "resnet18":
    FEATURE_NAMES = [
        #  ['layer2', '0', 'conv1'], ['layer2', '0', 'conv2'],
        #  ['layer2', '1', 'conv1'], ['layer2', '1', 'conv2'],
        #  ['layer3', '0', 'conv1'], ['layer3', '0', 'conv2'],
        #  ['layer3', '1', 'conv1'], ['layer3', '1', 'conv2'],
        #  ['layer4', '0', 'conv1'], ['layer4', '0', 'conv2'],
        #  ["layer4", "1", "conv1"],
        #  ["layer4", "1", "conv2"],
        'layer4'
    ]
    #  FEATURE_NAMES = ['layer4']
elif MODEL == 'resnet50':
    FEATURE_NAMES = ['layer4']
elif MODEL == "resnet101":
    FEATURE_NAMES = ["layer4"]
elif MODEL == "densenet161":
    FEATURE_NAMES = ["features"]
elif MODEL == "alexnet":
    # Not sure...
    FEATURE_NAMES = ['features']
elif MODEL == "vgg16":
    # Not sure...
    FEATURE_NAMES = ["layer4"]

if DATASET == "places365":
    if MODEL_CHECKPOINT is None and MODEL_DATA_PERCENT is None:
        # Default places365 network
        if MODEL == 'densenet161':
            MODEL_FILE = 'zoo/whole_densenet161_places365_python36.pth.tar'
        else:
            MODEL_FILE = f"zoo/{MODEL}_places365.pth.tar"
    else:
        # Custom trained places365 network - model to load depends on (1)
        # percent of data trained on (determines folder); (2) checkpoint (e.g.
        # epoch, 'latest', 'best')
        datapctstr = f"_{MODEL_DATA_PERCENT}" if MODEL_DATA_PERCENT is not None else ""
        MODEL_FILE = f"zoo/trained/places365/resnet18{datapctstr}/resnet18_{MODEL_CHECKPOINT}.pth.tar"
    MODEL_PARALLEL = True
elif DATASET == "imagenet":
    MODEL_FILE = None
    MODEL_PARALLEL = False
elif DATASET == "ade20k":
    MODEL_FILE = f"zoo/trained/{mbase}_ade20k_finetune/{MODEL_CHECKPOINT}.pth"
    MODEL_PARALLEL = False
elif DATASET is None:
    MODEL_FILE = "<UNTRAINED>"
    MODEL_PARALLEL = False

if TEST_MODE:
    WORKERS = 1
    BATCH_SIZE = 4
    TALLY_BATCH_SIZE = 2
    TALLY_AHEAD = 1
else:
    WORKERS = 12
    BATCH_SIZE = 128
    TALLY_BATCH_SIZE = 16
    TALLY_AHEAD = 4
