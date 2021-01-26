import settings
from loader.model_loader import loadmodel
from dissection.neuron import hook_feature, NeuronOperator
from dissection import contrib
from visualize.report import (
    neuron as vneuron,
    final as vfinal,
    index as vindex,
)
from util.clean import clean
from util.misc import safe_layername
from tqdm import tqdm
from scipy.spatial import distance
import torch
#  import torch.nn.functional as F
import pickle
import os
import pandas as pd
from loader.data_loader import ade20k
from loader.data_loader import formula as F
import numpy as np
import pycocotools.mask as cmask
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from scipy import ndimage
import matplotlib.colors
from collections import defaultdict

HEADER = """
<!DOCTYPE html>
<html>
<title>Supplementary material</title>
<link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;700&family=Roboto+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
.card {
    padding: 10px;
    margin: 10px;
    width: 540px;
}
body {
    font-family: 'Open Sans', sans-serif;
}
p {
    margin-top: 5px;
    margin-bottom: 5px;
}
.description {
    font-weight: normal;
    font-size: 16px;
}
.unitname {
    font-weight: bold;
    font-size: 24px;
}
.length-1 {
    color: #C00000;
}
.length-3 {
    color: #4472C4;
}
.length-10 {
    color: #70AD47;
}
.roboto {
    font-family: 'Roboto Mono', monospace;
}
.example {
    width: 100px;
    height: 100px;
    padding: 5px;
}
.length {
    width: 90px;
    display: inline-block;
}
.iou {
    width: 90px;
    display: inline-block;
}
tr {
    vertical-align: top;
}
.labelabel:first-child {
    height: 25px;
}
.labelabel:nth-child(2) {
    height: 50px;
}
.labelabel:last-child {
    height: 125px;
}
img:first-child {
    padding-left: 0px;
}
img:last-child {
    padding-right: 0px;
}
</style>
<head>
</head>
<body>
"""

FOOTER = """
</body>
</html>
"""

MG = 1
BLUE_TINT = np.array([-MG * 255, -MG * 255, MG * 255])
PURPLE_TINT = np.array([MG * 255, -MG * 600, MG * 255])
RED_TINT = np.array([MG * 255, -MG * 255, -MG * 255])


# m -> meaningful
# u -> unmeaningful
# p -> polysemantic
# s -> specialization
# NOTE - difference between "specialization" with AND NOT is that AND NOT ...
# is within SPECIFIC parts...and not across DIFFERENT kinds of things.
DESC_KEY = {
    'm': 'lexical and perceptual',
    'u': 'perceptual only',
    's': 'specialization',
    'p': 'polysemantic',
    '': 'N/A'
}

DESC = {
    0: ('m', 'bars/surfaces'),
    1: ('m', 'windows/shelves'),
    2: ('m', 'islands/grass/water'),
    3: ('m', 'screens'),
    4: ('p', 'columns/chandeliers'),
    5: ('u', 'debris'),
    6: ('m', 'domes'),
    7: ('m', 'fields of plants'),
    8: ('m', 'plants'),
    9: ('p', 'grass/balls/other'),
    10: ('p', 'mountains/highway'),
    11: ('p', 'people/others'),
    12: ('p', 'dining rooms/fire escapes/others'),
    13: ('p', 'islands/canopies/others'),
    14: ('u', 'fences/horizontal lines'),
    15: ('u', 'beds'),
    16: ('p', 'gyms/windmills/other'),
    17: ('u', 'flat areas'),
    18: ('p', 'rocks/forests/other'),
    19: ('m', 'cases'),
    20: ('m', 'houses/decks'),
    21: ('p', 'bookcases/fire stations'),
    22: ('m', 'bridges, possibly over water'),
    23: ('u', 'vertical/perspective lines'),
    24: ('p', 'beds/fireplaces/other'),
    25: ('u', 'empty corridors'),
    26: ('m', 'aqueducts'),
    27: ('u', 'dome-like things'),
    28: ('m', 'alleys'),
    29: ('m', 'house facades'),
    30: ('m', 'porches'),
    31: ('p', 'pool tables/others'),
    32: ('u', 'red things'),
    33: ('m', 'landscapes/horizons'),
    34: ('p', 'beds and shelves'),
    35: ('p', 'water/other structures'),
    36: ('u', 'complex white structures'),
    37: ('u', 'empty halls/rooms'),
    38: ('u', 'things on grass'),
    39: ('u', 'flat surfaces'),
}


def clean_label(l):
    return l.replace('-s', '').replace('-c', '').replace('_', ' ').replace('-', ' ')


def format_labels(labels, ious):
    formatted = []
    for length, label, iou in zip(LENGTHS, labels, ious):
        formatted.append(
            f"<p class='labelabel'><span class='length'><strong>Length</strong> {length}</span><span class='iou'><strong>IoU</strong> {iou:.3f}</span><span class='label length-{length} roboto'>{label}</span></p>"
        )
    return ''.join(formatted)


def format_images(images):
    return ''.join(f"<img class='example' src='{iname}'>" for iname in images)


def format_desc(desc):
    typ, desc = desc
    typ = DESC_KEY[typ]
    return f"({typ}: {desc})"


def card(n, labels, ious, images, desc):
    card_html = f"""
    <div class="card">
        <div class="unitname">Unit {n} <span class='description'>{format_desc(desc)}</span></div>
        <div class="labels">{format_labels(labels, ious)}</div>
        <div class="images">{format_images(images)}</div>
    </div>
    """
    return card_html


def add_heatmap(img, acts, actmin=0.0, actmax=1.0):
    cmap = plt.cm.rainbow
    norm = matplotlib.colors.Normalize(vmin=actmin, vmax=actmax)
    act_colors = cmap(norm(acts))
    act_colors = act_colors[:, :, :3]
    # weighted combination of colors and mask
    weighted = (0.5 * act_colors * 255) + (0.5 * img)
    weighted = np.clip(weighted, 0, 255)
    weighted = np.round(weighted).astype(np.uint8)
    return weighted


def iou(m, n):
    intersection = np.logical_and(m, n).sum()
    union = np.logical_or(m, n).sum()
    return (intersection) / (union + 1e-10)


def iou_mask(img, neuron, mask):
    img = img.astype(np.int64)

    nborder = get_border(neuron)
    mborder = get_border(mask)
    border = np.logical_or(nborder, mborder)

    intersection = np.logical_and(neuron, mask)
    #  union = np.logical_xor(neuron, mask)

    int_mask = intersection[:, :, np.newaxis] * PURPLE_TINT
    int_mask = np.round(int_mask).astype(np.int64)

    neuron_mask = neuron[:, :, np.newaxis] * RED_TINT
    neuron_mask = np.round(neuron_mask).astype(np.int64)

    mask_mask = mask[:, :, np.newaxis] * BLUE_TINT
    mask_mask = np.round(mask_mask).astype(np.int64)

    img += neuron_mask + mask_mask + int_mask

    img[border] = (255, 255, 0)

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def noop(*args, **kwargs):
    return None


layernames = list(map(safe_layername, settings.FEATURE_NAMES))


hook_modules = []


model = loadmodel(hook_feature, hook_modules=hook_modules)
fo = NeuronOperator()

# ==== STEP 1: Feature extraction ====
# features: list of activations - one 63305 x c x h x w tensor for each feature
# layer (defined by settings.FEATURE_NAMES; default is just layer4)
# maxfeature: the maximum activation across the input map for each channel (e.g. for layer 4, there is a 7x7 input map; what's the max value). one 63305 x c tensor for each feature
features, maxfeature, preds, logits = fo.feature_extraction(model=model)

# ==== STEP 2: Threshold quantization ====
thresholds = [
    fo.quantile_threshold(lf, savepath=f"quantile_{ln}.npy")
    for lf, ln in zip(features, layernames)
]

# ==== New: multilayer case - neuron contributions ====
contrs_spread = [{} for _ in layernames + ["final"]]

# Zip it all together
ranger = tqdm(
    zip(
        layernames,
        features,
        maxfeature,
        thresholds,
        [None, *layernames],
        [None, *features],
        [None, *thresholds],
        contrs_spread,
    ),
    total=len(layernames),
)

tallies = []
# Load cached card htmls if they exist - will be overwritten if not skipping
# summarization step
all_card_htmls_fname = os.path.join(settings.OUTPUT_FOLDER, "card_htmls.pkl")
if os.path.exists(all_card_htmls_fname):
    print(f"Loading cached card htmls {all_card_htmls_fname}")
    with open(all_card_htmls_fname, "rb") as f:
        all_card_htmls = pickle.load(f)
else:
    all_card_htmls = {}


ds = fo.data


def upsample(actmap, shape):
    actmap_im_rsz = Image.fromarray(actmap).resize(shape, resample=Image.BILINEAR)
    actmap_rsz = np.array(actmap_im_rsz)
    return actmap_rsz


def get_border(x):
    x = x.astype(np.uint8)
    border = x - ndimage.morphology.binary_dilation(x)
    border = border.astype(np.bool)
    return border


def add_mask(img, mask):
    img = img.astype(np.int64)
    mask_border = get_border(mask)

    mask_weights = np.clip(mask.astype(np.float32), 0.25, 1.0)

    img_masked = img * mask_weights[:, :, np.newaxis]
    img_masked[mask_border] = (255, 255, 0)
    img_masked = np.clip(np.round(img_masked), 0, 255).astype(np.uint8)
    return img_masked


def friendly(mstr):
    return mstr.replace('(', '').replace(')', '').replace(' ', '_').replace(',', '')


LENGTHS = [1, 3, 10]
for (
    layername,
    layer_features,
    layer_maxfeature,
    layer_thresholds,
    prev_layername,
    prev_features,
    prev_thresholds,
    layer_contrs,
) in ranger:
    ranger.set_description(f"Layer {layername}")

    # ==== STEP 3: calculating IoU scores ====
    # Get tally dfname

    # Load each record
    EXPDIRS = [
        (1, 'result/complete_resnet18_places365_broden_ade20k_neuron_1/'),
        (3, 'result/complete_resnet18_places365_broden_ade20k_neuron_3/'),
        (10, 'result/complete_resnet18_places365_broden_ade20k_neuron_10/'),
    ]

    if settings.UNIT_RANGE is None:
        tally_dfname = f"tally_{layername}.csv"
    else:
        # Only use a subset of units
        tally_dfname = f"tally_{layername}_{min(settings.UNIT_RANGE)}_{max(settings.UNIT_RANGE)}.csv"

    results = defaultdict(dict)
    for length, expdir in EXPDIRS:
        tally_result, mc = fo.tally(
            layer_features, layer_thresholds, full_savepath=os.path.join(expdir, tally_dfname)
        )
        for record in tally_result:
            results[record['unit']][length] = record

    top = np.argsort(layer_maxfeature, 0)[: -1 -5 : -1, :].transpose()

    cards = []

    os.makedirs(os.path.join('suppl', 'img'), exist_ok=True)

    TOTAL = 40
    for N in range(TOTAL):
        #  visdir = os.path.join('suppl', f'vis_{N}')
        #  os.makedirs(visdir, exist_ok=True)

        # DO THE FULL UPSAMPLING OF THIS FEATURE MASK (this is very inefficient_)
        feats = layer_features[:, N]
        from tqdm import tqdm
        acts = np.stack([
            upsample(feats_i, (112, 112))
            for feats_i in tqdm(feats, desc='upsampling')
        ])
        neuron_masks = acts > layer_thresholds[N]

        # For each length, go through...

        record_labels = []
        record_ious = []
        for length, record in sorted(results[N].items(), key=lambda x: x[0]):
            rl = clean_label(record["label"])
            record_labels.append(rl)
            record_ious.append(float(record["score"]))

        # Neuron mask
        # Upsample
        # Get images
        record_imagenames = []
        for i, index in enumerate(tqdm(top[N], desc='topn')):
            # Most popular images
            imfn = ds.filename(index)
            img = np.array(Image.open(imfn))
            # Neuron activations - upsample

            # Here's your neuron mask
            acts_up = acts[index]
            acts_up = upsample(acts_up, img.shape[:2])

            # Find borders
            neuron_mask = acts_up > layer_thresholds[N]
            img_masked = add_mask(img, neuron_mask)
            new_imfn = os.path.join('img', f"{N}_{i}.jpg")
            Image.fromarray(img_masked).save(os.path.join('suppl', new_imfn))
            record_imagenames.append(new_imfn)

        try:
            desc = DESC[N]
        except KeyError:
            desc = ('', '')

        c = card(N, record_labels, record_ious, record_imagenames, desc)
        cards.append(c)
    cards_th = []
    for i in range(0, len(cards), 2):
        card1 = cards[i]
        try:
            card2 = cards[i + 1]
        except IndexError:
            card2 = ''
        cards_th.append(f"<tr><td>{card1}</td><td>{card2}</td></tr>")
    cards_th = f"<table class='cardtable'><tbody>{''.join(cards_th)}</tbody></table>"

    with open(os.path.join('suppl', 'index.html'), 'w') as f:
        f.write(HEADER)
        f.write(cards_th)
        f.write(FOOTER)
